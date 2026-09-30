"""Unit tests for Plant Disease Detection GPU training and data pipeline."""
import os
import sys
import unittest
import yaml
import numpy as np
from PIL import Image

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.prepare_data import UnionFind, compute_dhash, extract_leaf_id

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from training.dataset import ManifestDataset
    from training.engine import fit, run_epoch
    from training.utils import EarlyStopping, build_transforms, clear_gpu_memory, get_gpu_memory_info, setup_gpu
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TestDataPreparationAndAntiLeakage(unittest.TestCase):
    """Tests for data prep, perceptual hashing, and cross-split leakage prevention."""

    def test_leaf_series_regex(self):
        """Verify leaf-series regex correctly matches physical leaf tracking IDs."""
        self.assertEqual(extract_leaf_id("123___GH_HL Leaf 434.JPG"), "gh_hl leaf 434")
        self.assertEqual(extract_leaf_id("abc___GHLB_PS Leaf 24 Day 13.jpg"), "ghlb_ps leaf 24")
        self.assertEqual(extract_leaf_id("xyz___GHLB2 Leaf 102.JPG"), "ghlb2 leaf 102")
        self.assertIsNone(extract_leaf_id("random_leaf_image.jpg"))

    def test_union_find_clustering(self):
        """Test Disjoint Set Union correctly clusters connected duplicate pairs."""
        items = ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg"]
        uf = UnionFind(items)
        uf.union("img1.jpg", "img2.jpg")
        uf.union("img2.jpg", "img3.jpg")

        groups = uf.get_groups()
        # img1, img2, img3 should be in one cluster, img4 alone
        clustered = [sorted(g) for g in groups]
        self.assertIn(["img1.jpg", "img2.jpg", "img3.jpg"], clustered)
        self.assertIn(["img4.jpg"], clustered)

    def test_dhash_synthetic(self):
        """Test perceptual dHash produces deterministic 64-bit integer."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            img = Image.new("RGB", (64, 64), color=(120, 150, 180))
            img.save(tmp.name)
            h1 = compute_dhash(tmp.name)
            h2 = compute_dhash(tmp.name)
            self.assertIsNotNone(h1)
            self.assertEqual(h1, h2)

    def test_compute_binary_roc(self):
        """Test ROC curve FPR, TPR, and AUC calculation."""
        from training.evaluate import compute_binary_roc
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        fpr, tpr, auc = compute_binary_roc(y_true, y_score)
        self.assertEqual(auc, 1.0)
        self.assertEqual(len(fpr), len(tpr))

    def test_manifest_leakage_audit(self):

        """Verify generated manifest exists and has 0 cross-split leaks."""
        manifest_path = "data/processed/manifest.csv"
        if not os.path.exists(manifest_path):
            self.skipTest("manifest.csv not yet generated")

        import csv
        import hashlib
        from collections import defaultdict

        md5_splits = defaultdict(set)
        leaf_splits = defaultdict(set)

        with open(manifest_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                fpath = row["filepath"]
                split = row["split"]
                if os.path.exists(fpath):
                    with open(fpath, "rb") as fh:
                        h = hashlib.md5(fh.read()).hexdigest()
                    md5_splits[h].add(split)

                lid = extract_leaf_id(os.path.basename(fpath))
                if lid:
                    leaf_splits[lid].add(split)

        # Confirm 0 cross-split leakage
        leaked_md5 = [k for k, s in md5_splits.items() if len(s) > 1]
        leaked_leaves = [k for k, s in leaf_splits.items() if len(s) > 1]
        self.assertEqual(len(leaked_md5), 0, f"Found leaked MD5 duplicates across splits: {leaked_md5}")
        self.assertEqual(len(leaked_leaves), 0, f"Found leaked leaf series across splits: {leaked_leaves}")


class TestTrainingConfig(unittest.TestCase):
    def setUp(self):
        self.config_path = "config/training_config.yaml"

    def test_config_structure(self):
        """Verify training_config.yaml has required GPU, model, and regularization keys."""
        self.assertTrue(os.path.exists(self.config_path), f"Missing {self.config_path}")
        with open(self.config_path) as f:
            cfg = yaml.safe_load(f)

        self.assertIn("crop_classifier", cfg)
        self.assertIn("disease_classifier", cfg)
        self.assertIn("batch_size", cfg)
        self.assertIn("img_size", cfg)
        self.assertIn("label_smoothing", cfg)
        self.assertIn("drop_rate", cfg)
        self.assertIn("early_stopping_metric", cfg)
        self.assertEqual(cfg["img_size"], 224)
        self.assertEqual(cfg["early_stopping_metric"], "val_loss")
        self.assertIn(cfg["crop_classifier"]["model_name"], ["vit_base_patch16_224", "efficientnet_b0"])
        self.assertEqual(cfg["disease_classifier"]["model_name"], "efficientnet_b0")



class TestTorchPipeline(unittest.TestCase):
    """Tests requiring PyTorch."""

    def setUp(self):
        if not TORCH_AVAILABLE:
            self.skipTest("PyTorch is not installed in this environment (run on GPU machine).")

    def test_build_transforms(self):
        """Verify train (strong and standard) and eval transforms produce expected shapes."""
        train_tf_strong, eval_tf = build_transforms(224, strong_aug=True)
        train_tf_std, _ = build_transforms(224, strong_aug=False)
        dummy_img = Image.new("RGB", (300, 300), color=(100, 150, 200))

        train_out = train_tf_strong(dummy_img)
        eval_out = eval_tf(dummy_img)

        self.assertEqual(train_out.shape, (3, 224, 224))
        self.assertEqual(eval_out.shape, (3, 224, 224))
        self.assertEqual(train_out.dtype, torch.float32)

    def test_early_stopping_min_mode(self):
        """Test EarlyStopping in 'min' mode (loss tracking) with patience."""
        stopper = EarlyStopping(patience=3, mode="min", min_delta=1e-3)
        self.assertFalse(stopper.should_stop)

        # Loss decreases (improvement)
        improved = stopper.step(1.50, epoch=1)
        self.assertTrue(improved)
        self.assertEqual(stopper.best, 1.50)
        self.assertEqual(stopper.best_epoch, 1)

        # Loss decreases further
        improved = stopper.step(1.20, epoch=2)
        self.assertTrue(improved)
        self.assertEqual(stopper.best, 1.20)
        self.assertEqual(stopper.best_epoch, 2)

        # Loss increases (diverging / overfitting step 1)
        improved = stopper.step(1.25, epoch=3)
        self.assertFalse(improved)
        self.assertEqual(stopper.counter, 1)

        # Overfitting step 2
        stopper.step(1.30, epoch=4)
        self.assertEqual(stopper.counter, 2)
        self.assertFalse(stopper.should_stop)

        # Overfitting step 3 -> triggers early stop
        stopper.step(1.40, epoch=5)
        self.assertEqual(stopper.counter, 3)
        self.assertTrue(stopper.should_stop)
        self.assertEqual(stopper.best_epoch, 2)

    def test_engine_synthetic_fit(self):
        """Test engine fit with synthetic model, loss tracking, and history.json export."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
        ).to(device)

        x_train = torch.randn(20, 3, 32, 32)
        y_train = torch.randint(0, 3, (20,))
        x_val = torch.randn(10, 3, 32, 32)
        y_val = torch.randint(0, 3, (10,))

        train_ds = TensorDataset(x_train, y_train)
        val_ds = TensorDataset(x_val, y_val)

        train_loader = DataLoader(train_ds, batch_size=5)
        val_loader = DataLoader(val_ds, batch_size=5)

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            class_to_idx = {"a": 0, "b": 1, "c": 2}
            best_metric = fit(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                epochs=2,
                patience=2,
                output_dir=tmp_dir,
                class_to_idx=class_to_idx,
                use_amp=False,
                early_stopping_metric="val_loss",
            )
            self.assertIsNotNone(best_metric)
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "class_to_idx.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "best.pt")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "history.json")))


if __name__ == "__main__":
    unittest.main()
