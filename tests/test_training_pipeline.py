"""Unit tests for Plant Disease Detection GPU training pipeline."""
import os
import sys
import unittest
import yaml
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.dataset import ManifestDataset
from training.engine import fit, run_epoch
from training.utils import EarlyStopping, build_transforms, clear_gpu_memory, get_gpu_memory_info, setup_gpu


class TestTrainingPipeline(unittest.TestCase):
    def setUp(self):
        self.config_path = "config/training_config.yaml"

    def test_config_structure(self):
        """Verify training_config.yaml has required GPU and model keys."""
        self.assertTrue(os.path.exists(self.config_path), f"Missing {self.config_path}")
        with open(self.config_path) as f:
            cfg = yaml.safe_load(f)

        self.assertIn("crop_classifier", cfg)
        self.assertIn("disease_classifier", cfg)
        self.assertIn("batch_size", cfg)
        self.assertIn("img_size", cfg)
        self.assertIn("device", cfg)
        self.assertIn("mixed_precision", cfg)
        self.assertIn("tf32", cfg)
        self.assertIn("pin_memory", cfg)
        self.assertEqual(cfg["img_size"], 224)

    def test_build_transforms(self):
        """Verify train and eval transforms process images to expected shape."""
        train_tf, eval_tf = build_transforms(224)
        dummy_img = Image.new("RGB", (300, 300), color=(100, 150, 200))

        train_out = train_tf(dummy_img)
        eval_out = eval_tf(dummy_img)

        self.assertEqual(train_out.shape, (3, 224, 224))
        self.assertEqual(eval_out.shape, (3, 224, 224))
        self.assertEqual(train_out.dtype, torch.float32)

    def test_early_stopping(self):
        """Test EarlyStopping counter and trigger."""
        stopper = EarlyStopping(patience=3, mode="max")
        self.assertFalse(stopper.should_stop)

        # Improvement
        improved = stopper.step(0.80)
        self.assertTrue(improved)
        self.assertEqual(stopper.best, 0.80)
        self.assertEqual(stopper.counter, 0)

        # No improvement 1
        improved = stopper.step(0.79)
        self.assertFalse(improved)
        self.assertEqual(stopper.counter, 1)
        self.assertFalse(stopper.should_stop)

        # No improvement 2
        stopper.step(0.78)
        self.assertEqual(stopper.counter, 2)
        self.assertFalse(stopper.should_stop)

        # No improvement 3 -> trigger
        stopper.step(0.75)
        self.assertEqual(stopper.counter, 3)
        self.assertTrue(stopper.should_stop)

    def test_gpu_utils(self):
        """Test GPU setup, memory query, and cache clearing."""
        device = setup_gpu(device_name="cuda", enable_tf32=True, benchmark=True)
        self.assertIsNotNone(device)
        clear_gpu_memory()

        if torch.cuda.is_available():
            mem = get_gpu_memory_info(device)
            self.assertIsNotNone(mem)
            self.assertIn("allocated_mb", mem)

    def test_engine_synthetic_fit(self):
        """Test engine run_epoch and fit with synthetic model & data."""
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

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            class_to_idx = {"a": 0, "b": 1, "c": 2}
            best_acc = fit(
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
            )
            self.assertIsNotNone(best_acc)
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "class_to_idx.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp_dir, "best.pt")))


if __name__ == "__main__":
    unittest.main()
