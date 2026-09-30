"""Comprehensive Model Evaluation & Multi-Class ROC Curve Analysis.

Computes:
1. One-vs-Rest (OvR) ROC curve coordinates (FPR, TPR, thresholds) for each class
2. Class-specific ROC-AUC, Macro-averaged ROC-AUC, and Micro-averaged ROC-AUC
3. Multi-class Brier Score (probability calibration metric)
4. Class-wise accuracy and sample distribution
5. Saves structured results to reports/roc_evaluation_<model>_<split>.json

Uses secure model loading (weights_only=True) and pure NumPy calculations
so it runs without requiring external heavy packages.
"""
import argparse
import json
import os
import sys
import yaml
import numpy as np
from PIL import Image

# Protection against PIL decompression bombs
Image.MAX_IMAGE_PIXELS = 10_000_000



def compute_binary_roc(y_true_binary: np.ndarray, y_score: np.ndarray):
    """Computes FPR, TPR, and AUC for a binary indicator vector and score array."""
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true_binary[desc_score_indices]

    distinct_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_indices, y_true.size - 1]

    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    # Add (0, 0) point
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    if tps[-1] <= 0:
        tpr = np.zeros_like(tps, dtype=np.float64)
    else:
        tpr = tps / tps[-1]

    if fps[-1] <= 0:
        fpr = np.zeros_like(fps, dtype=np.float64)
    else:
        fpr = fps / fps[-1]

    # Trapezoidal rule for ROC-AUC (supports NumPy 2.0+ and 1.x)
    trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc = float(trapezoid(tpr, fpr))
    return fpr, tpr, auc



def evaluate_model(model, loader, device, class_names):
    import torch
    import torch.nn.functional as F

    model.eval()
    all_targets = []
    all_probs = []

    with torch.no_grad():

        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            all_targets.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    num_classes = len(class_names)
    n_samples = len(y_true)

    # 1. Overall Accuracy
    y_pred = np.argmax(y_prob, axis=1)
    acc = float(np.mean(y_pred == y_true))

    # 2. Brier Score: 1/N * sum((prob - one_hot)^2)
    one_hot = np.zeros((n_samples, num_classes))
    one_hot[np.arange(n_samples), y_true] = 1.0
    brier_score = float(np.mean(np.sum((y_prob - one_hot) ** 2, axis=1)))

    # 3. Class-wise One-vs-Rest ROC
    roc_results = {}
    class_aucs = []
    for c_idx, c_name in enumerate(class_names):
        binary_true = (y_true == c_idx).astype(int)
        c_prob = y_prob[:, c_idx]
        class_samples = int(np.sum(binary_true))

        if class_samples > 0 and class_samples < n_samples:
            fpr, tpr, auc = compute_binary_roc(binary_true, c_prob)
            class_aucs.append(auc)

            # Subsample ROC curve to at most 50 points for compact JSON reporting
            step = max(1, len(fpr) // 50)
            sampled_fpr = [round(float(v), 4) for v in fpr[::step]]
            sampled_tpr = [round(float(v), 4) for v in tpr[::step]]
            if sampled_fpr[-1] != round(float(fpr[-1]), 4):
                sampled_fpr.append(round(float(fpr[-1]), 4))
                sampled_tpr.append(round(float(tpr[-1]), 4))

            roc_results[c_name] = {
                "samples": class_samples,
                "roc_auc": round(auc, 4),
                "curve_points": {
                    "fpr": sampled_fpr,
                    "tpr": sampled_tpr,
                }
            }
        else:
            roc_results[c_name] = {
                "samples": class_samples,
                "roc_auc": None,
                "note": "Cannot compute ROC curve with 0 positive or 0 negative samples in split"
            }

    # 4. Macro and Micro ROC-AUC
    macro_auc = float(np.mean(class_aucs)) if class_aucs else 0.0

    micro_true = one_hot.ravel()
    micro_prob = y_prob.ravel()
    _, _, micro_auc = compute_binary_roc(micro_true, micro_prob)

    return {
        "total_samples": n_samples,
        "accuracy": round(acc, 4),
        "macro_roc_auc": round(macro_auc, 4),
        "micro_roc_auc": round(micro_auc, 4),
        "brier_score": round(brier_score, 4),
        "classes": roc_results,
    }


def main():
    import timm
    import torch
    from torch.utils.data import DataLoader

    # Add project root to sys.path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from training.dataset import ManifestDataset
    from training.utils import build_transforms, setup_gpu

    parser = argparse.ArgumentParser(description="Evaluate plant disease model and generate ROC curves")
    parser.add_argument("--type", choices=["crop", "disease"], required=True, help="Model type to evaluate")
    parser.add_argument("--crop", default=None, help="Crop name if evaluating disease classifier (e.g. Tomato)")
    parser.add_argument("--split", choices=["test", "val", "train"], default="test", help="Dataset split")
    parser.add_argument("--config", default="config/training_config.yaml")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weights", default=None, help="Explicit path to best.pt checkpoint")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    device = setup_gpu(device_name="cuda", enable_tf32=False, benchmark=False)

    _, eval_tf = build_transforms(cfg["img_size"])


    if args.type == "crop":
        sub_cfg = cfg["crop_classifier"]
        model_dir = sub_cfg["output_dir"]
        label_col = "crop"
        filter_crop = None
        model_tag = "crop_classifier"
    else:
        if not args.crop:
            parser.error("--crop is required when evaluating disease classifier")
        sub_cfg = cfg["disease_classifier"]
        model_dir = os.path.join(sub_cfg["output_dir"], args.crop)
        label_col = "category"
        filter_crop = args.crop
        model_tag = f"disease_{args.crop}"

    class_idx_path = os.path.join(model_dir, "class_to_idx.json")
    if not os.path.isfile(class_idx_path):
        raise FileNotFoundError(f"Missing {class_idx_path}. Train the model first before evaluating.")

    with open(class_idx_path) as f:
        class_to_idx = json.load(f)
    ordered_classes = sorted(class_to_idx, key=class_to_idx.get)

    weights_path = args.weights or os.path.join(model_dir, "best.pt")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Missing checkpoint at {weights_path}. Train the model first.")

    dataset = ManifestDataset(
        sub_cfg["manifest"],
        split=args.split,
        label_col=label_col,
        crop=filter_crop,
        transform=eval_tf,
        class_to_idx=class_to_idx,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"\nEvaluating {model_tag} on split '{args.split}' ({len(dataset)} samples)...")

    # Secure model instantiation and checkpoint loading (weights_only=True)
    model = timm.create_model(sub_cfg["model_name"], pretrained=False, num_classes=len(ordered_classes))
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    metrics = evaluate_model(model, loader, device, ordered_classes)
    metrics["model_name"] = sub_cfg["model_name"]
    metrics["split"] = args.split
    metrics["checkpoint"] = weights_path

    os.makedirs("reports", exist_ok=True)
    out_path = f"reports/roc_eval_{model_tag}_{args.split}.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 65)
    print(f"EVALUATION SUMMARY ({model_tag} | Split: {args.split})")
    print("=" * 65)
    print(f"Accuracy:        {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro ROC-AUC:   {metrics['macro_roc_auc']:.4f}")
    print(f"Micro ROC-AUC:   {metrics['micro_roc_auc']:.4f}")
    print(f"Brier Score:     {metrics['brier_score']:.4f} (calibration quality; closer to 0 is better)")
    print("-" * 65)
    print(f"{'Class Name':38s} | {'Samples':7s} | {'ROC-AUC':7s}")
    print("-" * 65)
    for c_name, data in metrics["classes"].items():
        auc_str = f"{data['roc_auc']:.4f}" if data.get("roc_auc") is not None else "N/A"
        print(f"{c_name[:38]:38s} | {data['samples']:7d} | {auc_str:7s}")
    print("=" * 65)
    print(f"Full metrics and ROC curve points saved to {out_path}")


if __name__ == "__main__":
    main()
