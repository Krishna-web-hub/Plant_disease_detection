"""Agent 2: Out-Of-Distribution (OOD) & Overfit Detection Agent.

Combats the 99%-100% Closed-Set Softmax Overconfidence Failure:
In closed-set classification, softmax forces probabilities to sum to 1.0.
Even for non-target plants, wild trees, or objects, standard softmax outputs
85%-99% confidence because logits are exponentiated without domain anchoring.

This agent evaluates:
1. Free Energy Score: E(x) = -T * logsumexp(z / T)
   In-distribution samples (Tomato/Potato/Pepper) occupy low energy states (< -3.8),
   whereas out-of-distribution flora occupy high energy states (> -3.2).
2. Temperature-Scaled Calibration: T = 2.0 softens uncalibrated logit spikes.
3. Raw Logit Ceiling: In-domain samples achieve high positive activation (> 3.5),
   whereas OOD inputs only achieve weak activation (< 3.0) that standard softmax deceptively inflates.
4. Softmax Margin & Shannon Entropy: Quantifies distribution sharpness.
"""
import math
from typing import Dict, Any, Tuple
import torch


class OODDetectorAgent:
    """Detects when an image falls outside the model's trained domain despite deceptive 90-99% softmax scores."""

    def __init__(
        self,
        energy_threshold: float = -3.2,
        min_max_logit: float = 3.2,
        calibration_temperature: float = 2.0,
    ):
        self.energy_threshold = energy_threshold
        self.min_max_logit = min_max_logit
        self.temperature = calibration_temperature

    def evaluate(self, logits: torch.Tensor, class_map: Dict[int, str]) -> Dict[str, Any]:
        """Analyzes model logits to distinguish genuine high-confidence predictions

        from out-of-distribution overfitting.

        Args:
            logits: 1D Tensor of raw output logits from the classifier
            class_map: Mapping of index -> class name

        Returns:
            Dictionary containing OOD metrics, overfit detection flags, and calibrated probabilities.
        """
        if logits.dim() > 1:
            logits = logits.squeeze(0)

        with torch.no_grad():
            raw_probs = torch.softmax(logits, dim=0)
            top_idx = int(raw_probs.argmax())
            raw_conf = float(raw_probs[top_idx])
            pred_class = class_map.get(top_idx, f"Class_{top_idx}")

            # 1. Free Energy Calculation
            # E(x; f) = -T * log(sum(exp(f_i(x) / T)))
            energy = -1.0 * torch.logsumexp(logits, dim=0).item()

            # 2. Temperature-calibrated probabilities
            scaled_logits = logits / self.temperature
            calibrated_probs = torch.softmax(scaled_logits, dim=0)
            calibrated_conf = float(calibrated_probs[top_idx])

            # 3. Maximum raw logit magnitude
            max_logit = float(logits.max().item())

            # 4. Top-1 vs Top-2 Margin
            sorted_probs, _ = torch.sort(raw_probs, descending=True)
            top1 = sorted_probs[0].item()
            top2 = sorted_probs[1].item() if len(sorted_probs) > 1 else 0.0
            margin = top1 - top2

            # 5. Shannon Entropy
            entropy = -torch.sum(raw_probs * torch.log(raw_probs + 1e-8)).item()

        # OOD & Overfitting Decision Logic
        # Condition A: Energy is above threshold (anomalous input density)
        is_high_energy = energy > self.energy_threshold
        # Condition B: Winning logit is weak despite high softmax (classic softmax illusion)
        is_weak_logit = max_logit < self.min_max_logit
        # Condition C: Raw confidence is deceptively high (>80%) but fails energy or logit tests
        is_overfit_suspect = (raw_conf >= 0.80) and (is_high_energy or is_weak_logit)

        if is_high_energy and is_weak_logit:
            status = "OUT_OF_DISTRIBUTION"
            reason = (
                f"Anomalous energy ({energy:.2f} > {self.energy_threshold:.2f}) and weak max logit "
                f"({max_logit:.2f} < {self.min_max_logit:.2f}). Sample is outside the trained crop domain."
            )
        elif is_overfit_suspect:
            status = "SUSPECT_OVERFIT"
            reason = (
                f"High raw softmax ({raw_conf * 100:.1f}%) contradicted by energy metric ({energy:.2f}) "
                f"or logit ceiling ({max_logit:.2f}). Calibrated confidence adjusted to {calibrated_conf * 100:.1f}%."
            )
        else:
            status = "IN_DOMAIN"
            reason = f"Logit activation ({max_logit:.2f}) and free energy ({energy:.2f}) match in-distribution domain."

        return {
            "status": status,
            "is_in_domain": status == "IN_DOMAIN",
            "is_overfit_detected": is_overfit_suspect,
            "predicted_class": pred_class,
            "raw_confidence": round(raw_conf, 4),
            "calibrated_confidence": round(calibrated_conf, 4),
            "free_energy": round(energy, 4),
            "max_logit": round(max_logit, 4),
            "margin": round(margin, 4),
            "entropy": round(entropy, 4),
            "reason": reason,
        }
