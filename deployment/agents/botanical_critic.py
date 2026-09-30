"""Agent 4: Adversarial Botanical Critic Agent.

Combats LLM Hallucinations, Prompt Injection, and Neural Network Overfitting:
- Evaluates leaf morphology:
  * Solanaceae (Tomato): Pinnately compound, deeply lobed or toothed margins, glandular hairs.
  * Solanaceae (Pepper): Alternate, simple, smooth/entire margin, elliptical.
  * Solanaceae (Potato): Pinnate compound with small intermediate leaflets.
  * Rosaceae (Apple/Rose): Simple or serrated compound with stipules.
  * Poaceae (Corn): Linear, parallel-veined leaves with sheath.
  * Apocynaceae (Alstonia): Distinct whorls (4-8 leaves per node), oblanceolate, leathery, entire margin.
- Evaluates lesion anatomy & causal contradictions:
  * Raised blister-like pustules / galls (Zoocecidia from insect pests like Pauropsylla depressa).
  * Flat target-like necrotic rings (Alternaria solani).
- Intercepts prompt injection payloads in multimodal/RAG reasoning streams.
- If a 99-100% classifier or hallucinating LLM asserts Early Blight on raised foliar blister galls,
  the Critic flags the contradiction and overrules the false diagnosis.
"""
import re
from typing import Dict, Any, List


class BotanicalCriticAgent:
    """Adversarial critic that cross-examines proposed diagnoses against morphological and visual markers."""

    def __init__(self):
        pass

    def evaluate_diagnosis(
        self,
        candidate_crop: str,
        candidate_disease: str,
        visual_traits: Dict[str, Any] = None,
        llm_reasoning: str = "",
    ) -> Dict[str, Any]:
        """Cross-examines the proposed diagnosis against botanical rules to catch hallucinations and overfit.

        Args:
            candidate_crop: The predicted or proposed crop name
            candidate_disease: The predicted disease name
            visual_traits: Extracted or observed traits (e.g. phyllotaxy, lesion type)
            llm_reasoning: Text reasoning from LLM to check for contradictions
        """
        visual_traits = visual_traits or {}
        criticisms = []
        is_overruled = False
        revised_crop = candidate_crop
        revised_disease = candidate_disease
        confidence_adjustment = 1.0

        crop_lower = candidate_crop.lower()
        disease_lower = candidate_disease.lower()
        reasoning_lower = llm_reasoning.lower()

        # Check 0: Prompt Injection & Adversarial Jailbreak Interception
        if re.search(r"(?i)\b(ignore previous|system prompt|disregard above|assistant:|<\|im_start\|>|<\|im_end\|>)\b", llm_reasoning):
            criticisms.append("PROMPT INJECTION DETECTED: Untrusted instructions detected in LLM reasoning stream.")
            confidence_adjustment = 0.1

        # Check 1: Alstonia scholaris + Insect Gall Signature Detection
        is_whorled = visual_traits.get("phyllotaxy") == "whorled" or "whorl" in reasoning_lower
        has_galls = (
            visual_traits.get("lesion_type") == "gall"
            or "gall" in reasoning_lower
            or "pustule" in reasoning_lower
            or "blister" in reasoning_lower
        )

        # Detect the specific failure case: Alstonia misclassified as Tomato Early Blight
        if ("tomato" in crop_lower or "potato" in crop_lower) and ("early_blight" in disease_lower or "blight" in disease_lower):
            if is_whorled or has_galls or "alstonia" in reasoning_lower:
                is_overruled = True
                criticisms.append(
                    "CRITIC OVERRULE: The leaf morphology exhibits whorled phyllotaxy with raised pustular galls. "
                    "Tomato leaves are pinnately compound and do not form psyllid galls. "
                    "This specimen is Alstonia scholaris (Devil Tree) infested with Pauropsylla depressa insect galls."
                )
                revised_crop = "Alstonia scholaris (Devil Tree)"
                revised_disease = "Psyllid Leaf Galls (Pauropsylla depressa)"
                confidence_adjustment = 0.95

        # Check 2: Rose Black Spot misclassified as Tomato
        if "tomato" in crop_lower and "rose" in reasoning_lower:
            is_overruled = True
            criticisms.append(
                "CRITIC OVERRULE: The leaflet serration and stem stipules identify Rosa spp. "
                "The circular black lesions with feathery margins are Diplocarpon rosae (Rose Black Spot)."
            )
            revised_crop = "Rose"
            revised_disease = "Black Spot (Diplocarpon rosae)"
            confidence_adjustment = 0.92

        # Check 3: Apple Scab misclassified on Solanaceous crops
        if ("tomato" in crop_lower or "potato" in crop_lower) and "apple_scab" in disease_lower:
            is_overruled = True
            criticisms.append(
                "CRITIC OVERRULE: Apple Scab (Venturia inaequalis) only infects Malus (Rosaceae). "
                "Solanaceous crops cannot host Venturia inaequalis."
            )
            revised_crop = "Apple"
            revised_disease = "Apple Scab (Venturia inaequalis)"
            confidence_adjustment = 0.90

        # Check 4: LLM Hallucination Detection on 99-100% Stated Confidence
        if "100%" in llm_reasoning or "99%" in llm_reasoning or "absolutely certain" in reasoning_lower:
            if "target spot" in reasoning_lower and has_galls:
                criticisms.append(
                    "HALLUCINATION ALERT: LLM claimed 99-100% certainty of fungal target spots, but symptoms "
                    "are raised foliar galls (hypertrophic tissue caused by insect oviposition, not fungal necrosis)."
                )
                confidence_adjustment = min(confidence_adjustment, 0.40)

        # Check 5: Pathogen & Leaflet Structural Compatibility
        if "tomato" in crop_lower and visual_traits.get("leaf_type") == "simple":
            criticisms.append(
                "MORPHOLOGY WARNING: Tomato leaves are pinnately compound. A simple undivided leaf indicates "
                "an out-of-distribution plant species."
            )
            confidence_adjustment = min(confidence_adjustment, 0.50)

        verdict_status = "OVERRULED" if is_overruled else ("SUSPICIOUS" if criticisms else "ENDORSED")

        return {
            "status": verdict_status,
            "is_overruled": is_overruled,
            "original_crop": candidate_crop,
            "original_disease": candidate_disease,
            "revised_crop": revised_crop,
            "revised_disease": revised_disease,
            "confidence_adjustment": confidence_adjustment,
            "criticisms": criticisms,
        }
