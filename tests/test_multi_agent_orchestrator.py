"""Unit and integration tests for Multi-Agent Orchestration Pipeline.

Tests:
1. SecuritySentinelAgent (Magic bytes, file size, decompression bomb protection)
2. OODDetectorAgent (Free energy calculation, overfit detection, logit thresholds)
3. TaxonomyGuardAgent (Biological compatibility matrix)
4. BotanicalCriticAgent (Morphology vs symptom cross-examination)
5. MultiAgentOrchestrator (End-to-end diagnosis on OOD and in-domain specimens)
"""
import io
import os
import unittest
import torch
from PIL import Image

from deployment.agents.security_sentinel import SecuritySentinelAgent
from deployment.agents.ood_detector import OODDetectorAgent
from deployment.agents.taxonomy_guard import TaxonomyGuardAgent
from deployment.agents.botanical_critic import BotanicalCriticAgent
from deployment.agents.safety_rag_agent import SafetyRAGAgent
from deployment.agents.orchestrator import MultiAgentOrchestrator
from deployment.api.model_loader import ModelBundle
from deployment.api.treatment import get_treatment


class TestSecuritySentinel(unittest.TestCase):
    def setUp(self):
        self.sentinel = SecuritySentinelAgent(max_size_bytes=1024 * 1024)

    def test_valid_jpeg(self):
        # Create valid tiny JPEG with non-uniform pattern (not blank)
        img = Image.new("RGB", (64, 64), color="green")
        for x in range(32):
            img.putpixel((x, x), (255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        valid, msg, out_img = self.sentinel.inspect(buf.getvalue())
        self.assertTrue(valid)
        self.assertIsNotNone(out_img)

    def test_invalid_magic_bytes(self):
        # Disguised text payload
        fake_payload = b"GET /admin HTTP/1.1\r\nHost: evil.com\r\n"
        valid, msg, out_img = self.sentinel.inspect(fake_payload)
        self.assertFalse(valid)
        self.assertIn("Invalid file format signature", msg)

    def test_size_limit(self):
        oversized = b"\xff\xd8\xff" + b"0" * (2 * 1024 * 1024)
        valid, msg, out_img = self.sentinel.inspect(oversized)
        self.assertFalse(valid)
        self.assertIn("exceeds", msg)


class TestOODDetector(unittest.TestCase):
    def setUp(self):
        self.detector = OODDetectorAgent(energy_threshold=-3.2, min_max_logit=3.2)
        self.classes = {0: "Pepper", 1: "Potato", 2: "Tomato"}

    def test_in_domain_high_logit(self):
        # True in-distribution tomato logit profile (high positive logit, low energy)
        logits = torch.tensor([-2.5, -3.0, 5.8])
        result = self.detector.evaluate(logits, self.classes)
        self.assertEqual(result["status"], "IN_DOMAIN")
        self.assertTrue(result["is_in_domain"])
        self.assertFalse(result["is_overfit_detected"])
        self.assertLess(result["free_energy"], -3.2)

    def test_ood_overfit_interception(self):
        # Out-of-distribution logit profile (weak positive logit, deceptive 88% softmax)
        logits = torch.tensor([0.5, -2.5, 2.56])
        result = self.detector.evaluate(logits, self.classes)
        self.assertIn(result["status"], ["OUT_OF_DISTRIBUTION", "SUSPECT_OVERFIT"])
        self.assertFalse(result["is_in_domain"])
        self.assertTrue(result["is_overfit_detected"])
        self.assertGreater(result["free_energy"], -3.2)


class TestTaxonomyGuard(unittest.TestCase):
    def setUp(self):
        self.guard = TaxonomyGuardAgent()

    def test_compatible_pairing(self):
        is_compat, msg, meta = self.guard.validate_pairing("Tomato", "early_blight")
        self.assertTrue(is_compat)
        self.assertIn("Biologically compatible", msg)

    def test_incompatible_pairing(self):
        # Alternaria solani cannot infect Alstonia scholaris
        is_compat, msg, meta = self.guard.validate_pairing("Alstonia scholaris", "early_blight")
        self.assertFalse(is_compat)
        self.assertIn("BIOLOGICAL INCOMPATIBILITY", msg)


class TestBotanicalCritic(unittest.TestCase):
    def setUp(self):
        self.critic = BotanicalCriticAgent()

    def test_critic_overrules_alstonia_tomato_confusion(self):
        eval_result = self.critic.evaluate_diagnosis(
            candidate_crop="Tomato",
            candidate_disease="Tomato_Early_blight",
            visual_traits={"phyllotaxy": "whorled", "lesion_type": "gall"},
        )
        self.assertTrue(eval_result["is_overruled"])
        self.assertEqual(eval_result["status"], "OVERRULED")
        self.assertIn("Alstonia scholaris", eval_result["revised_crop"])
        self.assertIn("Pauropsylla depressa", eval_result["revised_disease"])


class TestMultiAgentOrchestratorEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = ModelBundle()
        cls.orchestrator = MultiAgentOrchestrator(cls.bundle, treatment_fn=get_treatment)

    def test_alstonia_test_image_intercepted(self):
        test_path = "tests/DISEASED_PLANT.jpg"
        if not os.path.exists(test_path):
            self.skipTest(f"{test_path} not found")

        with open(test_path, "rb") as f:
            raw_bytes = f.read()

        result = self.orchestrator.process(raw_bytes, use_rag=False)
        self.assertEqual(result["type"], "OOD_OVERFIT_INTERCEPTED")
        self.assertFalse(result["is_in_domain"])
        self.assertIn("Alstonia scholaris", result["crop"])
        self.assertIn("Pauropsylla depressa", result["disease"])
        self.assertTrue(len(result["agent_audit_trail"]) >= 4)


if __name__ == "__main__":
    unittest.main()
