"""Offline checks for the analysis request composer."""
import unittest

from ui_composer import TASKS, compose_request


class ComposerTest(unittest.TestCase):
    def test_each_advertised_output_can_be_composed(self):
        for task, spec in TASKS.items():
            for output in spec["outputs"]:
                with self.subTest(task=task, output=output):
                    result = compose_request(task, spec["subject_default"], "near water", output)
                    self.assertIn(spec["subject_default"], result)
                    self.assertIn("near water", result)
                    self.assertTrue(result.endswith("."))

    def test_output_intent_and_field_subject_are_explicit(self):
        self.assertIn("classification map", compose_request(
            "Land-cover segmentation", "water, roads", "", "Classification map"))
        self.assertIn("pixel coverage", compose_request(
            "Land-cover segmentation", "water", "", "Coverage"))
        self.assertIn("binary mask", compose_request("Buildings", "buildings", "", "Binary mask"))
        self.assertIn("agricultural field boundaries", compose_request(
            "Agricultural fields", "fields", "", "Field polygons"))

    def test_invalid_or_missing_subject_is_rejected(self):
        with self.assertRaises(ValueError):
            compose_request("Buildings", " ", "", "Count")
        with self.assertRaises(ValueError):
            compose_request("Buildings", "houses", "", "Field polygons")


if __name__ == "__main__":
    unittest.main()
