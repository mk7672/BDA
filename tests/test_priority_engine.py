import unittest

from src.priority.priority_engine import assign_priority


class TestPriorityEngine(unittest.TestCase):

    def test_positive_review_is_low(self):
        self.assertEqual(
            assign_priority("positive", 5, []),
            "Low",
        )

    def test_safety_complaint_is_critical(self):
        self.assertEqual(
            assign_priority("negative", 1, ["overheats"]),
            "Critical",
        )

    def test_one_star_negative_review_is_high(self):
        self.assertEqual(
            assign_priority("negative", 1, []),
            "High",
        )

    def test_serious_product_issue_is_high(self):
        self.assertEqual(
            assign_priority("negative", 3, ["broken"]),
            "High",
        )

    def test_general_negative_review_is_medium(self):
        self.assertEqual(
            assign_priority("negative", 3, ["poor quality"]),
            "Medium",
        )

    def test_invalid_sentiment_raises_error(self):
        with self.assertRaises(ValueError):
            assign_priority("neutral", 3, [])

    def test_invalid_rating_raises_error(self):
        with self.assertRaises(ValueError):
            assign_priority("negative", 6, [])


if __name__ == "__main__":
    unittest.main()
