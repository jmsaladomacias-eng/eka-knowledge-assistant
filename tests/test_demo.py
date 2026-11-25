import unittest

from demo import run_demo


class DemoTests(unittest.TestCase):
    def test_demo_scenarios(self):
        results = run_demo()

        answered = results["answered"]
        self.assertFalse(answered.abstained)
        self.assertEqual(len(answered.citations), 1)
        self.assertEqual(answered.citations[0].source_path, "docs/architecture.md")

        abstained = results["abstained"]
        self.assertTrue(abstained.abstained)
        self.assertEqual(abstained.citations, ())
        self.assertEqual(abstained.reason, "no_externally_authorized_evidence")

        self.assertEqual(
            results["blocked_reason"],
            "Document has no formal ELIGIBLE RAG admission.",
        )


if __name__ == "__main__":
    unittest.main()
