import copy
import json
import unittest
from pathlib import Path

from scripts.verify import verify


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads(Path("results/reference/report.json").read_text())

    def test_recorded_decisions_replay(self):
        verify(self.report)

    def test_tampering_rejected(self):
        cases = [
            ("prompt", "wrong"),
            ("seconds", -1),
            ("answer_correct", False),
            ("source_id", "fake"),
            ("answer", "wrong"),
        ]
        for field, value in cases:
            with self.subTest(field=field):
                report = copy.deepcopy(self.report)
                report["rows"][0][field] = value
                with self.assertRaises(ValueError):
                    verify(report)
        report = copy.deepcopy(self.report)
        report["summary"]["semantic"]["correct_hits"] += 1
        with self.assertRaises(ValueError):
            verify(report)
        report = copy.deepcopy(self.report)
        hit = next(r for r in report["rows"] if r["hit"])
        hit["correct_hit"] = not hit["correct_hit"]
        with self.assertRaises(ValueError):
            verify(report)
        report = copy.deepcopy(self.report)
        report["rows"].pop()
        with self.assertRaises(ValueError):
            verify(report)

    def test_repository_conventions(self):
        expected = [
            "Problem",
            "Scope",
            "Quick start",
            "Usage",
            "Architecture",
            "Reproduction",
            "Results",
            "Testing",
            "Limitations",
            "Contributing",
            "License",
        ]
        headings = [
            line[3:]
            for line in Path("README.md").read_text().splitlines()
            if line.startswith("## ")
        ]
        self.assertEqual(headings, expected)
        for directory in (
            Path("semantic_cache"),
            Path("scripts"),
            Path("tests"),
            Path("docs"),
            Path("data"),
        ):
            for path in directory.rglob("*"):
                if path.suffix in (".py", ".md", ".json"):
                    self.assertNotIn(chr(8212), path.read_text(), str(path))
