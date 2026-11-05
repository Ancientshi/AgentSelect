"""Tests for conversion of the optimized JSONL release."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_optimized_dataset.py"
PARTS = ("PartI", "PartII", "PartIII")


class PrepareDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "input"
        self.output = self.root / "output"
        self.source.mkdir()
        for index, part in enumerate(PARTS):
            folder = self.source / part
            folder.mkdir()
            self.write_jsonl(folder / "queries.jsonl", [{"query_id": "same", "query": f"query {index}"}])
            self.write_jsonl(folder / "agents.jsonl", [
                {"agent_id": "a", "llm": "model", "tools": ["tool"]},
                {"agent_id": "b", "llm": None, "tools": []},
            ])
            self.write_jsonl(folder / "rankings.jsonl", [{"query_id": "same", "agent_ids": ["b", "a"]}])
        self.write_jsonl(self.source / "tools.jsonl", [{"name": "tool", "description": "example"}])
        self.write_jsonl(self.source / "llms.jsonl", [{"name": "model", "description": "example"}])

    @staticmethod
    def write_jsonl(path, records):
        path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    def run_converter(self):
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--input", str(self.source), "--output", str(self.output)],
            capture_output=True, text=True, check=False,
        )

    def read_output(self, *path):
        return json.loads((self.output.joinpath(*path)).read_text(encoding="utf-8"))

    def test_unique_queries_and_preserved_data(self):
        result = self.run_converter()
        self.assertEqual(result.returncode, 0, result.stderr)
        query_ids = set()
        for index, part in enumerate(PARTS):
            questions = self.read_output(part, "questions", "merge.json")
            agents = self.read_output(part, "agents", "merge.json")
            rankings = self.read_output(part, "rankings", "merge.json")["rankings"]
            self.assertEqual(len(questions), 1)
            qid, question = next(iter(questions.items()))
            self.assertNotIn(qid, query_ids)
            query_ids.add(qid)
            self.assertEqual(question["input"], f"query {index}")
            self.assertEqual(question["source_query_id"], "same")
            self.assertEqual(question["part"], part)
            self.assertEqual(rankings, {qid: ["b", "a"]})
            self.assertTrue(set(rankings[qid]).issubset(agents))
            self.assertEqual(agents["a"], {"M": {"id": "model", "name": "model"}, "T": {"tools": ["tool"]}})
            self.assertEqual(agents["b"]["M"]["name"], "")
        self.assertEqual(self.read_output("Tools", "merge.json")["tool"]["description"], "example")
        self.assertEqual(self.read_output("LLMs", "merge.json")["model"]["description"], "example")

    def test_invalid_reference_leaves_no_output(self):
        self.write_jsonl(self.source / "PartII" / "rankings.jsonl", [
            {"query_id": "same", "agent_ids": ["missing"]},
        ])
        result = self.run_converter()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing agent", result.stderr)
        self.assertFalse(self.output.exists())
        self.assertEqual(list(self.root.glob(".output-*")), [])

    def test_existing_output_is_not_overwritten(self):
        self.output.mkdir()
        sentinel = self.output / "keep.txt"
        sentinel.write_text("unchanged", encoding="utf-8")
        result = self.run_converter()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged")


if __name__ == "__main__":
    unittest.main()
