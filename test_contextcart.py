import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from contextcart import Infeasible, InvalidPlan, load, optimize


def variant(vid="full", tokens=3, utility=5, text="Synthetic evidence"):
    return {"id": vid, "tokens": tokens, "utility": utility, "text": text}


def group(gid="a", required=False, variants=None):
    return {"id": gid, "required": required, "variants": variants if variants is not None else [variant()]}


def plan(groups=None, budget=10):
    return {"context_window": budget+6, "prompt_tokens": 2, "output_tokens": 3, "safety_tokens": 1,
            "groups": groups if groups is not None else [group()]}


def brute_force(spec):
    groups = sorted(spec["groups"], key=lambda g: g["id"])
    choices = [([None] if not g["required"] else [])+g["variants"] for g in groups]
    budget = spec["context_window"]-sum(spec[k] for k in ("prompt_tokens", "output_tokens", "safety_tokens"))
    solutions = []
    for selection in itertools.product(*choices):
        used = sum(v["tokens"] for v in selection if v)
        if used <= budget:
            score = sum(v["utility"] for v in selection if v)
            key = tuple((g["id"], v["id"]) for g, v in zip(groups, selection) if v)
            solutions.append((-score, used, key))
    return min(solutions) if solutions else None


class ContextTests(unittest.TestCase):
    def test_example_outperforms_density_greedy(self):
        report = optimize(load("example.json"))
        self.assertEqual(report["total_utility"], 20)
        self.assertEqual(report["used_tokens"], 12)
        self.assertEqual(report["omitted_source_ids"], ["a"])

    def test_randomized_exhaustive_oracle(self):
        rng = random.Random(16)
        for case in range(250):
            groups = [group(str(i), rng.random()<.3,
                            [variant(str(j), rng.randint(1, 8), rng.randint(0, 12))
                             for j in range(rng.randint(1, 3))]) for i in range(rng.randint(0, 6))]
            spec = plan(groups, rng.randint(0, 20))
            expected = brute_force(spec)
            with self.subTest(case=case):
                if expected is None:
                    with self.assertRaises(Infeasible): optimize(spec)
                else:
                    report = optimize(spec)
                    actual = (-report["total_utility"], report["used_tokens"],
                              tuple((r["source_id"], r["variant_id"]) for r in report["selected"]))
                    self.assertEqual(actual, expected)

    def test_required_source_kept(self):
        report = optimize(plan([group("must", True, [variant(utility=0)]), group("optional")], 3))
        self.assertEqual([r["source_id"] for r in report["selected"]], ["must"])

    def test_required_sources_cannot_fit(self):
        with self.assertRaises(Infeasible): optimize(plan([group(required=True)], 2))

    def test_optional_source_skipped_when_oversized(self):
        self.assertEqual(optimize(plan(budget=2))["selected"], [])

    def test_only_one_variant_per_source(self):
        report = optimize(plan([group(variants=[variant("full", 5, 9), variant("short", 2, 4)])], 10))
        self.assertEqual(len(report["selected"]), 1)
        self.assertEqual(report["total_utility"], 9)

    def test_shorter_variant_unlocks_better_combination(self):
        report = optimize(plan([group("a", True, [variant("full", 8, 10), variant("brief", 2, 4)]),
                                group("b", False, [variant("full", 6, 9)])], 8))
        self.assertEqual(report["total_utility"], 13)

    def test_zero_budget_optional(self):
        self.assertEqual(optimize(plan(budget=0))["used_tokens"], 0)

    def test_empty_groups(self):
        self.assertEqual(optimize(plan([]))["total_utility"], 0)

    def test_reservations_exceed_window(self):
        spec = plan()
        spec["output_tokens"] = 100
        with self.assertRaises(Infeasible): optimize(spec)

    def test_tie_prefers_lower_cost(self):
        report = optimize(plan([group(variants=[variant("a", 4, 5), variant("b", 3, 5)])]))
        self.assertEqual(report["selected"][0]["variant_id"], "b")

    def test_tie_prefers_lexical_selection(self):
        report = optimize(plan([group("b"), group("a")], 3))
        self.assertEqual(report["selected"][0]["source_id"], "a")

    def test_order_invariance_including_fingerprint(self):
        spec = load("example.json")
        first = optimize(spec)
        spec["groups"].reverse()
        for g in spec["groups"]: g["variants"].reverse()
        self.assertEqual(first, optimize(spec))

    def test_does_not_modify_input(self):
        spec = load("example.json")
        before = copy.deepcopy(spec)
        optimize(spec)
        self.assertEqual(spec, before)

    def test_text_and_hash_preserved(self):
        text = "Quotes stay exact.\nUnicode: café 日本語"
        report = optimize(plan([group(variants=[variant(text=text)])]))
        self.assertEqual(report["selected"][0]["text"], text)
        self.assertEqual(report["selected"][0]["text_sha256"], hashlib.sha256(text.encode()).hexdigest())

    def test_changed_text_changes_fingerprint(self):
        spec = plan()
        first = optimize(spec)
        spec["groups"][0]["variants"][0]["text"] += " changed"
        self.assertNotEqual(first["input_sha256"], optimize(spec)["input_sha256"])

    def test_invalid_numeric_types(self):
        for invalid in (True, 1.5, "3", -1, None, float("nan"), float("inf")):
            for field in ("tokens", "utility"):
                spec = plan()
                spec["groups"][0]["variants"][0][field] = invalid
                with self.subTest(field=field, invalid=invalid), self.assertRaises(InvalidPlan): optimize(spec)

    def test_zero_cost_rejected(self):
        with self.assertRaises(InvalidPlan): optimize(plan([group(variants=[variant(tokens=0)])]))

    def test_required_type(self):
        with self.assertRaises(InvalidPlan): optimize(plan([group(required=1)]))

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(InvalidPlan): optimize(plan([group(), group()]))
        with self.assertRaises(InvalidPlan): optimize(plan([group(variants=[variant(), variant()])]))

    def test_strict_fields(self):
        spec = plan()
        spec["ignore_constraints"] = True
        with self.assertRaises(InvalidPlan): optimize(spec)

    def test_bad_identifiers(self):
        for value in ("", "../a", "a b", None, "x"*81):
            with self.assertRaises(InvalidPlan): optimize(plan([group(value)]))

    def test_empty_and_large_text_rejected(self):
        for text in ("", "  ", None, "a"*50001):
            with self.assertRaises(InvalidPlan): optimize(plan([group(variants=[variant(text=text)])]))

    def test_group_and_variant_limits(self):
        with self.assertRaises(InvalidPlan): optimize(plan([group(str(i)) for i in range(41)]))
        with self.assertRaises(InvalidPlan): optimize(plan([group(variants=[])]))
        with self.assertRaises(InvalidPlan): optimize(plan([group(variants=[variant(str(i)) for i in range(9)])]))

    def test_work_limit(self):
        with patch("contextcart.MAX_WORK", 1):
            with self.assertRaises(InvalidPlan): optimize(plan())

    def test_evidence_budget_limit(self):
        with self.assertRaises(InvalidPlan): optimize(plan(budget=20001))

    def test_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"bad.json"
            path.write_text('{"x":1,"x":2}')
            with self.assertRaises(InvalidPlan): load(path)

    def test_input_size_limit(self):
        with patch("contextcart.MAX_INPUT_BYTES", 10):
            with self.assertRaises(InvalidPlan): load("example.json")

    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"input.json"
            for spec, code in ((plan(), 0), (plan([group(required=True)], 0), 3), ({}, 2)):
                path.write_text(json.dumps(spec))
                proc = subprocess.run([sys.executable, "contextcart.py", str(path)], capture_output=True, text=True)
                self.assertEqual(proc.returncode, code)
                self.assertIsInstance(json.loads(proc.stdout if code == 0 else proc.stderr), dict)


if __name__ == "__main__":
    unittest.main()
