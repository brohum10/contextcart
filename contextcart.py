"""Exact multiple-choice context-budget planning. No model or tokenizer required."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

MAX_INPUT_BYTES = 2_000_000
MAX_WORK = 2_000_000


class InvalidPlan(ValueError):
    pass


class Infeasible(ValueError):
    pass


def integer(value, name, low=0, high=100_000):
    if type(value) is not int or not low <= value <= high:
        raise InvalidPlan(f"{name} must be an integer in [{low}, {high}]")
    return value


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value):
        raise InvalidPlan("IDs must contain 1–80 ASCII letters, digits, dots, underscores or hyphens")
    return value


def fields(obj, expected, name):
    if not isinstance(obj, dict) or set(obj) != set(expected):
        raise InvalidPlan(f"{name} must have exactly these fields: {', '.join(expected)}")


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise InvalidPlan(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path):
    with open(path, "rb") as handle:
        raw = handle.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise InvalidPlan("input exceeds 2 MB")
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=strict_object)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise InvalidPlan("input must be a UTF-8 JSON object") from exc


def validate(spec):
    fields(spec, ("context_window", "prompt_tokens", "output_tokens", "safety_tokens", "groups"), "plan")
    window = integer(spec["context_window"], "context_window", 1)
    reserved = sum(integer(spec[k], k) for k in ("prompt_tokens", "output_tokens", "safety_tokens"))
    budget = window - reserved
    if budget < 0:
        raise Infeasible("prompt, output, and safety reservations exceed the context window")
    if budget > 20_000:
        raise InvalidPlan("evidence budget exceeds the supported 20,000 units")
    if not isinstance(spec["groups"], list) or len(spec["groups"]) > 40:
        raise InvalidPlan("groups must be a list of at most 40 sources")
    seen, groups = set(), []
    for group in spec["groups"]:
        fields(group, ("id", "required", "variants"), "group")
        key = identifier(group["id"])
        if key in seen:
            raise InvalidPlan("duplicate group ID")
        seen.add(key)
        if type(group["required"]) is not bool:
            raise InvalidPlan("required must be boolean")
        variants = group["variants"]
        if not isinstance(variants, list) or not 1 <= len(variants) <= 8:
            raise InvalidPlan("each group needs 1–8 variants")
        variant_ids = set()
        for variant in variants:
            fields(variant, ("id", "tokens", "utility", "text"), "variant")
            vid = identifier(variant["id"])
            if vid in variant_ids:
                raise InvalidPlan("duplicate variant ID within source")
            variant_ids.add(vid)
            integer(variant["tokens"], "tokens", 1)
            integer(variant["utility"], "utility", 0, 1_000_000)
            if not isinstance(variant["text"], str) or not variant["text"].strip():
                raise InvalidPlan("variant text must be nonempty")
            if len(variant["text"].encode("utf-8")) > 50_000:
                raise InvalidPlan("variant text exceeds 50 KB")
        groups.append({**group, "variants": sorted(variants, key=lambda v: v["id"])})
    groups.sort(key=lambda g: g["id"])
    work = (budget + 1) * sum(len(g["variants"]) + (not g["required"]) for g in groups)
    if work > MAX_WORK:
        raise InvalidPlan("optimizer work bound exceeded; reduce evidence budget or variants")
    return groups, budget, reserved


def optimize(spec):
    groups, budget, reserved = validate(spec)
    # At each exact cost retain highest utility, then lexicographically smallest
    # selection key. A state contains decisions only from already processed groups.
    states = {0: (0, ())}
    for group in groups:
        next_states = {}
        options = ([None] if not group["required"] else []) + group["variants"]
        for cost, (utility, path) in states.items():
            for variant in options:
                total = cost + (variant["tokens"] if variant else 0)
                if total > budget:
                    continue
                candidate = (utility + (variant["utility"] if variant else 0),
                             path + ((group["id"], variant["id"]),) if variant else path)
                old = next_states.get(total)
                if old is None or (-candidate[0], candidate[1]) < (-old[0], old[1]):
                    next_states[total] = candidate
        if not next_states:
            raise Infeasible("required sources cannot fit in the evidence budget")
        states = next_states
    used, (utility, path) = min(states.items(), key=lambda item: (-item[1][0], item[0], item[1][1]))
    selected_keys = dict(path)
    selected, omitted = [], []
    for group in groups:
        if group["id"] not in selected_keys:
            omitted.append(group["id"])
            continue
        variant = next(v for v in group["variants"] if v["id"] == selected_keys[group["id"]])
        selected.append({"source_id": group["id"], "variant_id": variant["id"],
                         "required": group["required"], "tokens": variant["tokens"],
                         "utility": variant["utility"], "text": variant["text"],
                         "text_sha256": hashlib.sha256(variant["text"].encode()).hexdigest()})
    normalized = {**spec, "groups": groups}
    digest = hashlib.sha256(json.dumps(normalized, sort_keys=True, ensure_ascii=False,
                                       separators=(",", ":")).encode()).hexdigest()
    return {"version": 1, "input_sha256": digest, "algorithm": "exact_multiple_choice_knapsack",
            "context_window": spec["context_window"], "reserved_tokens": reserved,
            "evidence_budget": budget, "used_tokens": used, "remaining_tokens": budget-used,
            "total_utility": utility, "selected": selected, "omitted_source_ids": omitted,
            "notice": "Costs and utility are caller-supplied. Re-tokenize the complete prompt before dispatch."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args(argv)
    try:
        result = optimize(load(args.input))
    except Infeasible as exc:
        print(json.dumps({"error": str(exc), "kind": "infeasible"}), file=sys.stderr)
        return 3
    except (ValueError, OSError) as exc:
        print(json.dumps({"error": str(exc), "kind": "invalid_input"}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
