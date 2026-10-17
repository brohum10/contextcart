"""Synthetic costs/utilities illustrate the optimizer, not a tokenizer benchmark."""

import json
from pathlib import Path
from contextcart import load, optimize


def main():
    result = optimize(load(Path(__file__).with_name("example.json")))
    assert result["total_utility"] == 20
    assert result["used_tokens"] == 12
    assert [x["source_id"] for x in result["selected"]] == ["b", "c", "policy"]
    print("SYNTHETIC DEMO: token costs are invented teaching units, not measured text lengths.")
    print("Required policy + sources B and C: utility 20. Greedy density picks policy + A: utility 14.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
