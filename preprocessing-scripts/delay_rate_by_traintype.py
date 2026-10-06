"""
Computes the delay rate (>=5 min late at final stop) broken down by
trainType, using the collapsed origin/destination data structure.

Usage:
    python delay_rate_by_traintype.py processed-data/
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

DELAY_THRESHOLD_MINUTES = 5


def analyze(root_folder: str) -> None:
    root = Path(root_folder)
    json_files = sorted(root.rglob("*.json"))

    if not json_files:
        print(f"No .json files found under {root_folder}")
        return

    # trainType -> [delayed_count, total_count]
    stats = defaultdict(lambda: [0, 0])

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            trains = json.load(f)

        for train in trains:
            destination = train.get("destination", {})
            delay = destination.get("differenceInMinutes")
            if delay is None:
                continue

            train_type = train.get("trainType", "UNKNOWN")
            stats[train_type][1] += 1
            if delay >= DELAY_THRESHOLD_MINUTES:
                stats[train_type][0] += 1

    sorted_types = sorted(stats.items(), key=lambda x: -x[1][1])

    print(f"{'trainType':<12}{'total':>10}{'delayed':>10}{'delay %':>10}")
    print("-" * 42)
    for train_type, (delayed, total) in sorted_types:
        pct = 100 * delayed / total if total > 0 else 0
        print(f"{train_type:<12}{total:>10}{delayed:>10}{pct:>9.1f}%")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python delay_rate_by_traintype.py <folder>")
        sys.exit(1)

    analyze(sys.argv[1])
