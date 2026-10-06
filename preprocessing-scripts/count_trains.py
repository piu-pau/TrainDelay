"""
Counts total trains across all processed JSON files (with "origin" and
"destination" fields, e.g. from processed-data/), and checks how many
ended up >=5 minutes late at their final stop vs. on time.

Usage:
    python count_trains.py processed-data/
"""

import json
import sys
from pathlib import Path

DELAY_THRESHOLD_MINUTES = 5


def count_trains(root_folder: str) -> None:
    root = Path(root_folder)
    json_files = sorted(root.rglob("*.json"))

    if not json_files:
        print(f"No .json files found under {root_folder}")
        return

    total_trains = 0
    total_files = 0
    missing_delay_info = 0
    delayed_count = 0
    on_time_count = 0

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            trains = json.load(f)

        total_files += 1
        total_trains += len(trains)

        for train in trains:
            destination = train.get("destination", {})
            delay = destination.get("differenceInMinutes")

            if delay is None:
                missing_delay_info += 1
                continue

            if delay >= DELAY_THRESHOLD_MINUTES:
                delayed_count += 1
            else:
                on_time_count += 1

    labeled_total = delayed_count + on_time_count

    print(f"Files processed:              {total_files}")
    print(f"Total trains:                 {total_trains}")
    print(f"  missing delay info:         {missing_delay_info}")
    print()
    print(f"Labeled trains (usable):      {labeled_total}")
    if labeled_total > 0:
        print(f"  delayed (>={DELAY_THRESHOLD_MINUTES} min):       {delayed_count} "
              f"({100 * delayed_count / labeled_total:.1f}%)")
        print(f"  on time (<{DELAY_THRESHOLD_MINUTES} min):        {on_time_count} "
              f"({100 * on_time_count / labeled_total:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python count_trains.py <folder>")
        sys.exit(1)

    count_trains(sys.argv[1])
