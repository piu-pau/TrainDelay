"""
Flattens all train JSON files under a folder (e.g. processed-data/) into
a single CSV file with one row per train.

Besides the raw fields, adds the derived features used by the models:
  weekday     = 0-6 (Mon-Sun), from origin scheduledTime in Finnish local time
  hour        = hour of day (with minutes as fraction), Finnish local time
  travelTime  = destination - origin scheduledTime, in minutes
  delayed     = 1 if arrival delay at final station >= 5 min, else 0

Trains with missing values are skipped.

Usage:
    python to_csv.py processed-data/ trains.csv
"""

import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

DELAY_THRESHOLD_MINUTES = 5
LOCAL_TZ = ZoneInfo("Europe/Helsinki")

COLUMNS = [
    "trainNumber", "departureDate", "trainType", "numStops",
    "originStation", "originScheduledTime", "departureDelay",
    "destinationStation", "destinationScheduledTime", "arrivalDelay",
    "weekday", "hour", "travelTime", "delayed",
]


def parse_time(value: str) -> datetime:
    # "2025-01-03T04:54:00.000Z" -> timezone-aware UTC datetime
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def train_to_row(train: dict) -> dict | None:
    origin = train.get("origin", {})
    destination = train.get("destination", {})

    row = {
        "trainNumber": train.get("trainNumber"),
        "departureDate": train.get("departureDate"),
        "trainType": train.get("trainType"),
        "numStops": train.get("numStops"),
        "originStation": origin.get("stationShortCode"),
        "originScheduledTime": origin.get("scheduledTime"),
        "departureDelay": origin.get("differenceInMinutes"),
        "destinationStation": destination.get("stationShortCode"),
        "destinationScheduledTime": destination.get("scheduledTime"),
        "arrivalDelay": destination.get("differenceInMinutes"),
    }
    if any(value is None for value in row.values()):
        return None

    origin_time = parse_time(row["originScheduledTime"])
    destination_time = parse_time(row["destinationScheduledTime"])
    local_time = origin_time.astimezone(LOCAL_TZ)

    row["weekday"] = local_time.weekday()
    row["hour"] = round(local_time.hour + local_time.minute / 60, 3)
    row["travelTime"] = (destination_time - origin_time).total_seconds() / 60
    row["delayed"] = int(row["arrivalDelay"] >= DELAY_THRESHOLD_MINUTES)
    return row


def convert(root_folder: str, output_path: str) -> None:
    json_files = sorted(Path(root_folder).rglob("*.json"))
    if not json_files:
        print(f"No .json files found under {root_folder}")
        return

    written = 0
    skipped = 0
    with open(output_path, "w", newline = "", encoding = "utf-8") as out:
        writer = csv.DictWriter(out, fieldnames = COLUMNS)
        writer.writeheader()

        for file_path in json_files:
            with open(file_path, "r", encoding = "utf-8") as f:
                trains = json.load(f)

            for train in trains:
                row = train_to_row(train)
                if row is None:
                    skipped += 1
                    continue
                writer.writerow(row)
                written += 1

    print(f"Read {len(json_files)} files")
    print(f"Wrote {written} trains to {output_path} (skipped {skipped} with missing values)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python to_csv.py <input_folder> <output.csv>")
        sys.exit(1)

    convert(sys.argv[1], sys.argv[2])