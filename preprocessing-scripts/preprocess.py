"""
Runs the whole preprocessing for every Digitraffic /trains/<date> JSON file
under original-data/ and saves the results to processed-data/ with the same
folder structure. For each train:

  1. Keeps only long-distance trains (trainCategory == "Long-distance")
     that were not cancelled.
  2. Keeps only passenger stops (trainStopping and commercialStop), counts
     the stations as "numStops" and collapses the timetable to "origin"
     (first stop) and "destination" (last stop). Trains with fewer than
     two stops are skipped.
  3. Keeps only the fields used in the project:
       trainNumber, departureDate, trainType, numStops and
       stationShortCode, scheduledTime, differenceInMinutes for
       origin and destination
  4. Removes trains that are not passenger traffic:
       MV  = Kaukoliikenteen tyhjävaunujuna (long-distance empty stock move)
       V   = Henkiloliikenteen tyhjavaunujuna (passenger-service empty stock move)
       MUS = Museojunat (museum trains)

Usage:
    python preprocess.py [input_folder] [output_folder]

Defaults: junadata/original-data and junadata/processed-data
"""

import json
import sys
from pathlib import Path

JUNADATA = Path(__file__).parent.parent / "junadata"
NON_PASSENGER_TYPES = {"MV", "V", "MUS"}
ORIGIN_DEST_FIELDS_TO_KEEP = ["stationShortCode", "scheduledTime", "differenceInMinutes"]


def is_real_stop(row: dict) -> bool:
    return row.get("trainStopping", True) and row.get("commercialStop", True)


def process_train(train: dict) -> dict | None:
    """Returns the cleaned train, or None if the train is dropped."""
    if train.get("trainCategory") != "Long-distance" or train.get("cancelled", False):
        return None
    if train.get("trainType") in NON_PASSENGER_TYPES:
        return None

    real_stops = [row for row in train.get("timeTableRows", []) if is_real_stop(row)]
    if len(real_stops) < 2:
        return None

    # Each station has an ARRIVAL + DEPARTURE row, so count the stations, not rows
    origin, destination = real_stops[0], real_stops[-1]
    return {
        "trainNumber": train.get("trainNumber"),
        "departureDate": train.get("departureDate"),
        "trainType": train.get("trainType"),
        "numStops": len({row["stationShortCode"] for row in real_stops}),
        "origin": {k: origin.get(k) for k in ORIGIN_DEST_FIELDS_TO_KEEP},
        "destination": {k: destination.get(k) for k in ORIGIN_DEST_FIELDS_TO_KEEP},
    }


def preprocess(input_folder: Path, output_folder: Path) -> None:
    json_files = sorted(input_folder.rglob("*.json"))
    if not json_files:
        print(f"No .json files found under {input_folder}")
        return

    total_in = 0
    total_out = 0
    for input_path in json_files:
        with open(input_path, "r", encoding = "utf-8") as f:
            trains = json.load(f)

        kept = [t for t in map(process_train, trains) if t is not None]
        total_in += len(trains)
        total_out += len(kept)

        output_path = output_folder / input_path.relative_to(input_folder)
        output_path.parent.mkdir(parents = True, exist_ok = True)
        with open(output_path, "w", encoding = "utf-8") as f:
            json.dump(kept, f, ensure_ascii = False, indent = 2)

    print(f"Files processed:  {len(json_files)}")
    print(f"Trains in input:  {total_in}")
    print(f"Trains kept:      {total_out}")
    print(f"Written to:       {output_folder}")


if __name__ == "__main__":
    if len(sys.argv) not in (1, 3):
        print("Usage: python preprocess.py [input_folder] [output_folder]")
        sys.exit(1)

    if len(sys.argv) == 3:
        preprocess(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        preprocess(JUNADATA / "original-data", JUNADATA / "processed-data")
