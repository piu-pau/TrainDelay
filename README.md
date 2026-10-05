# Machine learning project - predicting train delays

Predicts whether a long-distance train in Finland arrives at least 5 min late at its final station. Data is from Digitraffic (all of 2025, ~71,000 trains). We compare logistic regression and random forest.

## Files

- `common.py` - data loading, features, train/val/test split and metrics shared by both methods
- `method1_logistic_regression.py` - logistic regression
- `method2_random_forest.py` - random forest
- `compare_methods.py` - compares the two methods on the test set
- `plots.py` - code for the figures
- `preprocessing-scripts/` - scripts used to clean the raw data
- `junadata/filtered-data4/` - final data after preprocessing
- `figures/`, `results/` - output of the scripts
- `project-report1.pdf` - report

## How to run

Install packages:

```
pip install numpy pandas scikit-learn matplotlib seaborn
```

trains.csv is not committed, so first create it from filtered-data4:

```
python preprocessing-scripts/to_csv.py junadata/filtered-data4/ junadata/trains.csv
```

Then run:

```
python method1_logistic_regression.py
python method2_random_forest.py
python compare_methods.py
```

compare_methods.py uses the test predictions saved by the two method scripts, so run those first.

## Preprocessing

The raw data is too big for GitHub, so it is only stored locally. The clean scripts were run in order:

1. `clean1_distance_stops.py` - keep long-distance trains that were not cancelled
2. `clean2_collapse_stops.py` - keep only the first and last stop and count the stops
3. `clean3_keep_features.py` - remove fields we don't need
4. `clean4_keep_passenger.py` - remove empty trains and museum trains

Other scripts:

- `to_csv.py` - turns filtered-data4 into trains.csv
- `count_trains.py` - counts the trains and how many are delayed
- `delay_rate_by_traintype.py` - delay rate for each train type
