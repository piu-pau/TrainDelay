# Machine learning project: predicting train delays

## Files

- `common.py` - data loading, features, train/val/test split and metrics shared by both methods
- `method1_logistic_regression.py` - logistic regression
- `method2_random_forest.py` - random forest
- `compare_methods.py` - compares the two methods on the test set (run the two methods first)
- `plots.py` - code for the figures
- `junadata/processed-data/` - final data after preprocessing
- `figures/`, `results/` - output of the scripts
- `project-report1.pdf` - report

## Preprocessing

The raw data is too big for GitHub, so it is only stored locally in `junadata/original-data`. `preprocess.py` reads it and saves the result to `junadata/processed-data`. For each train it:

1. keeps long-distance trains that were not cancelled
2. keeps only the first and last stop and counts the stops
3. removes fields we don't need
4. removes empty trains and museum trains

Other scripts:

- `to_csv.py` - turns processed-data into trains.csv
- `count_trains.py` - counts the trains and how many are delayed
- `delay_rate_by_traintype.py` - delay rate for each train type
