"""Settings shared by all analyses. Changing anything here changes the reported results."""

SEED = 42

N_SPLITS = 10
N_REPEATS = 10

TARGET = "TYPE"
ID_COLUMN = "SUBJECT_ID"

# CA72-4: 68.8% missing. NEU: missingness pattern differs across the cohort and
# would act as a proxy for the measurement period rather than for the outcome.
EXCLUDED_COLUMNS = ["CA72-4", "NEU"]

# Values outside these ranges are physiologically impossible and treated as missing.
PLAUSIBLE_RANGES = {"Ca": (1.5, 3.5)}  # serum calcium, mmol/L

# Threshold for the threshold-dependent metrics in Table 2.
DECISION_THRESHOLD = 0.60

N_BOOTSTRAP = 2000

# Comparator settings (fixed; no hyperparameter search was performed).
RANDOM_FOREST = dict(n_estimators=50, max_depth=3)
XGBOOST = dict(n_estimators=25, max_depth=2, learning_rate=0.05,
               subsample=0.5, colsample_bytree=0.9)
DECISION_TREE = dict(max_depth=5)
