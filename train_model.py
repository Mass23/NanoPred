"""
train_model.py - Train pairwise regression models to predict percent identity
from sequence metrics.

Pipeline:
  1. Reduced model (fast screen): GC/Length/Quality features only.
     - Regressor thresholded as classifier at multiple cutoffs (0.85–0.99).
     - Optimized for recall at 0.85.
     - Saved as reduced_model.pkl.

  2. Full model (weighted regression):
     - Sample weights: (y_true / 100)^2  (y_true normalised to [0,1] first).
     - Separate model suites for each k-mer/hash configuration.
     - Feature sets of size 5, 10, 20.
     - Also 50 random feature sets per size.
     - Reports R², MAE, RMSE globally and on the high-identity subset
       (y_true ≥ 85.0, i.e. 85 % in [0–100] scale).

Data scale note:
  Throughout this script, `y` is in [0, 100] (percent identity).
  The "high identity" threshold is HIGH_THRESHOLD = 85.0 (not 0.85).
  Classifier thresholds are likewise expressed in [0, 100].
  Sample weights are (y / 100)^2 so that each weight is in [0, 1].

Usage:
    python train_model.py --input all_pairs_data.csv --output-dir models/ \\
        [--results results.csv] [--seed 23]
"""

import argparse
import os
import random as rnd

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler


# =========================================================
# CONSTANTS
# =========================================================

# y is in [0, 100].  "High identity" is ≥ 85 %.
HIGH_THRESHOLD = 85.0

# Thresholds for classifier-style stats on the reduced model (same scale).
CLASSIFIER_THRESHOLDS = [85.0, 90.0, 95.0, 97.0, 99.0]

# Feature set sizes to test for the full model.
FULL_MODEL_FEATURE_SIZES = [5, 10, 20]

# Number of random feature sets generated per (kmer_config, size) combination.
N_RANDOM_FEATURE_SETS = 50

# Feature group prefixes used throughout for quota enforcement.
GC_PREFIX      = "gc"
LENGTH_PREFIX  = "length"
QUALITY_PREFIX = "quality"
KMER_PREFIX    = "kmer"


# =========================================================
# DATA LOADING AND PREPARATION
# =========================================================

def load_and_prepare_data(csv_path: str) -> tuple:
    """
    Load all_pairs_data.csv, apply length filtering and drop missing values.

    Filtering:
    - Keep rows where 900 ≤ length_min and length_max ≤ 3000.

    Returns:
        (X, y) where X is a DataFrame of numeric features and
        y is a Series of real_percent_identity values in [0, 100].
    """
    print(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns.")

    before = len(df)
    df = df[(df['length_min'] >= 900) & (df['length_max'] <= 3000)]
    print(f"  After length filter (900–3000 bp): {len(df):,} rows "
          f"(removed {before - len(df):,}).")

    before = len(df)
    df = df.dropna()
    print(f"  After dropping NAs: {len(df):,} rows (removed {before - len(df):,}).")

    if len(df) == 0:
        raise ValueError("No data remaining after filtering. Check your CSV file.")

    target = 'real_percent_identity'
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {csv_path}.")

    y = df[target]

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target]
    X = df[feature_cols]

    print(f"  Features: {len(feature_cols)}, Target: '{target}'")
    print(f"  Target range: [{y.min():.2f}, {y.max():.2f}]  mean={y.mean():.2f}")
    return X, y


# =========================================================
# FEATURE ENGINEERING
# =========================================================

def expand_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Create log and sqrt versions of all numeric features.

    For each original column `col`:
      - log version  (`col__log`):  np.log1p(x)
      - sqrt version (`col__sqrt`): np.sqrt(|x|) * sign(x)

    Returns a new DataFrame with original + log + sqrt columns (3× count).
    """
    parts = [X]
    for col in X.columns:
        vals = X[col].values.astype(float)
        log_vals  = np.log1p(vals)
        sqrt_vals = np.sqrt(np.abs(vals)) * np.sign(vals)
        parts.append(pd.Series(log_vals,  index=X.index, name=f"{col}__log"))
        parts.append(pd.Series(sqrt_vals, index=X.index, name=f"{col}__sqrt"))
    return pd.concat(parts, axis=1)


# =========================================================
# FEATURE HELPERS
# =========================================================

def base_name(col: str) -> str:
    """Strip __log / __sqrt suffix to recover the base feature name."""
    if col.endswith("__log"):
        return col[:-5]
    if col.endswith("__sqrt"):
        return col[:-6]
    return col


def get_feature_prefix(b: str) -> str:
    """Return the group prefix of a base feature name (gc/length/quality/kmer/other)."""
    for prefix in [GC_PREFIX, LENGTH_PREFIX, QUALITY_PREFIX, KMER_PREFIX]:
        if b == prefix or b.startswith(prefix + "_"):
            return prefix
    return "other"


def build_base_to_col(cols) -> dict:
    """Map each base feature name to the list of its available transform columns."""
    mapping = {}
    for c in cols:
        b = base_name(c)
        mapping.setdefault(b, []).append(c)
    return mapping


# =========================================================
# FEATURE SELECTION
# =========================================================

def _compute_avg_rank(X: pd.DataFrame, y: pd.Series, seed: int = 0):
    """
    Compute combined average rank for every column in X using:
      1. |Spearman correlation| with y
      2. |Pearson correlation|  with y
      3. RandomForest feature importance

    Also collapses transforms: for each base feature, keeps only the column
    with the best (lowest) avg_rank.

    Returns:
        (best_col_for_base, best_rank_for_base)
        where keys are base feature names.
    """
    cols   = X.columns.tolist()
    y_vals = y.values

    spearman_scores, pearson_scores = [], []
    for col in cols:
        vals = X[col].values
        if np.std(vals) == 0:
            spearman_scores.append(0.0)
            pearson_scores.append(0.0)
        else:
            r_sp, _ = spearmanr(vals, y_vals)
            r_pe, _ = pearsonr(vals, y_vals)
            spearman_scores.append(abs(r_sp) if not np.isnan(r_sp) else 0.0)
            pearson_scores.append(abs(r_pe) if not np.isnan(r_pe) else 0.0)

    rf = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(X, y_vals)
    rf_scores = list(rf.feature_importances_)

    def to_rank(scores: list) -> np.ndarray:
        arr   = np.array(scores, dtype=float)
        order = np.argsort(-arr)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, len(arr) + 1)
        return ranks

    avg_rank = (to_rank(spearman_scores) + to_rank(pearson_scores) + to_rank(rf_scores)) / 3.0

    best_col_for_base  = {}
    best_rank_for_base = {}
    for c, r in zip(cols, avg_rank):
        b = base_name(c)
        if (b not in best_rank_for_base) or (r < best_rank_for_base[b]):
            best_rank_for_base[b] = float(r)
            best_col_for_base[b]  = c

    return best_col_for_base, best_rank_for_base


def select_topn_combined(
    X: pd.DataFrame,
    y: pd.Series,
    n: int,
    seed: int = 0,
    quotas: dict = None,
    allowed_prefixes: list = None,
) -> list:
    """
    Select the top-n features using a combined ranking approach.

    For each base feature, only the best-performing transform is kept
    (among base, base__log, base__sqrt).

    Args:
        n:                Number of features to select.
        quotas:           Dict mapping group prefix → minimum count guaranteed
                          in the result.  E.g. {"gc": 1, "length": 1, "kmer": 1}.
        allowed_prefixes: If given, restrict candidate base features to those
                          whose prefix is in this list (used for reduced model).

    Returns:
        List of chosen column names (length ≤ n).
    """
    if allowed_prefixes is not None:
        cols_to_use = [
            c for c in X.columns
            if get_feature_prefix(base_name(c)) in allowed_prefixes
        ]
        if not cols_to_use:
            return []
        X = X[cols_to_use]

    if X.shape[1] == 0:
        return []

    best_col_for_base, best_rank_for_base = _compute_avg_rank(X, y, seed=seed)

    # Bases sorted from best to worst rank
    ranked_bases = [
        b for b, _ in sorted(best_rank_for_base.items(), key=lambda kv: kv[1])
    ]

    if quotas is None:
        quotas = {}

    selected_bases: list = []
    seen: set = set()

    # 1) Guarantee quota features for each group
    for prefix, q in quotas.items():
        if q <= 0:
            continue
        group_bases = [
            b for b in ranked_bases
            if get_feature_prefix(b) == prefix and b not in seen
        ]
        for b in group_bases[:q]:
            selected_bases.append(b)
            seen.add(b)

    # 2) Back-fill to reach n from global ranking
    for b in ranked_bases:
        if len(selected_bases) >= n:
            break
        if b not in seen:
            selected_bases.append(b)
            seen.add(b)

    return [best_col_for_base[b] for b in selected_bases[:n]]


# =========================================================
# BASE REGRESSORS
# =========================================================

def get_base_regressors(seed: int = 23) -> list:
    """
    Return the list of (name, model) base regressors.

    Excluded: MLPRegressor, ElasticNet, ensemble meta-learner.
    """
    return [
        # Linear models
        ("LinearRegression", LinearRegression()),
        ("Ridge_a1e-1",      Ridge(alpha=1e-1)),
        ("Ridge_a1",         Ridge(alpha=1.0)),
        ("Ridge_a10",        Ridge(alpha=10.0)),

        # Random Forest
        ("RF_md4",  RandomForestRegressor(n_estimators=100, max_depth=4,
                                          random_state=seed, n_jobs=-1)),
        ("RF_md8",  RandomForestRegressor(n_estimators=200, max_depth=8,
                                          random_state=seed, n_jobs=-1)),
        ("RF_md12", RandomForestRegressor(n_estimators=300, max_depth=12,
                                          random_state=seed, n_jobs=-1)),

        # Gradient Boosting
        ("GBR_d3_lr01",  GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                   learning_rate=0.10, random_state=seed)),
        ("GBR_d5_lr01",  GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                   learning_rate=0.10, random_state=seed)),
        ("GBR_d3_lr005", GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                   learning_rate=0.05, random_state=seed)),
    ]


# =========================================================
# REDUCED MODEL  (GC / Length / Quality only)
# =========================================================

def compute_classifier_stats(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> dict:
    """
    Treat the regression output as a binary classifier at `threshold`.

    Positive class  = y ≥ threshold  (high-identity pairs).
    Negative class  = y <  threshold.

    y_true and y_pred are expected in [0, 100] (percent identity scale).

    Returns dict with: threshold, tp, fp, tn, fn, recall, precision, f1.
    """
    y_bin_true = (y_true >= threshold).astype(int)
    y_bin_pred = (y_pred >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_bin_true, y_bin_pred, labels=[0, 1]).ravel()

    recall    = tp / (tp + fn)           if (tp + fn) > 0           else 0.0
    precision = tp / (tp + fp)           if (tp + fp) > 0           else 0.0
    f1        = (2 * precision * recall
                 / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        'threshold': threshold,
        'tp':        int(tp),
        'fp':        int(fp),
        'tn':        int(tn),
        'fn':        int(fn),
        'recall':    recall,
        'precision': precision,
        'f1':        f1,
    }


def train_reduced_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    base_models: list,
    seed: int = 23,
) -> dict:
    """
    Train and evaluate reduced models using only GC / Length / Quality features.

    Steps:
      1. Restrict candidate features to gc/length/quality prefixes.
      2. Select top-10 features via combined ranking (with minimum 1 per group).
      3. For each base model:
         - Train on the training set (no sample weights for reduced model).
         - Evaluate on the test set.
         - Compute classifier-style stats at thresholds 85, 90, 95, 97, 99.
      4. Select the best model: maximise recall@85, tie-break by precision@85.

    Returns:
        dict with best_model, best_model_name, features, best_stats,
        all_model_stats.  Empty dict if no suitable features found.
    """
    print("\n--- REDUCED MODEL (GC / Length / Quality only) ---")

    REDUCED_PREFIXES = [GC_PREFIX, LENGTH_PREFIX, QUALITY_PREFIX]
    REDUCED_QUOTAS   = {GC_PREFIX: 1, LENGTH_PREFIX: 1, QUALITY_PREFIX: 1}
    N_REDUCED_FEATS  = 10

    print(f"  Selecting top {N_REDUCED_FEATS} GC/Length/Quality features...")
    reduced_features = select_topn_combined(
        X_train, y_train, N_REDUCED_FEATS,
        seed=seed,
        quotas=REDUCED_QUOTAS,
        allowed_prefixes=REDUCED_PREFIXES,
    )

    if not reduced_features:
        print("  WARNING: No GC/Length/Quality features found. Skipping reduced model.")
        return {}

    print(f"  Selected features: {reduced_features}")

    X_tr_r = X_train[reduced_features]
    X_te_r = X_test[reduced_features]
    y_tr   = y_train.values
    y_te   = y_test.values

    model_stats = []

    for name, model in base_models:
        m = clone(model)
        m.fit(X_tr_r, y_tr)
        y_pred = m.predict(X_te_r)

        test_r2   = r2_score(y_te, y_pred)
        test_mae  = mean_absolute_error(y_te, y_pred)
        test_rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))

        thresh_stats = [
            compute_classifier_stats(y_te, y_pred, t)
            for t in CLASSIFIER_THRESHOLDS
        ]
        stats_85 = thresh_stats[0]   # first threshold is 85.0

        print(f"  [{name}] R²={test_r2:.4f} | "
              f"recall@85={stats_85['recall']:.4f}  "
              f"prec@85={stats_85['precision']:.4f}  "
              f"f1@85={stats_85['f1']:.4f}")

        model_stats.append({
            'model_name':  name,
            'model':       m,
            'features':    reduced_features,
            'test_r2':     test_r2,
            'test_mae':    test_mae,
            'test_rmse':   test_rmse,
            'thresh_stats': thresh_stats,
            'recall_85':   stats_85['recall'],
            'precision_85': stats_85['precision'],
            'f1_85':       stats_85['f1'],
        })

    best = max(model_stats, key=lambda x: (x['recall_85'], x['precision_85']))
    print(f"\n  Best reduced model: {best['model_name']} "
          f"(recall@85={best['recall_85']:.4f}, "
          f"prec@85={best['precision_85']:.4f})")

    return {
        'best_model':      best['model'],
        'best_model_name': best['model_name'],
        'features':        reduced_features,
        'best_stats':      best,
        'all_model_stats': model_stats,
    }


# =========================================================
# FULL MODEL  –  K-mer experiment helpers
# =========================================================

def detect_kmer_configs(X: pd.DataFrame) -> list:
    """
    Detect kmer/hash configuration columns from the expanded feature DataFrame.

    A kmer configuration is a unique base feature name that starts with "kmer_",
    e.g. "kmer_3_hashjaccard_64".  All three transforms (base, __log, __sqrt) of
    each configuration are grouped together.

    Returns:
        Sorted list of (config_name, [col1, col2, ...]) tuples.
    """
    kmer_cols: dict = {}
    for col in X.columns:
        b = base_name(col)
        if b.startswith(KMER_PREFIX + "_"):
            kmer_cols.setdefault(b, []).append(col)

    configs = sorted(kmer_cols.items())
    names   = [c for c, _ in configs]
    print(f"  Detected {len(configs)} kmer configuration(s): {names}")
    return configs


def get_non_kmer_columns(X: pd.DataFrame) -> list:
    """Return all column names whose base feature is NOT a kmer feature."""
    return [c for c in X.columns if get_feature_prefix(base_name(c)) != KMER_PREFIX]


def compute_high_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = HIGH_THRESHOLD,
) -> dict:
    """
    Compute R², MAE, RMSE on the high-identity subset (y_true ≥ threshold).

    Returns NaN values when no samples meet the threshold.
    """
    mask = y_true >= threshold
    if mask.sum() == 0:
        return {'r2_high': float('nan'), 'mae_high': float('nan'), 'rmse_high': float('nan')}
    y_t = y_true[mask]
    y_p = y_pred[mask]
    return {
        'r2_high':   r2_score(y_t, y_p),
        'mae_high':  mean_absolute_error(y_t, y_p),
        'rmse_high': float(np.sqrt(mean_squared_error(y_t, y_p))),
    }


def generate_random_feature_sets(
    all_base_features: list,
    base_to_col: dict,
    n_features: int,
    n_sets: int = N_RANDOM_FEATURE_SETS,
    seed: int = 42,
    required_prefixes: list = None,
) -> list:
    """
    Generate random feature sets of size `n_features` that respect group quotas.

    Each set guarantees:
      - At least 1 feature from each prefix in `required_prefixes`.
      - Only one transform (base / __log / __sqrt) per base feature.

    Args:
        all_base_features: All available base feature names.
        base_to_col:       Mapping base feature → list of available transform columns.
        n_features:        Desired number of features per set.
        n_sets:            Number of random sets to generate.
        seed:              RNG seed.
        required_prefixes: Group prefixes that must appear at least once.

    Returns:
        List of feature-column lists (each list has ≤ n_features entries).
    """
    if required_prefixes is None:
        required_prefixes = [GC_PREFIX, LENGTH_PREFIX, QUALITY_PREFIX, KMER_PREFIX]

    by_prefix: dict = {}
    other_bases: list = []
    for b in all_base_features:
        p = get_feature_prefix(b)
        if p in required_prefixes:
            by_prefix.setdefault(p, []).append(b)
        else:
            other_bases.append(b)

    min_slots = len(required_prefixes)
    if n_features < min_slots:
        print(f"  WARNING: n_features={n_features} < min required slots={min_slots}. "
              "Skipping random feature sets for this size.")
        return []

    # Check all required groups have at least one feature
    missing = [p for p in required_prefixes if not by_prefix.get(p)]
    if missing:
        print(f"  WARNING: No features for group(s) {missing}. "
              "Skipping random feature sets.")
        return []

    # Allow up to 30× retries per desired set to handle duplicate-base edge cases.
    _MAX_RETRY_MULTIPLIER = 30
    rng = rnd.Random(seed)
    result_sets: list = []
    attempts = 0
    max_attempts = n_sets * _MAX_RETRY_MULTIPLIER

    while len(result_sets) < n_sets and attempts < max_attempts:
        attempts += 1

        # Mandatory: one from each required group
        chosen_bases: list = []
        for prefix in required_prefixes:
            chosen_bases.append(rng.choice(by_prefix[prefix]))

        # Remove duplicates while preserving insertion order (rare but possible if a
        # base feature satisfies more than one required prefix).
        chosen_bases = list(dict.fromkeys(chosen_bases))

        remaining_slots = n_features - len(chosen_bases)
        pool = [b for b in all_base_features if b not in set(chosen_bases)]

        if remaining_slots > len(pool):
            extra = pool[:]
        elif remaining_slots > 0:
            extra = rng.sample(pool, remaining_slots)
        else:
            extra = []

        chosen_bases.extend(extra)

        # For each base feature, randomly pick one transform column.
        # Skip bases with no mapped columns (should not happen with well-formed data).
        chosen_cols = []
        for b in chosen_bases:
            transforms = base_to_col.get(b, [])
            if not transforms:
                continue
            chosen_cols.append(rng.choice(transforms))
        result_sets.append(chosen_cols)

    return result_sets


# =========================================================
# FULL MODEL TRAINING
# =========================================================

def train_full_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    base_models: list,
    seed: int = 23,
) -> list:
    """
    Train weighted regression models for every kmer configuration.

    Sample weights:
        Each training sample is weighted by ``(y_true / 100) ** 2``.
        This up-weights high-identity pairs (y close to 100) while keeping
        all samples in the fit.  The factor (y/100) normalises to [0,1]
        before squaring, so weights are always in [0, 1].

    For each kmer configuration:
      - Candidate features = all non-kmer columns + this kmer config's columns.
      - Feature selection (sizes 5, 10, 20) with quotas:
            ≥1 gc, ≥1 length, ≥1 quality, ≥1 kmer.
      - Each base model is trained with sample weights and evaluated on test set:
            global R², MAE, RMSE
            high-only R²_high, MAE_high, RMSE_high  (y_true ≥ 85.0)

    Additionally, for each (kmer_config, size), 50 random feature sets are
    generated and evaluated with all base models.

    Note on result dicts:
        ``result['model']`` is ``None`` for random feature-set experiments —
        only the metrics are stored for those runs.  Top-N models have their
        fitted model object available for saving.

    Returns:
        List of result dicts (one per model × feature_set combination).
    """
    print("\n--- FULL MODEL EXPERIMENTS ---")

    # Sample weights (training only); y in [0,100] → weight in [0,1]
    # Quadratically up-weight high-identity pairs so that the model is sensitive
    # to errors at the top of the identity range.  y/100 maps to [0,1]; squaring
    # amplifies the weight for values close to 1 while keeping all samples in fit.
    sample_weights = (y_train.values / 100.0) ** 2

    y_tr = y_train.values
    y_te = y_test.values

    kmer_configs  = detect_kmer_configs(X_train)
    non_kmer_cols = get_non_kmer_columns(X_train)

    if not kmer_configs:
        print("  WARNING: No kmer features found; running a single configuration "
              "with non-kmer features only (no kmer quota enforced).")
        kmer_configs = [("no_kmer", [])]

    full_quotas = {
        GC_PREFIX:      1,
        LENGTH_PREFIX:  1,
        QUALITY_PREFIX: 1,
        KMER_PREFIX:    1,
    }

    all_results: list = []

    for kmer_config_name, kmer_col_list in kmer_configs:
        print(f"\n  === K-mer config: {kmer_config_name} ===")

        available_cols = non_kmer_cols + kmer_col_list
        X_tr_sub = X_train[available_cols]
        X_te_sub = X_test[available_cols]

        # ----- Top-N feature sets -----
        topn_feature_sets: dict = {}
        for n_feat in FULL_MODEL_FEATURE_SIZES:
            q = full_quotas if kmer_col_list else {
                GC_PREFIX: 1, LENGTH_PREFIX: 1, QUALITY_PREFIX: 1
            }
            feats = select_topn_combined(
                X_tr_sub, y_train, n_feat, seed=seed, quotas=q,
            )
            topn_feature_sets[n_feat] = feats
            print(f"    Top {n_feat} features: {feats}")

        for n_feat in FULL_MODEL_FEATURE_SIZES:
            feats = topn_feature_sets.get(n_feat, [])
            if not feats:
                continue

            X_tr_f = X_tr_sub[feats]
            X_te_f = X_te_sub[feats]

            for model_name, model in base_models:
                m = clone(model)
                try:
                    m.fit(X_tr_f, y_tr, sample_weight=sample_weights)
                except TypeError:
                    m.fit(X_tr_f, y_tr)

                y_pred = m.predict(X_te_f)

                high = compute_high_metrics(y_te, y_pred)
                all_results.append({
                    'model_name':       model_name,
                    'kmer_config':      kmer_config_name,
                    'n_features':       n_feat,
                    'feature_set_type': 'topN',
                    'features':         feats,
                    'model':            m,
                    'test_r2':          r2_score(y_te, y_pred),
                    'test_mae':         mean_absolute_error(y_te, y_pred),
                    'test_rmse':        float(np.sqrt(mean_squared_error(y_te, y_pred))),
                    **high,
                })

        # ----- Random feature sets -----
        print(f"\n  Random feature sets for {kmer_config_name}...")

        available_base_features = list({base_name(c) for c in available_cols})
        local_base_to_col = build_base_to_col(available_cols)

        req_pfx = (
            [GC_PREFIX, LENGTH_PREFIX, QUALITY_PREFIX, KMER_PREFIX]
            if kmer_col_list
            else [GC_PREFIX, LENGTH_PREFIX, QUALITY_PREFIX]
        )

        for n_feat in FULL_MODEL_FEATURE_SIZES:
            rand_sets = generate_random_feature_sets(
                available_base_features,
                local_base_to_col,
                n_feat,
                n_sets=N_RANDOM_FEATURE_SETS,
                seed=seed + n_feat,
                required_prefixes=req_pfx,
            )

            for set_idx, feats in enumerate(rand_sets):
                # Guard: only use columns that actually exist in X_tr_sub
                feats = [f for f in feats if f in X_tr_sub.columns]
                if not feats:
                    continue

                X_tr_f = X_tr_sub[feats]
                X_te_f = X_te_sub[feats]

                for model_name, model in base_models:
                    m = clone(model)
                    try:
                        m.fit(X_tr_f, y_tr, sample_weight=sample_weights)
                    except TypeError:
                        m.fit(X_tr_f, y_tr)

                    y_pred = m.predict(X_te_f)

                    high = compute_high_metrics(y_te, y_pred)
                    all_results.append({
                        'model_name':       model_name,
                        'kmer_config':      kmer_config_name,
                        'n_features':       n_feat,
                        'feature_set_type': f'random_{set_idx:03d}',
                        'features':         feats,
                        'model':            None,   # random-set models are not persisted to disk
                        'test_r2':          r2_score(y_te, y_pred),
                        'test_mae':         mean_absolute_error(y_te, y_pred),
                        'test_rmse':        float(np.sqrt(mean_squared_error(y_te, y_pred))),
                        **high,
                    })

    return all_results


# =========================================================
# SAVING
# =========================================================

def save_reduced_model(
    output_dir: str,
    result: dict,
    feature_scaler: StandardScaler = None,
) -> None:
    """Save the best reduced model and its feature list to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    if not result:
        print("  No reduced model to save (empty result).")
        return

    model_path = os.path.join(output_dir, 'reduced_model.pkl')
    feats_path = os.path.join(output_dir, 'reduced_model_features.pkl')

    joblib.dump(result['best_model'], model_path)
    joblib.dump(result['features'],   feats_path)

    if feature_scaler is not None:
        joblib.dump(feature_scaler, os.path.join(output_dir, 'feature_scaler.pkl'))

    print(f"  ✓ Reduced model  → {model_path}")
    print(f"  ✓ Reduced feats  → {feats_path}")


def save_full_models(
    output_dir: str,
    results: list,
    feature_scaler: StandardScaler = None,
) -> None:
    """Save the top-10 full models (by R²_high) and a feature scaler."""
    os.makedirs(output_dir, exist_ok=True)

    if feature_scaler is not None:
        joblib.dump(feature_scaler, os.path.join(output_dir, 'feature_scaler.pkl'))

    valid = [r for r in results if not np.isnan(r.get('r2_high', float('nan')))]
    sorted_results = sorted(valid, key=lambda x: x['r2_high'], reverse=True)

    for rank, res in enumerate(sorted_results[:10], 1):
        if res.get('model') is None:
            continue
        safe_name  = res['model_name'].replace(' ', '_')
        kmer_safe  = res['kmer_config'].replace(' ', '_')
        n          = res['n_features']
        model_path = os.path.join(
            output_dir,
            f"rank{rank:02d}_{safe_name}_{kmer_safe}_top{n}.pkl",
        )
        feats_path = model_path.replace('.pkl', '_features.pkl')
        joblib.dump(res['model'],    model_path)
        joblib.dump(res['features'], feats_path)
        print(f"  ✓ Rank-{rank:02d}: {model_path}")

    print(f"  ✓ Full model artifacts saved to {output_dir}/")


def save_results_csv(
    results_path: str,
    full_results: list,
    reduced_result: dict,
) -> None:
    """Save a flat CSV of all full-model results."""
    rows = []
    for res in full_results:
        rows.append({
            'model_name':       res['model_name'],
            'kmer_config':      res['kmer_config'],
            'n_features':       res['n_features'],
            'feature_set_type': res['feature_set_type'],
            'test_r2':          res['test_r2'],
            'test_mae':         res['test_mae'],
            'test_rmse':        res['test_rmse'],
            'r2_high':          res.get('r2_high',   float('nan')),
            'mae_high':         res.get('mae_high',  float('nan')),
            'rmse_high':        res.get('rmse_high', float('nan')),
        })

    pd.DataFrame(rows).to_csv(results_path, index=False)
    print(f"  ✓ Results CSV → {results_path}")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train pairwise regression models to predict percent identity."
    )
    parser.add_argument(
        '--input', '-i',
        default='all_pairs_data.csv',
        help='Path to all_pairs_data CSV (default: all_pairs_data.csv).',
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='models',
        help='Directory in which to save trained models (default: models/).',
    )
    parser.add_argument(
        '--results',
        default='results.csv',
        help='Path to save results CSV (default: results.csv).',
    )
    parser.add_argument(
        '--test-fraction',
        type=float,
        default=0.1,
        help='Fraction of data held out for testing (default: 0.1).',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=23,
        help='Random seed (default: 23).',
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PAIRWISE REGRESSION MODEL TRAINING")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n1. Loading and preparing data...")
    X, y = load_and_prepare_data(args.input)

    # ------------------------------------------------------------------
    # 1b. Expand features (log + sqrt variants)
    # ------------------------------------------------------------------
    print("\n1b. Expanding features (log and sqrt variants)...")
    X = expand_features(X)
    print(f"  Features after expansion: {X.shape[1]}")

    # ------------------------------------------------------------------
    # 2. Train / test split
    # ------------------------------------------------------------------
    print("\n2. Splitting into train / test sets...")
    rng = np.random.default_rng(args.seed)
    n   = len(X)
    test_size  = int(n * args.test_fraction)
    all_idx    = rng.permutation(n)
    test_idx   = all_idx[:test_size]
    train_idx  = all_idx[test_size:]

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test  = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test  = y.iloc[test_idx].reset_index(drop=True)

    print(f"  Train: {len(X_train):,}   Test: {len(X_test):,}")
    print(f"  High-identity (≥{HIGH_THRESHOLD}) in test: "
          f"{(y_test >= HIGH_THRESHOLD).sum():,}")

    # ------------------------------------------------------------------
    # 2b. Scale features (fit on train, apply to both)
    # ------------------------------------------------------------------
    print("\n2b. Scaling features (StandardScaler fit on training data only)...")
    feature_scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        feature_scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        feature_scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    # ------------------------------------------------------------------
    # 3. Base models
    # ------------------------------------------------------------------
    print("\n3. Defining base regressors...")
    base_models = get_base_regressors(seed=args.seed)
    print(f"  {len(base_models)} base models defined: "
          f"{[n for n, _ in base_models]}")

    # ------------------------------------------------------------------
    # 4. Reduced model
    # ------------------------------------------------------------------
    print("\n4. Training reduced model (GC / Length / Quality features only)...")
    reduced_result = train_reduced_model(
        X_train_scaled, y_train,
        X_test_scaled,  y_test,
        base_models,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # 5. Full model experiments (per kmer config)
    # ------------------------------------------------------------------
    print("\n5. Training full models (per kmer configuration)...")
    full_results = train_full_models(
        X_train_scaled, y_train,
        X_test_scaled,  y_test,
        base_models,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # 6. Save artefacts
    # ------------------------------------------------------------------
    print("\n6. Saving models and results...")
    save_reduced_model(args.output_dir, reduced_result, feature_scaler=feature_scaler)
    save_full_models(args.output_dir, full_results, feature_scaler=feature_scaler)
    save_results_csv(args.results, full_results, reduced_result)

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Reduced model classifier table
    if reduced_result:
        print(f"\nReduced Model — best: {reduced_result['best_model_name']}")
        print(f"  Features ({len(reduced_result['features'])}): "
              f"{reduced_result['features']}")
        print(f"  Global test  R²={reduced_result['best_stats']['test_r2']:.4f}  "
              f"MAE={reduced_result['best_stats']['test_mae']:.4f}  "
              f"RMSE={reduced_result['best_stats']['test_rmse']:.4f}")
        print()
        print(f"  {'Threshold':>10}  {'Recall':>8}  {'Precision':>9}  "
              f"{'F1':>8}  {'TP':>7}  {'FP':>7}  {'TN':>7}  {'FN':>7}")
        print("  " + "-" * 78)
        for s in reduced_result['best_stats']['thresh_stats']:
            print(f"  {s['threshold']:>10.2f}  {s['recall']:>8.4f}  "
                  f"{s['precision']:>9.4f}  {s['f1']:>8.4f}  "
                  f"{s['tp']:>7d}  {s['fp']:>7d}  {s['tn']:>7d}  {s['fn']:>7d}")

    # Top-10 full models table
    valid_full = [
        r for r in full_results
        if not np.isnan(r.get('r2_high', float('nan')))
    ]
    top10 = sorted(valid_full, key=lambda x: x['r2_high'], reverse=True)[:10]

    if top10:
        print(f"\nTop 10 full-model candidates  (ranked by R²_high ≥ {HIGH_THRESHOLD}):")
        hdr = (f"  {'#':>3}  {'Model':<25}  {'KmerConfig':<28}  "
               f"{'N':>4}  {'Type':<14}  "
               f"{'R²':>8}  {'R²_high':>8}  {'MAE_high':>9}  {'RMSE_high':>10}")
        print(hdr)
        print("  " + "-" * 125)
        for rank, res in enumerate(top10, 1):
            ftype = res['feature_set_type'][:14]
            print(
                f"  {rank:>3}  {res['model_name']:<25}  {res['kmer_config']:<28}  "
                f"{res['n_features']:>4}  {ftype:<14}  "
                f"{res['test_r2']:>8.4f}  {res['r2_high']:>8.4f}  "
                f"{res['mae_high']:>9.4f}  {res['rmse_high']:>10.4f}"
            )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
