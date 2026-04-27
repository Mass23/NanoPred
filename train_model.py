"""
train_model.py - Train pairwise regression models to predict percent identity
from sequence metrics.

Pipeline:
  Phase A — Fast model comparison on a 100 k subsample:
    1. A stratified (or oversampled) subsample of 100 k rows is drawn from the
       filtered dataset, with at least 10 % positives (y_bin == 1).
    2. The subsample is split 50 % / 25 % / 25 % (fit / val / test).
    3. Reduced-model candidates are trained and ranked on this subsample;
       the winning (classifier family × feature-set size) configuration is
       identified.
    4. Full-model experiments are also run on the subsample for reference.

  Phase B — Retrain final models on the full dataset:
    5. Reduced model: the Phase-A winning configuration is retrained on the
       full dataset (80 % fit / 20 % val for threshold tuning; test on the
       held-out test fraction).  Threshold is re-tuned to meet recall ≥ 0.99.
    6. Full models: all kmer-configuration experiments are repeated on the
       full dataset split.
    7. Final artefacts are saved: reduced_model.pkl, reduced_model_features.pkl,
       reduced_model_metadata.pkl, feature_scaler.pkl, and top-10 full models.

  Reduced model (fast binary screen):
    - Binary target: y_bin = (real_percent_identity >= 85).
    - True classifiers (LogisticRegression, RidgeClassifier,
      RandomForestClassifier, GradientBoostingClassifier, MLPClassifier,
      SGDClassifier).
    - Threshold selection: maximise precision subject to recall >= TARGET_RECALL
      (0.99).  If no candidate achieves the target recall, the candidate with
      the highest achievable recall is used with a loud warning.

  Full model (weighted regression):
    - Sample weights: (y_true / 100)^2  (y_true normalised to [0,1] first).
    - Separate model suites for each k-mer/hash configuration.
    - Feature sets of size 5, 10, 20.
    - Also 50 random feature sets per size.
    - Reports R², MAE, RMSE globally and on the high-identity subset
      (y_true ≥ 85.0, i.e. 85 % in [0–100] scale).

Data scale note:
  Throughout this script, `y` is in [0, 100] (percent identity).
  The "high identity" threshold is HIGH_THRESHOLD = 85.0 (not 0.85).
  Sample weights are (y / 100)^2 so that each weight is in [0, 1].

Usage:
    python train_model.py --input all_pairs_data.csv --output-dir models/ \\
        [--results results.csv] [--seed 23] \\
        [--subsample-size 100000] [--min-positive-frac 0.10] [--no-retrain]
"""

import argparse
import os
import random as rnd

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier, SGDClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor

# =========================================================
# CONSTANTS
# =========================================================

# y is in [0, 100].  "High identity" is ≥ 85 %.
HIGH_THRESHOLD = 85.0

# Minimum precision at cutoff 85 required for a reduced-model candidate to be
# considered during validation-based model selection.  Candidates that fail
# this floor receive a selection score of -inf and are never chosen as best.
PRECISION_MIN = 0.10

# Target recall that each reduced-model candidate must achieve on the
# validation split.  Candidates that cannot reach this recall at any threshold
# are treated as invalid for selection.
TARGET_RECALL = 0.99

# Feature set sizes to try for the reduced model.
REDUCED_FEATURE_SIZES = [5, 10]

# Feature set sizes to test for the full model.
FULL_MODEL_FEATURE_SIZES = [5, 10, 20]

# Number of random feature sets generated per (kmer_config, size) combination.
N_RANDOM_FEATURE_SETS = 50

# Feature group prefixes used throughout for quota enforcement.
GC_PREFIX      = "gc"
LENGTH_PREFIX  = "length"
QUALITY_PREFIX = "quality"
KMER_PREFIX    = "kmer"

# Default subsample size and minimum positive fraction for Phase A comparison.
SUBSAMPLE_SIZE    = 100_000
MIN_POSITIVE_FRAC = 0.10

# val_fraction used when calling train_reduced_model on the 75 % train+val
# portion of the 100 k subsample so that the overall split is 50/25/25.
# 25 k val / 75 k train+val = 1/3.
SUBSAMPLE_VAL_FRACTION = 1 / 3


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


def draw_subsample(
    X: pd.DataFrame,
    y: pd.Series,
    subsample_size: int = SUBSAMPLE_SIZE,
    min_pos_frac: float = MIN_POSITIVE_FRAC,
    seed: int = 23,
) -> tuple:
    """
    Draw a subsample of ``subsample_size`` rows from (X, y) with at least
    ``min_pos_frac`` fraction of positives (y >= HIGH_THRESHOLD).

    Sampling strategy:
    - If the natural positive fraction >= min_pos_frac: stratified sampling
      that preserves the natural class ratio (without replacement).
    - If the natural positive fraction < min_pos_frac: oversample positives
      (without replacement if enough samples are available; with replacement
      only if necessary) and fill the remainder with negatives.
    - If total rows < subsample_size the full dataset is returned unchanged.

    The sampling is fully deterministic via ``seed``.

    Returns:
        (X_sub, y_sub) as (DataFrame, Series) with reset index.
    """
    rng = np.random.default_rng(seed)
    n_total = len(X)

    y_bin = (y.values >= HIGH_THRESHOLD).astype(int)
    pos_idx = np.where(y_bin == 1)[0]
    neg_idx = np.where(y_bin == 0)[0]
    n_pos_natural = len(pos_idx)
    n_neg_natural = len(neg_idx)
    natural_pos_frac = n_pos_natural / max(1, n_total)

    n_sub = min(subsample_size, n_total)
    n_pos_min = int(np.ceil(n_sub * min_pos_frac))

    if natural_pos_frac >= min_pos_frac:
        # Stratified: preserve natural ratio but do not exceed available counts.
        n_pos_target = min(
            n_pos_natural,
            max(n_pos_min, round(n_sub * natural_pos_frac)),
        )
        n_neg_target = min(n_neg_natural, n_sub - n_pos_target)
        # Re-adjust positives if negatives were capped.
        n_pos_target = n_sub - n_neg_target
        sampled_pos = rng.choice(pos_idx, size=n_pos_target, replace=False)
        sampled_neg = rng.choice(neg_idx, size=n_neg_target, replace=False)
    else:
        # Not enough positives naturally; oversample to reach min_pos_frac.
        n_pos_target = n_pos_min
        n_neg_target = n_sub - n_pos_target
        replace_with_replacement_pos = n_pos_target > n_pos_natural
        if replace_with_replacement_pos:
            print(f"  NOTE: Oversampling positives with replacement "
                  f"({n_pos_natural:,} available, {n_pos_target:,} needed).")
        sampled_pos = rng.choice(pos_idx, size=n_pos_target, replace=replace_with_replacement_pos)
        replace_with_replacement_neg = n_neg_target > n_neg_natural
        if replace_with_replacement_neg:
            print(f"  NOTE: Oversampling negatives with replacement "
                  f"({n_neg_natural:,} available, {n_neg_target:,} needed).")
        sampled_neg = rng.choice(neg_idx, size=n_neg_target, replace=replace_with_replacement_neg)

    chosen_idx = np.concatenate([sampled_pos, sampled_neg])
    rng.shuffle(chosen_idx)

    X_sub = X.iloc[chosen_idx].reset_index(drop=True)
    y_sub = y.iloc[chosen_idx].reset_index(drop=True)
    return X_sub, y_sub


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
        ("Ridge_a001",      Ridge(alpha=0.01)),
        ("Ridge_a005",        Ridge(alpha=0.05)),
        ("Ridge_a01",      Ridge(alpha=0.1)),
        ("Ridge_a05",        Ridge(alpha=0.5)),
        ("Ridge_a1",         Ridge(alpha=1.0)),
        ("Ridge_a5",        Ridge(alpha=5.0)),
        ("Ridge_a10",        Ridge(alpha=10.0)),
        ("Ridge_a50",        Ridge(alpha=50.0)),

        # Random Forest
        ("RF_md4",  RandomForestRegressor(n_estimators=100, max_depth=4,
                                          random_state=seed, n_jobs=12)),
        ("RF_md6",  RandomForestRegressor(n_estimators=150, max_depth=6,
                                          random_state=seed, n_jobs=12)),
        ("RF_md8",  RandomForestRegressor(n_estimators=200, max_depth=8,
                                          random_state=seed, n_jobs=12)),
        ("RF_md10", RandomForestRegressor(n_estimators=250, max_depth=10,
                                           random_state=seed, n_jobs=12)),
        ("RF_md12", RandomForestRegressor(n_estimators=300, max_depth=12,
                                          random_state=seed, n_jobs=12)),
        ("RF_500",  RandomForestRegressor(n_estimators=500, max_depth=6,
                                          random_state=seed, n_jobs=12)),
        ("RF_1000", RandomForestRegressor(n_estimators=1000, max_depth=4,
                                          random_state=seed, n_jobs=12)),

        # Gradient Boosting
        ("GBR_d3_lr01",  GradientBoostingRegressor(n_estimators=200, max_depth=3,
                                                   learning_rate=0.10, random_state=seed)),
        ("GBR_d5_lr01",  GradientBoostingRegressor(n_estimators=200, max_depth=5,
                                                   learning_rate=0.10, random_state=seed)),
        ("GBR_d3_lr005", GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                   learning_rate=0.05, random_state=seed)),
        ("GBR_d5_lr005", GradientBoostingRegressor(n_estimators=300, max_depth=5,
                                                   learning_rate=0.05, random_state=seed)),
        ("GBR_d3_lr001", GradientBoostingRegressor(n_estimators=200, max_depth=7,
                                                   learning_rate=0.01, random_state=seed)),
        ("GBR_d5_lr001", GradientBoostingRegressor(n_estimators=300, max_depth=7,
                                                   learning_rate=0.01, random_state=seed)),
    
        # MLP
        ('MLP_32_16', MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=400)),
        ('MLP_32_', MLPRegressor(hidden_layer_sizes=(32,), max_iter=400)),
        ('MLP_16_32', MLPRegressor(hidden_layer_sizes=(16, 32), max_iter=400)),
        ('MLP_16', MLPRegressor(hidden_layer_sizes=(16,), max_iter=400)),
        ('MLP_32_32', MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=400)),
        ('MLP_64', MLPRegressor(hidden_layer_sizes=(64,), max_iter=400)),

        ]


def get_base_classifiers(seed: int = 23) -> list:
    """
    Return the list of (name, classifier) base classifiers for the reduced model.

    All classifiers target the binary label y_bin = (percent_identity >= 85).
    Variants of each family are provided to give the threshold-search optimiser
    a range of precision/recall trade-off surfaces.
    """
    return [
        # Logistic Regression (L2-regularised, varying C = inverse regularisation)
        ("LogReg_C001",  LogisticRegression(C=0.01,  max_iter=1000, random_state=seed)),
        ("LogReg_C01",   LogisticRegression(C=0.1,   max_iter=1000, random_state=seed)),
        ("LogReg_C1",    LogisticRegression(C=1.0,   max_iter=1000, random_state=seed)),
        ("LogReg_C10",   LogisticRegression(C=10.0,  max_iter=1000, random_state=seed)),
        ("LogReg_C100",  LogisticRegression(C=100.0, max_iter=1000, random_state=seed)),

        # Ridge Classifier (uses decision_function, not predict_proba)
        ("RidgeClf_a001", RidgeClassifier(alpha=0.01)),
        ("RidgeClf_a01",  RidgeClassifier(alpha=0.1)),
        ("RidgeClf_a1",   RidgeClassifier(alpha=1.0)),
        ("RidgeClf_a10",  RidgeClassifier(alpha=10.0)),
        ("RidgeClf_a100", RidgeClassifier(alpha=100.0)),

        # Random Forest Classifier
        ("RFC_md4",  RandomForestClassifier(n_estimators=100, max_depth=4,
                                            random_state=seed, n_jobs=12)),
        ("RFC_md6",  RandomForestClassifier(n_estimators=150, max_depth=6,
                                            random_state=seed, n_jobs=12)),
        ("RFC_md8",  RandomForestClassifier(n_estimators=200, max_depth=8,
                                            random_state=seed, n_jobs=12)),
        ("RFC_md10", RandomForestClassifier(n_estimators=250, max_depth=10,
                                            random_state=seed, n_jobs=12)),

        # Gradient Boosting Classifier
        ("GBC_d3_lr01",  GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                                    learning_rate=0.10, random_state=seed)),
        ("GBC_d5_lr01",  GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                    learning_rate=0.10, random_state=seed)),
        ("GBC_d3_lr005", GradientBoostingClassifier(n_estimators=300, max_depth=3,
                                                    learning_rate=0.05, random_state=seed)),
        ("GBC_d5_lr005", GradientBoostingClassifier(n_estimators=300, max_depth=5,
                                                    learning_rate=0.05, random_state=seed)),

        # MLP Classifier
        ("MLPClf_32_16", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=400,
                                       random_state=seed)),
        ("MLPClf_32",    MLPClassifier(hidden_layer_sizes=(32,),    max_iter=400,
                                       random_state=seed)),
        ("MLPClf_16_32", MLPClassifier(hidden_layer_sizes=(16, 32), max_iter=400,
                                       random_state=seed)),
        ("MLPClf_64",    MLPClassifier(hidden_layer_sizes=(64,),    max_iter=400,
                                       random_state=seed)),

        # SGD Classifier (log_loss → predict_proba; hinge → decision_function)
        ("SGDClf_log",  SGDClassifier(loss='log_loss',       max_iter=1000,
                                      random_state=seed)),
        ("SGDClf_hub",  SGDClassifier(loss='modified_huber', max_iter=1000,
                                      random_state=seed)),
    ]


# =========================================================
# REDUCED MODEL  (GC / Length / Quality only — binary classifier)
# =========================================================

def _get_classifier_scores(clf, X: pd.DataFrame) -> np.ndarray:
    """
    Return continuous scores for the positive class from a fitted classifier.

    Priority:
      1. predict_proba  → probability of the positive class (:, 1).
      2. decision_function → raw decision scores (monotone with class boundary).
      3. Fallback: integer 0/1 predictions cast to float (no threshold search).
    """
    if hasattr(clf, 'predict_proba'):
        return clf.predict_proba(X)[:, 1]
    if hasattr(clf, 'decision_function'):
        return clf.decision_function(X)
    return clf.predict(X).astype(float)


def _search_threshold_for_target_recall(
    scores: np.ndarray,
    y_bin: np.ndarray,
    target_recall: float = TARGET_RECALL,
) -> tuple:
    """
    Find the classification threshold that achieves recall >= target_recall
    with the highest possible precision.

    Candidate thresholds are the sorted unique score values.  When scores look
    like probabilities (all in [0, 1]), a uniform grid with step 0.01 is also
    added to ensure fine-grained coverage.

    Returns:
        (best_threshold, best_recall, best_precision, target_met)

    target_met is True when at least one threshold achieves recall >= target_recall.
    If no threshold meets the recall target, returns the threshold with the
    highest recall and target_met=False.
    """
    candidate_thrs = np.unique(scores)
    if scores.min() >= 0.0 and scores.max() <= 1.0:
        grid = np.arange(0.01, 1.0, 0.01)
        candidate_thrs = np.unique(np.concatenate([candidate_thrs, grid]))

    best_rec_target  = -1.0
    best_prec_target = 0.0
    best_thr_target  = float(np.median(scores))

    best_rec_any     = -1.0
    best_prec_any    = 0.0
    best_thr_any     = float(np.median(scores))

    for thr in candidate_thrs:
        y_pred_bin = (scores >= thr).astype(int)
        tp = int(((y_pred_bin == 1) & (y_bin == 1)).sum())
        fp = int(((y_pred_bin == 1) & (y_bin == 0)).sum())
        fn = int(((y_pred_bin == 0) & (y_bin == 1)).sum())

        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Track overall best (fallback when target cannot be met)
        if recall > best_rec_any or (recall == best_rec_any and precision > best_prec_any):
            best_rec_any  = recall
            best_prec_any = precision
            best_thr_any  = float(thr)

        # Track best among thresholds that meet the recall target:
        # maximise precision, then recall, then threshold (to minimise FPs).
        if recall >= target_recall:
            if (precision > best_prec_target
                    or (precision == best_prec_target and recall > best_rec_target)
                    or (precision == best_prec_target and recall == best_rec_target
                        and float(thr) > best_thr_target)):
                best_rec_target  = recall
                best_prec_target = precision
                best_thr_target  = float(thr)

    if best_rec_target >= 0.0:
        return best_thr_target, best_rec_target, best_prec_target, True
    return best_thr_any, best_rec_any, best_prec_any, False


def train_reduced_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    base_classifiers: list,
    seed: int = 23,
    val_fraction: float = 0.20,
    clf_filter: str = None,
    n_feat_filter: int = None,
) -> dict:
    """
    Train and select reduced binary classifiers using only GC/Length/Quality
    features.

    Steps:
      1. Split training into fit / validation using ``val_fraction`` (default 20 %).
      2. Binary target: y_bin = (y >= HIGH_THRESHOLD).
         Dataset is NOT rebalanced; the natural class distribution is kept.
      3. For each feature-set size N in REDUCED_FEATURE_SIZES ({5, 10})
         (or only ``n_feat_filter`` if provided):
         a. Select top-N GC/Length/Quality features on the fit split using
            combined ranking (only one transform per base feature; ≥1 per group).
         b. For each classifier (or only ``clf_filter`` if provided):
            - Fit on the binary fit split.
            - Obtain continuous scores on the validation split via
              predict_proba (positive class) or decision_function.
            - Choose a threshold achieving recall >= TARGET_RECALL with
              maximum precision (tie-break: higher threshold to reduce FPs).
            - Candidates that cannot reach TARGET_RECALL are marked invalid.
      4. Select the (classifier, N, threshold) triple with the highest
         validation precision among candidates that met TARGET_RECALL.
         Tie-breaks: higher recall, then fewer predicted positives, then name.
         If no candidate meets TARGET_RECALL, emit a loud warning and fall back
         to the candidate with the highest achievable recall.
      5. Evaluate the selected model on the TEST split at the chosen threshold:
         report TP/FP/TN/FN, recall, precision, FP proportion, F1.

    Args:
        val_fraction:  Fraction of X_train used for validation (default 0.20).
                       Set to ~1/3 when X_train is 75 % of a subsample so that
                       the overall split is 50/25/25.
        clf_filter:    When not None, only train classifiers whose name equals
                       this string (used in Phase B to retrain the Phase-A winner).
        n_feat_filter: When not None, only try this specific feature-set size
                       (used in Phase B to retrain the Phase-A winner).

    Returns:
        dict with best_model, best_model_name, features, threshold,
        target_recall, best_stats, all_model_stats.
        Empty dict if no GC/Length/Quality features are found.
    """
    print("\n--- REDUCED MODEL (GC / Length / Quality — binary classifier) ---")

    REDUCED_PREFIXES = [GC_PREFIX, LENGTH_PREFIX, QUALITY_PREFIX]
    REDUCED_QUOTAS   = {GC_PREFIX: 1, LENGTH_PREFIX: 1, QUALITY_PREFIX: 1}

    # ------------------------------------------------------------------
    # Train / validation split (for model selection only; test is reserved).
    # ------------------------------------------------------------------
    val_rng  = np.random.default_rng(seed)
    n_train  = len(X_train)
    val_size = max(1, int(n_train * val_fraction))
    perm     = val_rng.permutation(n_train)
    val_idx  = perm[:val_size]
    fit_idx  = perm[val_size:]

    y_tr_all    = y_train.values
    y_te        = y_test.values
    y_bin_all   = (y_tr_all >= HIGH_THRESHOLD).astype(int)
    y_bin_test  = (y_te     >= HIGH_THRESHOLD).astype(int)

    y_bin_fit = y_bin_all[fit_idx]
    y_bin_val = y_bin_all[val_idx]

    n_pos_fit = int(y_bin_fit.sum())
    n_pos_val = int(y_bin_val.sum())
    n_pos_te  = int(y_bin_test.sum())
    print(f"  Fit  split: {len(y_bin_fit):,}  ({n_pos_fit:,} positive = "
          f"{n_pos_fit / max(1, len(y_bin_fit)):.1%})")
    print(f"  Val  split: {len(y_bin_val):,}  ({n_pos_val:,} positive = "
          f"{n_pos_val / max(1, len(y_bin_val)):.1%})")
    print(f"  Test split: {len(y_bin_test):,}  ({n_pos_te:,} positive = "
          f"{n_pos_te / max(1, len(y_bin_test)):.1%})")
    print(f"  target_recall = {TARGET_RECALL}")

    # Apply optional filters (used in Phase B to retrain only the Phase-A winner).
    classifiers_to_run = (
        [(n, c) for n, c in base_classifiers if n == clf_filter]
        if clf_filter is not None
        else base_classifiers
    )
    feature_sizes_to_run = (
        [n_feat_filter]
        if n_feat_filter is not None
        else REDUCED_FEATURE_SIZES
    )
    if clf_filter is not None:
        print(f"  Classifier filter: {clf_filter!r}")
    if n_feat_filter is not None:
        print(f"  Feature-size filter: {n_feat_filter}")

    model_stats = []

    for n_feat in feature_sizes_to_run:
        print(f"\n  -- Feature set size: {n_feat} --")

        # Feature selection: use continuous y on the fit split for ranking.
        reduced_features = select_topn_combined(
            X_train.iloc[fit_idx], y_train.iloc[fit_idx], n_feat,
            seed=seed,
            quotas=REDUCED_QUOTAS,
            allowed_prefixes=REDUCED_PREFIXES,
        )

        if not reduced_features:
            print(f"  WARNING: No GC/Length/Quality features found for N={n_feat}. "
                  "Skipping.")
            continue

        print(f"  Selected features ({n_feat}): {reduced_features}")

        X_fit  = X_train.iloc[fit_idx][reduced_features]
        X_val  = X_train.iloc[val_idx][reduced_features]
        X_te_r = X_test[reduced_features]

        for clf_name, clf in classifiers_to_run:
            m = clone(clf)
            try:
                m.fit(X_fit, y_bin_fit)
            except Exception as exc:
                print(f"  [{clf_name} N={n_feat}] fit failed: {exc!r}")
                continue

            # ----------------------------------------------------------
            # Threshold tuning on validation split.
            # ----------------------------------------------------------
            val_scores = _get_classifier_scores(m, X_val)
            best_thr, val_rec, val_prec, target_met = \
                _search_threshold_for_target_recall(val_scores, y_bin_val)

            # ----------------------------------------------------------
            # Test evaluation at the chosen threshold (for reporting).
            # ----------------------------------------------------------
            test_scores    = _get_classifier_scores(m, X_te_r)
            y_te_bin_pred  = (test_scores >= best_thr).astype(int)

            tp = int(((y_te_bin_pred == 1) & (y_bin_test == 1)).sum())
            fp = int(((y_te_bin_pred == 1) & (y_bin_test == 0)).sum())
            tn = int(((y_te_bin_pred == 0) & (y_bin_test == 0)).sum())
            fn = int(((y_te_bin_pred == 0) & (y_bin_test == 1)).sum())

            test_recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            test_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            test_fp_prop   = fp / (tp + fp) if (tp + fp) > 0 else 0.0
            test_f1        = (
                2 * test_precision * test_recall
                / (test_precision + test_recall)
                if (test_precision + test_recall) > 0 else 0.0
            )

            val_fp_prop = 1.0 - val_prec if val_prec >= 0.0 else 0.0
            print(f"  [{clf_name} N={n_feat}] "
                  f"val_rec={val_rec:.4f}  val_prec={val_prec:.4f}  "
                  f"val_fp_prop={val_fp_prop:.4f}  "
                  f"thr={best_thr:.4f}  "
                  f"target_met={'✓' if target_met else '✗'} | "
                  f"test_recall={test_recall:.4f}  test_prec={test_precision:.4f}  "
                  f"test_fp_prop={test_fp_prop:.4f}  test_f1={test_f1:.4f}")

            model_stats.append({
                'model_name':      f"{clf_name}_N{n_feat}",
                'clf_name':        clf_name,
                'model':           m,
                'features':        reduced_features,
                'n_features':      n_feat,
                'threshold':       best_thr,
                'target_met':      target_met,
                'val_rec':         val_rec,
                'val_prec':        val_prec,
                'val_fp_prop':     val_fp_prop,
                'val_n_pred_pos':  int((val_scores >= best_thr).sum()),
                'test_tp':         tp,
                'test_fp':         fp,
                'test_tn':         tn,
                'test_fn':         fn,
                'test_recall':     test_recall,
                'test_precision':  test_precision,
                'test_fp_prop':    test_fp_prop,
                'test_f1':         test_f1,
            })

    if not model_stats:
        print("  WARNING: No reduced-model candidates found. Returning empty result.")
        return {}

    # Select best: among candidates that met TARGET_RECALL, choose the one
    # with highest validation precision; tie-break by higher recall, then
    # fewer predicted positives, then model name (alphabetical).
    valid = [s for s in model_stats if s['target_met']]
    if not valid:
        print(f"\n  {'!' * 62}")
        print(f"  WARNING: No reduced-model candidate achieved validation")
        print(f"           recall >= {TARGET_RECALL}!")
        print(f"           Falling back to the candidate with the highest")
        print(f"           achievable recall.  Check your data and features.")
        print(f"  {'!' * 62}")
        best = max(
            model_stats,
            key=lambda x: (
                x['val_rec'],
                x['val_prec'],
                -x['val_n_pred_pos'],
                x['model_name'],
            ),
        )
    else:
        best = max(
            valid,
            key=lambda x: (
                x['val_prec'],
                x['val_rec'],
                -x['val_n_pred_pos'],
                x['model_name'],
            ),
        )

    target_recall_met = bool(valid)
    if not target_recall_met:
        print(f"\n  Best reduced model (fallback — target recall NOT met): "
              f"{best['model_name']} "
              f"(val_rec={best['val_rec']:.4f}, "
              f"val_prec={best['val_prec']:.4f}, "
              f"val_fp_prop={best['val_fp_prop']:.4f}, "
              f"threshold={best['threshold']:.4f})")
    else:
        print(f"\n  Best reduced model: {best['model_name']} "
              f"(val_rec={best['val_rec']:.4f}, "
              f"val_prec={best['val_prec']:.4f}, "
              f"val_fp_prop={best['val_fp_prop']:.4f}, "
              f"threshold={best['threshold']:.4f}, "
              f"target_met={best['target_met']})")

    return {
        'best_model':         best['model'],
        'best_model_name':    best['model_name'],
        'features':           best['features'],
        'threshold':          best['threshold'],
        'target_recall':      TARGET_RECALL,
        'target_recall_met':  target_recall_met,
        'best_stats':         best,
        'all_model_stats':    model_stats,
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
    """Save the best reduced model, its feature list, and metadata to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    if not result:
        print("  No reduced model to save (empty result).")
        return

    model_path = os.path.join(output_dir, 'reduced_model.pkl')
    feats_path = os.path.join(output_dir, 'reduced_model_features.pkl')
    meta_path  = os.path.join(output_dir, 'reduced_model_metadata.pkl')

    joblib.dump(result['best_model'], model_path)
    joblib.dump(result['features'],   feats_path)
    joblib.dump(
        {
            'model_name':    result['best_model_name'],
            'features':      result['features'],
            'threshold':     result['threshold'],
            'target_recall': result['target_recall'],
        },
        meta_path,
    )

    if feature_scaler is not None:
        joblib.dump(feature_scaler, os.path.join(output_dir, 'feature_scaler.pkl'))

    print(f"  ✓ Reduced model    → {model_path}")
    print(f"  ✓ Reduced features → {feats_path}")
    print(f"  ✓ Reduced metadata → {meta_path}  "
          f"(threshold={result['threshold']:.4f}, "
          f"target_recall={result['target_recall']})")


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

def _print_reduced_summary(reduced_result: dict, header: str) -> None:
    """Print a summary table for a reduced model result dict."""
    if not reduced_result:
        print(f"  {header}: no result available.")
        return
    bs = reduced_result['best_stats']
    target_met = reduced_result.get('target_recall_met', bs.get('target_met', False))
    recall_tag = "✓ target recall met" if target_met else "✗ TARGET RECALL NOT MET"
    print(f"\n{header} — best: {reduced_result['best_model_name']}  [{recall_tag}]")
    print(f"  Features ({len(reduced_result['features'])}): {reduced_result['features']}")
    print(f"  Chosen threshold : {reduced_result['threshold']:.4f}")
    print(f"  target_recall    : {reduced_result['target_recall']}")
    print(f"  Val recall       : {bs['val_rec']:.4f}   "
          f"Val precision: {bs['val_prec']:.4f}   "
          f"Val FP proportion: {bs['val_fp_prop']:.4f}")
    print()
    print(f"  Test-set results at threshold {reduced_result['threshold']:.4f}:")
    print(f"  {'Metric':<16}  {'Value':>8}")
    print("  " + "-" * 26)
    print(f"  {'Recall':<16}  {bs['test_recall']:>8.4f}")
    print(f"  {'Precision':<16}  {bs['test_precision']:>8.4f}")
    print(f"  {'FP proportion':<16}  {bs['test_fp_prop']:>8.4f}")
    print(f"  {'F1':<16}  {bs['test_f1']:>8.4f}")
    print(f"  {'TP':<16}  {bs['test_tp']:>8d}")
    print(f"  {'FP':<16}  {bs['test_fp']:>8d}")
    print(f"  {'TN':<16}  {bs['test_tn']:>8d}")
    print(f"  {'FN':<16}  {bs['test_fn']:>8d}")


def _print_full_top10(full_results: list) -> None:
    """Print the top-10 full-model candidates ranked by R²_high."""
    valid_full = [
        r for r in full_results
        if not np.isnan(r.get('r2_high', float('nan')))
    ]
    top10 = sorted(valid_full, key=lambda x: x['r2_high'], reverse=True)[:10]
    if not top10:
        print("  No valid full-model results available.")
        return
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
        help='Fraction of full data held out for testing in Phase B (default: 0.1).',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=23,
        help='Random seed (default: 23).',
    )
    parser.add_argument(
        '--subsample-size',
        type=int,
        default=SUBSAMPLE_SIZE,
        help=f'Number of rows in the Phase-A comparison subsample '
             f'(default: {SUBSAMPLE_SIZE:,}).',
    )
    parser.add_argument(
        '--min-positive-frac',
        type=float,
        default=MIN_POSITIVE_FRAC,
        help=f'Minimum fraction of positives (y >= {HIGH_THRESHOLD}) enforced '
             f'in the subsample (default: {MIN_POSITIVE_FRAC}).',
    )
    parser.add_argument(
        '--no-retrain',
        action='store_true',
        default=False,
        help='Skip Phase B (do not retrain final models on the full dataset).',
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PAIRWISE REGRESSION MODEL TRAINING")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data and expand features
    # ------------------------------------------------------------------
    print("\n1. Loading and preparing data...")
    X, y = load_and_prepare_data(args.input)

    print("\n1b. Expanding features (log and sqrt variants)...")
    X = expand_features(X)
    print(f"  Features after expansion: {X.shape[1]}")

    # ------------------------------------------------------------------
    # 2. Base models (defined once; reused in both phases)
    # ------------------------------------------------------------------
    print("\n2. Defining base classifiers (reduced model) and regressors (full model)...")
    base_classifiers = get_base_classifiers(seed=args.seed)
    base_models      = get_base_regressors(seed=args.seed)
    print(f"  {len(base_classifiers)} classifiers for reduced model: "
          f"{[n for n, _ in base_classifiers]}")
    print(f"  {len(base_models)} regressors for full model: "
          f"{[n for n, _ in base_models]}")

    # ======================================================================
    # PHASE A: Fast model comparison on a fixed-size subsample
    # ======================================================================
    print("\n" + "=" * 70)
    print("PHASE A: Fast model comparison on subsample")
    print("=" * 70)
    print(f"  Subsample size       : {args.subsample_size:,}")
    print(f"  Min positive fraction: {args.min_positive_frac:.1%}")
    print(f"  Subsample split      : 50 % train / 25 % val / 25 % test")

    # A-1. Draw the subsample with the required positive fraction.
    print(f"\nA-1. Drawing subsample...")
    X_sub, y_sub = draw_subsample(
        X, y,
        subsample_size=args.subsample_size,
        min_pos_frac=args.min_positive_frac,
        seed=args.seed,
    )
    y_bin_sub = (y_sub.values >= HIGH_THRESHOLD).astype(int)
    n_pos_sub = int(y_bin_sub.sum())
    n_neg_sub = len(y_bin_sub) - n_pos_sub
    print(f"  Subsample: {len(X_sub):,} rows total")
    print(f"    Positives (y >= {HIGH_THRESHOLD}): {n_pos_sub:,}  "
          f"({n_pos_sub / max(1, len(y_bin_sub)):.1%})")
    print(f"    Negatives               : {n_neg_sub:,}  "
          f"({n_neg_sub / max(1, len(y_bin_sub)):.1%})")

    # A-2. 50 / 25 / 25 split of the subsample.
    #      Pass the 75 % train+val portion as X_train to train_reduced_model
    #      with val_fraction=1/3 so the internal fit/val split gives exactly
    #      50 % / 25 % of the total subsample size.
    n_sub = len(X_sub)
    sub_test_size = max(1, int(n_sub * 0.25))
    sub_rng  = np.random.default_rng(args.seed)
    sub_perm = sub_rng.permutation(n_sub)
    sub_test_idx  = sub_perm[:sub_test_size]
    sub_train_idx = sub_perm[sub_test_size:]

    X_sub_train = X_sub.iloc[sub_train_idx].reset_index(drop=True)
    X_sub_test  = X_sub.iloc[sub_test_idx].reset_index(drop=True)
    y_sub_train = y_sub.iloc[sub_train_idx].reset_index(drop=True)
    y_sub_test  = y_sub.iloc[sub_test_idx].reset_index(drop=True)

    y_bin_sub_train = (y_sub_train.values >= HIGH_THRESHOLD).astype(int)
    y_bin_sub_test  = (y_sub_test.values  >= HIGH_THRESHOLD).astype(int)

    print(f"\n  Subsample splits:")
    print(f"    Train+Val (75 %): {len(X_sub_train):,}  "
          f"({y_bin_sub_train.sum():,} pos = {y_bin_sub_train.mean():.1%})")
    print(f"    Test      (25 %): {len(X_sub_test):,}  "
          f"({y_bin_sub_test.sum():,} pos = {y_bin_sub_test.mean():.1%})")

    # A-3. Scale the subsample (scaler fit on train portion only).
    sub_scaler = StandardScaler()
    X_sub_train_scaled = pd.DataFrame(
        sub_scaler.fit_transform(X_sub_train),
        columns=X_sub_train.columns,
        index=X_sub_train.index,
    )
    X_sub_test_scaled = pd.DataFrame(
        sub_scaler.transform(X_sub_test),
        columns=X_sub_test.columns,
        index=X_sub_test.index,
    )

    # A-4. Reduced model comparison on subsample.
    print("\nA-4. Reduced model comparison on subsample "
          f"(val_fraction={SUBSAMPLE_VAL_FRACTION:.3f} → 50/25/25 split)...")
    reduced_result_phaseA = train_reduced_model(
        X_sub_train_scaled, y_sub_train,
        X_sub_test_scaled,  y_sub_test,
        base_classifiers,
        seed=args.seed,
        val_fraction=SUBSAMPLE_VAL_FRACTION,
    )

    # A-5. Full model comparison on subsample.
    print("\nA-5. Full model comparison on subsample...")
    full_results_phaseA = train_full_models(
        X_sub_train_scaled, y_sub_train,
        X_sub_test_scaled,  y_sub_test,
        base_models,
        seed=args.seed,
    )

    # Determine the winning reduced-model configuration from Phase A.
    if reduced_result_phaseA:
        phaseA_clf_name  = reduced_result_phaseA['best_stats']['clf_name']
        phaseA_n_feat    = reduced_result_phaseA['best_stats']['n_features']
        phaseA_target_ok = reduced_result_phaseA.get('target_recall_met', False)
        print(f"\n  Phase A winner: clf={phaseA_clf_name!r}  n_features={phaseA_n_feat}  "
              f"target_recall_met={phaseA_target_ok}")
    else:
        phaseA_clf_name = None
        phaseA_n_feat   = None
        phaseA_target_ok = False
        print("\n  *** Phase A: No valid reduced-model found. "
              "Phase B will train all configurations. ***")

    # Skip Phase B if requested.
    if args.no_retrain:
        print("\n--no-retrain specified: skipping Phase B.  Saving Phase-A results.")
        save_reduced_model(args.output_dir, reduced_result_phaseA,
                           feature_scaler=sub_scaler)
        save_full_models(args.output_dir, full_results_phaseA,
                         feature_scaler=sub_scaler)
        save_results_csv(args.results, full_results_phaseA, reduced_result_phaseA)

        print("\n" + "=" * 70)
        print("SUMMARY  (Phase A only — no retrain)")
        print("=" * 70)
        _print_reduced_summary(
            reduced_result_phaseA,
            "Phase A — Reduced Model (subsample)",
        )
        _print_full_top10(full_results_phaseA)
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        return

    # ======================================================================
    # PHASE B: Retrain final models on the full dataset
    # ======================================================================
    print("\n" + "=" * 70)
    print("PHASE B: Retrain final models on full dataset")
    print("=" * 70)

    # B-1. Full train / test split.
    print("\nB-1. Splitting full dataset into train / test sets...")
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

    y_bin_train = (y_train.values >= HIGH_THRESHOLD).astype(int)
    y_bin_test  = (y_test.values  >= HIGH_THRESHOLD).astype(int)
    print(f"  Train: {len(X_train):,}  "
          f"({y_bin_train.sum():,} pos = {y_bin_train.mean():.1%})")
    print(f"  Test : {len(X_test):,}   "
          f"({y_bin_test.sum():,} pos = {y_bin_test.mean():.1%})")

    # B-2. Scale features (fit on train, apply to both).
    print("\nB-2. Scaling features (StandardScaler fit on full training data)...")
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

    # B-3. Retrain reduced model on full data using Phase-A winning config.
    print("\nB-3. Retraining reduced model on full dataset "
          "(threshold re-tuned on validation split)...")
    reduced_result = train_reduced_model(
        X_train_scaled, y_train,
        X_test_scaled,  y_test,
        base_classifiers,
        seed=args.seed,
        clf_filter=phaseA_clf_name,
        n_feat_filter=phaseA_n_feat,
    )

    # B-4. Retrain full models on full dataset.
    print("\nB-4. Retraining full models on full dataset...")
    full_results = train_full_models(
        X_train_scaled, y_train,
        X_test_scaled,  y_test,
        base_models,
        seed=args.seed,
    )

    # B-5. Save final artefacts.
    print("\nB-5. Saving final models and results...")
    save_reduced_model(args.output_dir, reduced_result, feature_scaler=feature_scaler)
    save_full_models(args.output_dir, full_results, feature_scaler=feature_scaler)
    save_results_csv(args.results, full_results, reduced_result)

    # ======================================================================
    # SUMMARY
    # ======================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\n--- Phase A: subsample comparison ---")
    _print_reduced_summary(
        reduced_result_phaseA,
        "Phase A — Reduced Model (subsample)",
    )
    _print_full_top10(full_results_phaseA)

    print("\n--- Phase B: full-dataset final models ---")
    _print_reduced_summary(
        reduced_result,
        "Phase B — Reduced Model (full dataset)",
    )
    _print_full_top10(full_results)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

