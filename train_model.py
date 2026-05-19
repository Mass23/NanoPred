"""
train_model.py - Simplified two-model training workflow for NanoPred.

Workflow overview:
  1) FAST MODEL candidate search (binary classification, <85 vs >=85):
     - Build a balanced 10,000-row candidate-search subset.
     - Generate 2000 random candidates.
     - Each candidate picks:
         * one classifier from get_base_classifiers()
         * exactly 5 random GC/length/quality features (no kmer features)
     - Select winner by highest precision among candidates with recall > 0.99
       on held-out selection data.
       Fallback if none satisfy recall > 0.99: highest recall, then precision.

  2) FAST MODEL winner refinement + final retraining:
     - Refine the initial winner on a balanced 100,000-row subset by testing
       local numeric hyperparameter multipliers: 0.75x, 0.875x, 1.0x, 1.125x, 1.25x.
     - Use all datapoints to build a balanced (<85 vs >=85) dataset
       (oversampling the minority class if needed).
     - Hold out a balanced 10% test split.
     - Cap the training set to at most 200,000 rows while preserving class balance.
     - Train the winning configuration on the capped balanced train split.

  3) FULL MODEL candidate search (regression, 85 <= y <= 100):
     - Build a 10,000-row candidate-search subset from the high-identity rows.
     - Generate 2000 random candidates.
     - Each candidate picks:
         * one regressor from get_base_regressors()
         * exactly 10 random features
         * feature set must include at least one kmer feature
         * every candidate must include all selected fast-model features
         * kmer features must all use exactly one chosen (k, hash-size) configuration
           (e.g. k=5 with hash=64, or k=5 with hash=128)
     - Winner metric: lowest held-out RMSE on the high-identity subset
       (tie-breaks: higher R², then lower MAE).

  4) FULL MODEL winner refinement + final retraining:
     - Refine the initial winner on a homogeneous 100,000-row 85–100 subset by
       testing local numeric hyperparameter multipliers:
       0.75x, 0.875x, 1.0x, 1.125x, 1.25x.
     - Restrict data to 85 <= y <= 100.
     - Draw a homogeneously-balanced training set (up to 1,000,000 data points)
       and a separate homogeneously-balanced validation set (10,000 data points),
       with samples distributed uniformly across the 85–100 identity range.
     - Retrain the winning full-model configuration.

Outputs (saved under final_models/ by default):
  - final_models/fast_model.pkl
  - final_models/full_model.pkl
  - final_models/fast_model_metadata.json
  - final_models/full_model_metadata.json
"""

import argparse
import json
import os
import random as rnd
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier, SGDClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor

# =========================================================
# CONSTANTS
# =========================================================

HIGH_THRESHOLD = 85.0
TARGET_RECALL = 0.99

FAST_CANDIDATE_SUBSET_SIZE = 10_000
FULL_CANDIDATE_SUBSET_SIZE = 10_000
N_RANDOM_CANDIDATES = 2000

FAST_FEATURE_COUNT = 6
FULL_FEATURE_COUNT = 18

TEST_FRACTION = 0.10

FAST_RETRAIN_MAX_TRAIN = 200_000     # cap on training rows for the fast model final retrain
FULL_RETRAIN_MAX_TRAIN = 1_000_000   # cap on training rows for the full model final retrain
FULL_RETRAIN_VAL_SIZE  = 10_000    # fixed validation set size for the full model final retrain
REFINEMENT_SUBSET_SIZE = 100_000   # subset size used for winner-local hyperparameter refinement
REFINEMENT_MULTIPLIERS = (0.75, 0.875, 1.0, 1.125, 1.25)

GC_PREFIX = "gc"
LENGTH_PREFIX = "length"
QUALITY_PREFIX = "quality"
KMER_PREFIX = "kmer"


# =========================================================
# DATA LOADING AND PREPARATION
# =========================================================

def load_and_prepare_data(csv_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load all_pairs_data.csv, apply length filtering, drop NA, return numeric X and y."""
    print(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns.")

    before = len(df)
    df = df[(df['length_min'] >= 900) & (df['length_max'] <= 3000)]
    print(f"  After length filter (900–3000 bp): {len(df):,} rows (removed {before - len(df):,}).")

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


def expand_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create log and sqrt variants for each feature (original + __log + __sqrt)."""
    parts = [X]
    for col in X.columns:
        vals = X[col].values.astype(float)
        log_vals = np.log1p(vals)
        sqrt_vals = np.sqrt(np.abs(vals)) * np.sign(vals)
        parts.append(pd.Series(log_vals, index=X.index, name=f"{col}__log"))
        parts.append(pd.Series(sqrt_vals, index=X.index, name=f"{col}__sqrt"))
    return pd.concat(parts, axis=1)


# =========================================================
# FEATURE HELPERS
# =========================================================

def base_name(col: str) -> str:
    """Strip __log / __sqrt suffix to recover the original base feature name."""
    if col.endswith("__log"):
        return col[:-5]
    if col.endswith("__sqrt"):
        return col[:-6]
    return col


def get_feature_prefix(base_feature: str) -> str:
    for prefix in [GC_PREFIX, LENGTH_PREFIX, QUALITY_PREFIX, KMER_PREFIX]:
        if base_feature == prefix or base_feature.startswith(prefix + "_"):
            return prefix
    return "other"


def get_columns_by_prefix(X: pd.DataFrame, allowed_prefixes: List[str]) -> List[str]:
    return [c for c in X.columns if get_feature_prefix(base_name(c)) in allowed_prefixes]


def extract_kmer_k(base_feature: str) -> Optional[int]:
    """Extract k from a kmer feature named with a numeric suffix pattern like *_5_64."""
    base = base_name(base_feature)
    if get_feature_prefix(base) != KMER_PREFIX:
        return None
    numeric_tokens = [int(tok) for tok in base.split("_") if tok.isdigit()]
    if len(numeric_tokens) < 2:
        return None
    # Naming convention is permissive; for names like ..._5_64, k is the
    # second-to-last numeric token and hash size is the last token.
    return numeric_tokens[-2]


def extract_kmer_hash_size(base_feature: str) -> Optional[int]:
    """Extract hash size from a kmer feature named with a numeric suffix like *_5_64."""
    base = base_name(base_feature)
    if get_feature_prefix(base) != KMER_PREFIX:
        return None
    numeric_tokens = [int(tok) for tok in base.split("_") if tok.isdigit()]
    if len(numeric_tokens) < 2:
        return None
    return numeric_tokens[-1]


# =========================================================
# BASE MODELS
# =========================================================

def get_base_regressors(seed: int = 23) -> List[Tuple[str, object]]:
    """Return base regressors used for full-model random candidate search."""
    return [
        ("LinearRegression", LinearRegression()),
        ("Ridge_a001", Ridge(alpha=0.01)),
        ("Ridge_a005", Ridge(alpha=0.05)),
        ("Ridge_a01", Ridge(alpha=0.1)),
        ("Ridge_a05", Ridge(alpha=0.5)),
        ("Ridge_a1", Ridge(alpha=1.0)),
        ("Ridge_a5", Ridge(alpha=5.0)),
        ("Ridge_a10", Ridge(alpha=10.0)),
        ("Ridge_a50", Ridge(alpha=50.0)),
        ("RF_md4", RandomForestRegressor(n_estimators=100, max_depth=4, random_state=seed, n_jobs=12)),
        ("RF_md6", RandomForestRegressor(n_estimators=150, max_depth=6, random_state=seed, n_jobs=12)),
        ("RF_md8", RandomForestRegressor(n_estimators=200, max_depth=8, random_state=seed, n_jobs=12)),
        ("RF_md10", RandomForestRegressor(n_estimators=250, max_depth=10, random_state=seed, n_jobs=12)),
        ("RF_md12", RandomForestRegressor(n_estimators=300, max_depth=12, random_state=seed, n_jobs=12)),
        ("RF_500", RandomForestRegressor(n_estimators=500, max_depth=6, random_state=seed, n_jobs=12)),
        ("RF_1000", RandomForestRegressor(n_estimators=1000, max_depth=4, random_state=seed, n_jobs=12)),
        ("HGBR_d2_lr01", HistGradientBoostingRegressor(max_iter=150, max_depth=2, learning_rate=0.10, random_state=seed)),
        ("HGBR_d7_lr01", HistGradientBoostingRegressor(max_iter=200, max_depth=7, learning_rate=0.10, random_state=seed)),
        ("HGBR_d3_lr002", HistGradientBoostingRegressor(max_iter=500, max_depth=3, learning_rate=0.02, random_state=seed)),
        ("HGBR_d3_lr02", HistGradientBoostingRegressor(max_iter=120, max_depth=3, learning_rate=0.20, random_state=seed)),
        ("HGBR_d7_lr003", HistGradientBoostingRegressor(max_iter=600, max_depth=7, learning_rate=0.03, random_state=seed)),
        ("HGBR_d4_lr007", HistGradientBoostingRegressor(max_iter=250, max_depth=4, learning_rate=0.07, random_state=seed)),
        ("HGBR_d3_lr001", HistGradientBoostingRegressor(max_iter=1000, max_depth=3, learning_rate=0.01, random_state=seed)),
        ("HGBR_d6_lr015", HistGradientBoostingRegressor(max_iter=150, max_depth=6, learning_rate=0.15, random_state=seed)),
        ("HGBR_reg", HistGradientBoostingRegressor(max_iter=400, max_depth=5, learning_rate=0.03, min_samples_leaf=30, l2_regularization=1.0, early_stopping=True, random_state=seed)),      
        ("MLP_32_16", MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=400, random_state=seed)),
        ("MLP_32", MLPRegressor(hidden_layer_sizes=(32,), max_iter=400, random_state=seed)),
        ("MLP_16_32", MLPRegressor(hidden_layer_sizes=(16, 32), max_iter=400, random_state=seed)),
        ("MLP_16", MLPRegressor(hidden_layer_sizes=(16,), max_iter=400, random_state=seed)),
        ("MLP_32_32", MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=400, random_state=seed)),
        ("MLP_64", MLPRegressor(hidden_layer_sizes=(64,), max_iter=400, random_state=seed)),
    ]


def get_base_classifiers(seed: int = 23) -> List[Tuple[str, object]]:
    """Return base classifiers used for fast-model random candidate search."""
    return [
        ("LogReg_C001", LogisticRegression(C=0.01, max_iter=1000, random_state=seed)),
        ("LogReg_C01", LogisticRegression(C=0.1, max_iter=1000, random_state=seed)),
        ("LogReg_C1", LogisticRegression(C=1.0, max_iter=1000, random_state=seed)),
        ("LogReg_C10", LogisticRegression(C=10.0, max_iter=1000, random_state=seed)),
        ("LogReg_C100", LogisticRegression(C=100.0, max_iter=1000, random_state=seed)),
        ("RidgeClf_a001", RidgeClassifier(alpha=0.01)),
        ("RidgeClf_a01", RidgeClassifier(alpha=0.1)),
        ("RidgeClf_a1", RidgeClassifier(alpha=1.0)),
        ("RidgeClf_a10", RidgeClassifier(alpha=10.0)),
        ("RidgeClf_a100", RidgeClassifier(alpha=100.0)),
        ("RFC_md4", RandomForestClassifier(n_estimators=100, max_depth=4, random_state=seed, n_jobs=12)),
        ("RFC_md6", RandomForestClassifier(n_estimators=150, max_depth=6, random_state=seed, n_jobs=12)),
        ("RFC_md8", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed, n_jobs=12)),
        ("RFC_md10", RandomForestClassifier(n_estimators=250, max_depth=10, random_state=seed, n_jobs=12)),
        ("HGBC_d2_lr01", HistGradientBoostingClassifier(max_iter=150, max_depth=2, learning_rate=0.10, random_state=seed)),
        ("HGBC_d7_lr01", HistGradientBoostingClassifier(max_iter=200, max_depth=7, learning_rate=0.10, random_state=seed)),
        ("HGBC_d3_lr002", HistGradientBoostingClassifier(max_iter=500, max_depth=3, learning_rate=0.02, random_state=seed)),
        ("HGBC_d3_lr02", HistGradientBoostingClassifier(max_iter=120, max_depth=3, learning_rate=0.20, random_state=seed)),
        ("HGBC_d7_lr003", HistGradientBoostingClassifier(max_iter=600, max_depth=7, learning_rate=0.03, random_state=seed)),
        ("HGBC_d4_lr007", HistGradientBoostingClassifier(max_iter=250, max_depth=4, learning_rate=0.07, random_state=seed)),
        ("HGBC_d3_lr001", HistGradientBoostingClassifier(max_iter=1000, max_depth=3, learning_rate=0.01, random_state=seed)),
        ("HGBC_d6_lr015", HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.15, random_state=seed)),
        ("HGBC_reg", HistGradientBoostingClassifier(max_iter=400, max_depth=5, learning_rate=0.03, min_samples_leaf=30, l2_regularization=1.0, early_stopping=True, random_state=seed)),
        ("MLPClf_32_16", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=400, random_state=seed)),
        ("MLPClf_32", MLPClassifier(hidden_layer_sizes=(32,), max_iter=400, random_state=seed)),
        ("MLPClf_16_32", MLPClassifier(hidden_layer_sizes=(16, 32), max_iter=400, random_state=seed)),
        ("MLPClf_64", MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, random_state=seed)),
        ("SGDClf_log", SGDClassifier(loss='log_loss', max_iter=1000, random_state=seed)),
        ("SGDClf_hub", SGDClassifier(loss='modified_huber', max_iter=1000, random_state=seed)),
    ]


# =========================================================
# METRIC / THRESHOLD HELPERS
# =========================================================

def _get_classifier_scores(clf, X: pd.DataFrame) -> np.ndarray:
    """Return positive-class score using predict_proba, then decision_function, then predict."""
    if hasattr(clf, 'predict_proba'):
        return clf.predict_proba(X)[:, 1]
    if hasattr(clf, 'decision_function'):
        return clf.decision_function(X)
    return clf.predict(X).astype(float)


def _search_threshold_for_target_recall(
    scores: np.ndarray,
    y_bin: np.ndarray,
    target_recall: float = TARGET_RECALL,
) -> Tuple[float, float, float, bool]:
    """
    Find threshold with max precision among those reaching recall >= target_recall.

    Note: candidate-level winner selection can apply stricter logic (recall > 0.99),
    but threshold search itself uses >= to avoid missing near-boundary thresholds.
    """
    candidate_thrs = np.unique(scores)
    if scores.min() >= 0.0 and scores.max() <= 1.0:
        candidate_thrs = np.unique(np.concatenate([candidate_thrs, np.arange(0.01, 1.0, 0.01)]))

    best_rec_target, best_prec_target, best_thr_target = -1.0, 0.0, float(np.median(scores))
    best_rec_any, best_prec_any, best_thr_any = -1.0, 0.0, float(np.median(scores))

    for thr in candidate_thrs:
        y_pred = (scores >= thr).astype(int)
        tp = int(((y_pred == 1) & (y_bin == 1)).sum())
        fp = int(((y_pred == 1) & (y_bin == 0)).sum())
        fn = int(((y_pred == 0) & (y_bin == 1)).sum())

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if recall > best_rec_any or (recall == best_rec_any and precision > best_prec_any):
            best_rec_any, best_prec_any, best_thr_any = recall, precision, float(thr)

        if recall >= target_recall:
            if (
                precision > best_prec_target
                or (precision == best_prec_target and recall > best_rec_target)
                or (precision == best_prec_target and recall == best_rec_target and float(thr) > best_thr_target)
            ):
                best_rec_target, best_prec_target, best_thr_target = recall, precision, float(thr)

    if best_rec_target >= 0.0:
        return best_thr_target, best_rec_target, best_prec_target, True
    return best_thr_any, best_rec_any, best_prec_any, False


def compute_binary_metrics(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> Dict[str, float]:
    tp = int(((y_pred_bin == 1) & (y_true_bin == 1)).sum())
    fp = int(((y_pred_bin == 1) & (y_true_bin == 0)).sum())
    tn = int(((y_pred_bin == 0) & (y_true_bin == 0)).sum())
    fn = int(((y_pred_bin == 0) & (y_true_bin == 1)).sum())

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
        'recall': recall,
        'precision': precision,
        'f1': f1,
    }


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


# =========================================================
# SAMPLING / SPLIT HELPERS
# =========================================================

def _sample_indices(
    indices: np.ndarray,
    n: int,
    rng: np.random.Generator,
    context: str = "sampling",
) -> np.ndarray:
    if len(indices) == 0:
        raise ValueError(f"Cannot sample from an empty index set during {context}.")
    with_replacement = n > len(indices)
    return rng.choice(indices, size=n, replace=with_replacement)


def draw_balanced_fast_candidate_subset(
    X: pd.DataFrame,
    y: pd.Series,
    subset_size: int,
    threshold: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Draw balanced (<threshold vs >=threshold) candidate-search subset."""
    rng = np.random.default_rng(seed)
    y_bin = (y.values >= threshold).astype(int)
    pos_idx = np.where(y_bin == 1)[0]
    neg_idx = np.where(y_bin == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Fast-model sampling requires both classes (<85 and >=85).")

    n_pos = subset_size // 2
    n_neg = subset_size - n_pos

    chosen = np.concatenate([
        _sample_indices(pos_idx, n_pos, rng, context="fast candidate positive-class sampling"),
        _sample_indices(neg_idx, n_neg, rng, context="fast candidate negative-class sampling"),
    ])
    rng.shuffle(chosen)
    return X.iloc[chosen].reset_index(drop=True), y.iloc[chosen].reset_index(drop=True)


def _build_balanced_full_binary_indices(y: pd.Series, threshold: float, seed: int) -> np.ndarray:
    """Use all datapoints and oversample the minority class to build a balanced binary dataset."""
    rng = np.random.default_rng(seed)
    y_bin = (y.values >= threshold).astype(int)
    pos_idx = np.where(y_bin == 1)[0]
    neg_idx = np.where(y_bin == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Fast-model final training requires both classes (<85 and >=85).")

    target_per_class = max(len(pos_idx), len(neg_idx))

    if len(pos_idx) < target_per_class:
        pos_extra = _sample_indices(
            pos_idx,
            target_per_class - len(pos_idx),
            rng,
            context="fast final balancing positive oversampling",
        )
        pos_bal = np.concatenate([pos_idx, pos_extra])
    else:
        pos_bal = pos_idx

    if len(neg_idx) < target_per_class:
        neg_extra = _sample_indices(
            neg_idx,
            target_per_class - len(neg_idx),
            rng,
            context="fast final balancing negative oversampling",
        )
        neg_bal = np.concatenate([neg_idx, neg_extra])
    else:
        neg_bal = neg_idx

    balanced_idx = np.concatenate([pos_bal, neg_bal])
    rng.shuffle(balanced_idx)
    return balanced_idx


def _balanced_binary_train_test_split_indices(y_bin: np.ndarray, test_fraction: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y_bin == 1)[0]
    neg_idx = np.where(y_bin == 0)[0]

    n_pos_test = max(1, int(round(len(pos_idx) * test_fraction)))
    n_neg_test = max(1, int(round(len(neg_idx) * test_fraction)))

    pos_test = rng.choice(pos_idx, size=n_pos_test, replace=False)
    neg_test = rng.choice(neg_idx, size=n_neg_test, replace=False)

    test_idx = np.concatenate([pos_test, neg_test])
    train_mask = np.ones(len(y_bin), dtype=bool)
    train_mask[test_idx] = False
    train_idx = np.where(train_mask)[0]

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def draw_fast_final_balanced_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_fraction: float,
    threshold: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict[str, int]]:
    """Build balanced full-data binary dataset and produce a balanced 10% holdout split."""
    balanced_idx = _build_balanced_full_binary_indices(y, threshold, seed)
    X_bal = X.iloc[balanced_idx].reset_index(drop=True)
    y_bal = y.iloc[balanced_idx].reset_index(drop=True)

    y_bal_bin = (y_bal.values >= threshold).astype(int)
    train_idx, test_idx = _balanced_binary_train_test_split_indices(y_bal_bin, test_fraction, seed)

    X_train = X_bal.iloc[train_idx].reset_index(drop=True)
    y_train = y_bal.iloc[train_idx].reset_index(drop=True)
    X_test = X_bal.iloc[test_idx].reset_index(drop=True)
    y_test = y_bal.iloc[test_idx].reset_index(drop=True)

    summary = {
        'original_total': int(len(y)),
        'original_pos': int((y.values >= threshold).sum()),
        'original_neg': int((y.values < threshold).sum()),
        'balanced_total': int(len(y_bal)),
        'balanced_pos': int((y_bal.values >= threshold).sum()),
        'balanced_neg': int((y_bal.values < threshold).sum()),
        'train_total': int(len(y_train)),
        'test_total': int(len(y_test)),
    }
    return X_train, y_train, X_test, y_test, summary


def draw_full_candidate_subset(
    X: pd.DataFrame,
    y: pd.Series,
    subset_size: int,
    low: float,
    high: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Draw candidate-search subset from high-identity range [low, high]."""
    rng = np.random.default_rng(seed)
    mask = (y.values >= low) & (y.values <= high)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError("No rows in the 85–100 range for full-model candidate search.")

    chosen = _sample_indices(idx, subset_size, rng, context="full candidate subset sampling")
    rng.shuffle(chosen)
    return X.iloc[chosen].reset_index(drop=True), y.iloc[chosen].reset_index(drop=True)


def split_regression_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    n = len(X)
    if n < 2:
        raise ValueError("Need at least 2 rows to create train/test split.")
    n_test = max(1, int(round(n * test_fraction)))
    n_test = min(n_test, n - 1)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    return (
        X.iloc[train_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y.iloc[test_idx].reset_index(drop=True),
    )


def _draw_homogeneous_full_subset(
    y_vals: np.ndarray,
    n: int,
    low: float,
    high: float,
    seed: int,
    n_bins: int = 15,
) -> np.ndarray:
    """
    Draw n row indices from y_vals, distributed uniformly across n_bins equal-width
    bins spanning [low, high].

    Args:
        y_vals:  1-D array of target values (e.g., percent identity) to bin.
        n:       Total number of indices to return.
        low:     Lower bound of the sampling range (inclusive).
        high:    Upper bound of the sampling range (inclusive).
        seed:    Random seed for reproducibility.
        n_bins:  Number of equal-width bins to divide [low, high] into.
                 Only bins that contain at least one row are considered "active".

    Returns:
        1-D integer array of length n containing row indices into y_vals.
        Each active bin receives floor(n / n_active) indices; any remainder
        is distributed one-per-bin starting from the first active bin.
        Sampling within a bin uses replacement only when the bin contains
        fewer rows than the per-bin quota.
    """
    rng = np.random.default_rng(seed)
    bin_edges = np.linspace(low, high, n_bins + 1)

    # Collect indices per bin
    bins: List[np.ndarray] = []
    for i in range(n_bins):
        b_low, b_high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_vals >= b_low) & (y_vals <= b_high)
        else:
            mask = (y_vals >= b_low) & (y_vals < b_high)
        idx = np.where(mask)[0]
        if len(idx) > 0:
            bins.append(idx)

    if not bins:
        raise ValueError(f"No data in range [{low}, {high}] for homogeneous sampling.")

    n_active = len(bins)
    base = n // n_active
    extra = n % n_active

    chosen: List[np.ndarray] = []
    for i, bin_idx in enumerate(bins):
        n_bin = base + (1 if i < extra else 0)
        replace = n_bin > len(bin_idx)
        chosen.append(rng.choice(bin_idx, size=n_bin, replace=replace))

    result = np.concatenate(chosen)
    rng.shuffle(result)
    return result


# =========================================================
# RANDOM CANDIDATE HELPERS
# =========================================================

def draw_random_feature_set(
    rng: rnd.Random,
    feature_pool: List[str],
    n_features: int,
    required_pool: List[str] = None,
) -> List[str]:
    """Draw exactly n_features unique features, optionally forcing one from required_pool."""
    if required_pool is None:
        required_pool = []

    if len(feature_pool) < n_features:
        raise ValueError(f"Need at least {n_features} available features; got {len(feature_pool)}.")

    selected = []
    if required_pool:
        must_feature = rng.choice(required_pool)
        selected.append(must_feature)

    remaining_pool = [f for f in feature_pool if f not in selected]
    remaining_n = n_features - len(selected)
    if remaining_n > 0:
        selected.extend(rng.sample(remaining_pool, remaining_n))

    return selected


def generate_random_candidates(
    models: List[Tuple[str, object]],
    feature_pool: List[str],
    n_features: int,
    n_candidates: int,
    seed: int,
    required_feature_pool: List[str] = None,
) -> List[Dict]:
    """Generate random (model family + exact feature set) candidates."""
    rng = rnd.Random(seed)
    candidates: List[Dict] = []
    for candidate_id in range(n_candidates):
        model_name, model = rng.choice(models)
        features = draw_random_feature_set(
            rng=rng,
            feature_pool=feature_pool,
            n_features=n_features,
            required_pool=required_feature_pool or [],
        )
        candidates.append({
            'candidate_id': candidate_id,
            'model_name': model_name,
            'model': model,
            'features': features,
        })
    return candidates


def generate_random_full_candidates_single_k(
    models: List[Tuple[str, object]],
    feature_pool: List[str],
    n_features: int,
    n_candidates: int,
    seed: int,
    required_features: Optional[List[str]] = None,
) -> List[Dict]:
    """Generate full-model candidates constrained to one chosen (k, hash-size) kmer config."""
    rng = rnd.Random(seed)
    required_core = list(dict.fromkeys(required_features or []))

    if len(required_core) > n_features:
        raise ValueError(
            f"Too many required full-model features: {len(required_core)} required for n_features={n_features}."
        )
    missing_required = [f for f in required_core if f not in feature_pool]
    if missing_required:
        raise ValueError(f"Required full-model features are missing from feature pool: {missing_required[:5]}.")

    kmer_like_features: List[str] = []
    non_kmer_pool: List[str] = []
    kmer_groups: Dict[Tuple[int, int], List[str]] = {}
    for feature in feature_pool:
        base = base_name(feature)
        if get_feature_prefix(base) == KMER_PREFIX:
            kmer_like_features.append(feature)
            k_val = extract_kmer_k(base)
            hash_size = extract_kmer_hash_size(base)
            if k_val is not None and hash_size is not None:
                kmer_groups.setdefault((k_val, hash_size), []).append(feature)
        else:
            non_kmer_pool.append(feature)

    valid_k_configs: List[Tuple[int, int]] = []
    for k_config, k_features in kmer_groups.items():
        candidate_pool = non_kmer_pool + k_features
        if len(candidate_pool) < n_features:
            continue
        if any(req_feature not in candidate_pool for req_feature in required_core):
            continue
        remaining_available = [f for f in candidate_pool if f not in required_core]
        if len(remaining_available) < (n_features - len(required_core)):
            continue
        valid_k_configs.append(k_config)

    if not valid_k_configs:
        kmer_sample = ", ".join(kmer_like_features[:5]) if kmer_like_features else "<none>"
        discovered_groups = {
            f"k={k_val},hash={hash_size}": len(v)
            for (k_val, hash_size), v in sorted(kmer_groups.items())
        }
        raise ValueError(
            "Full model requires kmer features grouped by (k, hash), but no valid k/hash groups were found. "
            f"Found {len(kmer_like_features)} kmer-like columns (sample: {kmer_sample}). "
            f"Parsed k/hash groups: {discovered_groups}."
        )

    candidates: List[Dict] = []
    for candidate_id in range(n_candidates):
        model_name, model = rng.choice(models)
        chosen_k, chosen_hash_size = rng.choice(valid_k_configs)
        chosen_k_pool = kmer_groups[(chosen_k, chosen_hash_size)]
        candidate_pool = non_kmer_pool + chosen_k_pool
        selected = list(required_core)
        remaining_pool = [f for f in candidate_pool if f not in selected]

        has_kmer = any(f in chosen_k_pool for f in selected)
        if not has_kmer:
            kmer_options = [f for f in chosen_k_pool if f in remaining_pool]
            if not kmer_options:
                continue
            must_kmer = rng.choice(kmer_options)
            selected.append(must_kmer)
            remaining_pool.remove(must_kmer)

        remaining_n = n_features - len(selected)
        if remaining_n > 0:
            selected.extend(rng.sample(remaining_pool, remaining_n))

        candidates.append({
            'candidate_id': candidate_id,
            'model_name': model_name,
            'model': model,
            'features': selected,
            'chosen_kmer_k': chosen_k,
            'chosen_kmer_hash_size': chosen_hash_size,
            'chosen_kmer_config': {'k_values': [chosen_k], 'hash_sizes': [chosen_hash_size]},
        })
    return candidates


# =========================================================
# CANDIDATE EVALUATION / SELECTION
# =========================================================

def evaluate_and_select_fast_candidate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    candidates: List[Dict],
    target_recall: float,
) -> Tuple[Dict, List[Dict]]:
    """
    Evaluate fast-model candidates; select by precision among recall > target_recall.

    The strict `>` comparison is intentional and follows the issue requirement:
    winner selection is based on precision for candidates with recall > 0.99.
    """
    y_train_bin = (y_train.values >= HIGH_THRESHOLD).astype(int)
    y_eval_bin = (y_eval.values >= HIGH_THRESHOLD).astype(int)

    rows: List[Dict] = []
    for cand in candidates:
        model = clone(cand['model'])
        feats = cand['features']

        model.fit(X_train[feats], y_train_bin)
        eval_scores = _get_classifier_scores(model, X_eval[feats])
        threshold, recall, precision, target_met_ge = _search_threshold_for_target_recall(
            eval_scores,
            y_eval_bin,
            target_recall=target_recall,
        )

        # Strict '>' is intentional: issue requirement asks for recall > 0.99
        # during candidate ranking (not >=), even though threshold search uses >=.
        target_met = bool(recall > target_recall)

        rows.append({
            'candidate_id': cand['candidate_id'],
            'model_name': cand['model_name'],
            'features': feats,
            'threshold': threshold,
            'recall': recall,
            'precision': precision,
            'target_met': target_met,
            'target_met_ge': target_met_ge,
            'model_obj': model,
            'refinement_changes': cand.get('refinement_changes', {}),
        })

    if not rows:
        raise ValueError("No fast-model candidates were successfully evaluated.")

    valid = [r for r in rows if r['target_met']]
    if valid:
        best = max(valid, key=lambda r: (r['precision'], r['recall'], r['model_name']))
    else:
        best = max(rows, key=lambda r: (r['recall'], r['precision'], r['model_name']))

    return best, rows


def evaluate_and_select_full_candidate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
    candidates: List[Dict],
) -> Tuple[Dict, List[Dict]]:
    """
    Evaluate full-model candidates using held-out RMSE on 85–100 identity data.

    Criterion rationale:
      The full model is used only in the high-identity range (85–100), where
      absolute prediction error matters directly for final identity estimation.
      RMSE is chosen as primary selection metric because it penalizes larger
      errors more heavily than MAE, which is appropriate for avoiding outlier
      misses in this high-identity use case.
      Tie-break order is: highest R², then lowest MAE, then model name.
    """
    rows: List[Dict] = []
    for cand in candidates:
        model = clone(cand['model'])
        feats = cand['features']

        model.fit(X_train[feats], y_train.values)
        pred_eval = model.predict(X_eval[feats])
        metrics_eval = compute_regression_metrics(y_eval.values, pred_eval)

        rows.append({
            'candidate_id': cand['candidate_id'],
            'model_name': cand['model_name'],
            'features': feats,
            'eval_r2': metrics_eval['r2'],
            'eval_mae': metrics_eval['mae'],
            'eval_rmse': metrics_eval['rmse'],
            'model_obj': model,
            'chosen_kmer_k': cand.get('chosen_kmer_k'),
            'chosen_kmer_hash_size': cand.get('chosen_kmer_hash_size'),
            'chosen_kmer_config': cand.get('chosen_kmer_config'),
            'refinement_changes': cand.get('refinement_changes', {}),
        })

    if not rows:
        raise ValueError("No full-model candidates were successfully evaluated.")

    # Tie-break order: lowest RMSE, then highest R², then lowest MAE, then name.
    def _full_rank_key(row: Dict) -> Tuple[float, float, float, str]:
        return (row['eval_rmse'], -row['eval_r2'], row['eval_mae'], row['model_name'])

    best = min(rows, key=_full_rank_key)
    return best, rows


# =========================================================
# WINNER REFINEMENT
# =========================================================

def _is_refinable_numeric(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _scaled_numeric_value(base_value, multiplier: float):
    if isinstance(base_value, (int, np.integer)):
        return max(1, int(round(float(base_value) * multiplier)))
    scaled = float(base_value) * multiplier
    if float(base_value) > 0 and scaled <= 0:
        return float(base_value)
    return scaled


def get_refinable_param_names(model) -> List[str]:
    """Return a small set of relevant numeric hyperparameters for local winner refinement."""
    params = model.get_params()
    class_name = type(model).__name__
    preferred_by_model = {
        'HistGradientBoostingClassifier': ['learning_rate', 'max_iter', 'max_depth', 'min_samples_leaf'],
        'HistGradientBoostingRegressor': ['learning_rate', 'max_iter', 'max_depth', 'min_samples_leaf'],
        'RandomForestClassifier': ['n_estimators', 'max_depth', 'min_samples_leaf'],
        'RandomForestRegressor': ['n_estimators', 'max_depth', 'min_samples_leaf'],
        'LogisticRegression': ['C'],
        'Ridge': ['alpha'],
        'RidgeClassifier': ['alpha'],
        'SGDClassifier': ['alpha'],
        'MLPClassifier': ['alpha'],
        'MLPRegressor': ['alpha'],
    }
    preferred = preferred_by_model.get(class_name, [])
    return [p for p in preferred if p in params and _is_refinable_numeric(params[p])]


def build_refinement_candidates(
    winner: Dict,
    multipliers: Tuple[float, ...],
) -> List[Dict]:
    """Build a local hyperparameter neighborhood around the initial winner."""
    base_model = winner['model_obj']
    base_params = base_model.get_params()
    candidates: List[Dict] = [{
        'candidate_id': 0,
        'model_name': winner['model_name'],
        'model': clone(base_model),
        'features': winner['features'],
        'chosen_kmer_k': winner.get('chosen_kmer_k'),
        'chosen_kmer_hash_size': winner.get('chosen_kmer_hash_size'),
        'chosen_kmer_config': winner.get('chosen_kmer_config'),
        'refinement_changes': {},
    }]

    candidate_id = 1
    seen = set()
    for param_name in get_refinable_param_names(base_model):
        base_value = base_params[param_name]
        for multiplier in multipliers:
            scaled_value = _scaled_numeric_value(base_value, multiplier)
            if scaled_value == base_value:
                continue
            key = (param_name, scaled_value)
            if key in seen:
                continue
            seen.add(key)

            tuned_model = clone(base_model).set_params(**{param_name: scaled_value})
            candidates.append({
                'candidate_id': candidate_id,
                'model_name': f"{winner['model_name']}|{param_name}={scaled_value}",
                'model': tuned_model,
                'features': winner['features'],
                'chosen_kmer_k': winner.get('chosen_kmer_k'),
                'chosen_kmer_hash_size': winner.get('chosen_kmer_hash_size'),
                'chosen_kmer_config': winner.get('chosen_kmer_config'),
                'refinement_changes': {param_name: scaled_value},
            })
            candidate_id += 1

    return candidates


def draw_full_refinement_subset(
    X: pd.DataFrame,
    y: pd.Series,
    subset_size: int,
    test_fraction: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, Dict[str, int]]:
    """Draw homogeneous full-model train/eval subsets from 85–100 identity rows."""
    mask = (y.values >= HIGH_THRESHOLD) & (y.values <= 100.0)
    hi_idx = np.where(mask)[0]
    if len(hi_idx) < 2:
        raise ValueError("Need at least 2 rows in 85–100 range for full-model refinement.")

    X_hi = X.iloc[hi_idx].reset_index(drop=True)
    y_hi = y.iloc[hi_idx].reset_index(drop=True)

    n_total = min(subset_size, len(hi_idx))
    n_eval = max(1, int(round(n_total * test_fraction)))
    n_eval = min(n_eval, max(1, n_total // 2))

    eval_local_idx = _draw_homogeneous_full_subset(
        y_hi.values, n_eval, HIGH_THRESHOLD, 100.0, seed=seed,
    )
    eval_unique = np.unique(eval_local_idx)
    remaining_local = np.setdiff1d(np.arange(len(y_hi)), eval_unique)
    if len(remaining_local) == 0:
        raise ValueError("No rows left for full-model refinement training after eval draw.")

    X_remaining = X_hi.iloc[remaining_local].reset_index(drop=True)
    y_remaining = y_hi.iloc[remaining_local].reset_index(drop=True)

    n_train = min(n_total - len(eval_unique), len(remaining_local))
    if n_train < 1:
        n_train = min(len(remaining_local), 1)

    train_local_idx = _draw_homogeneous_full_subset(
        y_remaining.values, n_train, HIGH_THRESHOLD, 100.0, seed=seed + 1,
    )

    X_train = X_remaining.iloc[train_local_idx].reset_index(drop=True)
    y_train = y_remaining.iloc[train_local_idx].reset_index(drop=True)
    X_eval = X_hi.iloc[eval_local_idx].reset_index(drop=True)
    y_eval = y_hi.iloc[eval_local_idx].reset_index(drop=True)

    return X_train, y_train, X_eval, y_eval, {
        'subset_total': int(n_total),
        'train_total': int(len(y_train)),
        'eval_total': int(len(y_eval)),
        'high_identity_pool': int(len(hi_idx)),
    }


def refine_fast_winner(
    X: pd.DataFrame,
    y: pd.Series,
    winner: Dict,
    subset_size: int,
    test_fraction: float,
    seed: int,
) -> Tuple[Dict, Dict]:
    """Refine fast winner on a balanced subset using local hyperparameter multipliers."""
    X_sub, y_sub = draw_balanced_fast_candidate_subset(
        X, y, subset_size=subset_size, threshold=HIGH_THRESHOLD, seed=seed + 301,
    )
    y_sub_bin = (y_sub.values >= HIGH_THRESHOLD).astype(int)
    train_idx, eval_idx = _balanced_binary_train_test_split_indices(y_sub_bin, test_fraction, seed + 302)
    X_train = X_sub.iloc[train_idx].reset_index(drop=True)
    y_train = y_sub.iloc[train_idx].reset_index(drop=True)
    X_eval = X_sub.iloc[eval_idx].reset_index(drop=True)
    y_eval = y_sub.iloc[eval_idx].reset_index(drop=True)

    candidates = build_refinement_candidates(winner, REFINEMENT_MULTIPLIERS)
    refined_winner, _ = evaluate_and_select_fast_candidate(
        X_train, y_train, X_eval, y_eval, candidates, target_recall=TARGET_RECALL,
    )
    summary = {
        'subset_size': int(len(X_sub)),
        'train_size': int(len(X_train)),
        'eval_size': int(len(X_eval)),
        'n_candidates': int(len(candidates)),
        'multipliers': list(REFINEMENT_MULTIPLIERS),
        'initial_model_name': winner['model_name'],
        'selected_model_name': refined_winner['model_name'],
        'selected_changes': refined_winner.get('refinement_changes', {}),
    }
    return refined_winner, summary


def refine_full_winner(
    X: pd.DataFrame,
    y: pd.Series,
    winner: Dict,
    subset_size: int,
    test_fraction: float,
    seed: int,
) -> Tuple[Dict, Dict]:
    """Refine full winner on a homogeneous 85–100 subset using local multipliers."""
    X_train, y_train, X_eval, y_eval, draw_summary = draw_full_refinement_subset(
        X, y, subset_size=subset_size, test_fraction=test_fraction, seed=seed + 401,
    )

    candidates = build_refinement_candidates(winner, REFINEMENT_MULTIPLIERS)
    refined_winner, _ = evaluate_and_select_full_candidate(
        X_train, y_train, X_eval, y_eval, candidates,
    )
    summary = {
        **draw_summary,
        'n_candidates': int(len(candidates)),
        'multipliers': list(REFINEMENT_MULTIPLIERS),
        'initial_model_name': winner['model_name'],
        'selected_model_name': refined_winner['model_name'],
        'selected_changes': refined_winner.get('refinement_changes', {}),
    }
    return refined_winner, summary


# =========================================================
# FINAL RETRAINING
# =========================================================

def _cap_fast_train_balanced(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    threshold: float,
    max_train: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Cap the fast-model training set to at most max_train rows while preserving class balance.

    Rows are selected by taking equal halves from the <threshold and >=threshold classes.
    If one class has fewer than max_train // 2 rows, all of its rows are kept and the
    remaining budget is not redistributed to the other class, preserving balance.
    """
    if len(X_train) <= max_train:
        return X_train, y_train

    rng = np.random.default_rng(seed)
    y_bin = (y_train.values >= threshold).astype(int)
    pos_idx = np.where(y_bin == 1)[0]
    neg_idx = np.where(y_bin == 0)[0]

    per_class = max_train // 2
    pos_take = min(len(pos_idx), per_class)
    neg_take = min(len(neg_idx), per_class)

    pos_sel = rng.choice(pos_idx, size=pos_take, replace=False)
    neg_sel = rng.choice(neg_idx, size=neg_take, replace=False)

    chosen = np.concatenate([pos_sel, neg_sel])
    rng.shuffle(chosen)

    return (
        X_train.iloc[chosen].reset_index(drop=True),
        y_train.iloc[chosen].reset_index(drop=True),
    )


def retrain_fast_model(
    X: pd.DataFrame,
    y: pd.Series,
    winner: Dict,
    test_fraction: float,
    seed: int,
) -> Dict:
    """Retrain winning fast-model configuration on full balanced binary data.

    Training is capped at FAST_RETRAIN_MAX_TRAIN (200,000) rows while preserving
    the <85 vs >=85 class balance.
    """
    X_train, y_train, X_test, y_test, balance_summary = draw_fast_final_balanced_split(
        X,
        y,
        test_fraction=test_fraction,
        threshold=HIGH_THRESHOLD,
        seed=seed,
    )

    X_train, y_train = _cap_fast_train_balanced(
        X_train,
        y_train,
        threshold=HIGH_THRESHOLD,
        max_train=FAST_RETRAIN_MAX_TRAIN,
        seed=seed,
    )

    print(
        f"  Fast model retrain — train: {len(y_train):,} | test: {len(y_test):,} "
        f"(balanced pool: {balance_summary['balanced_total']:,})"
    )

    y_train_bin = (y_train.values >= HIGH_THRESHOLD).astype(int)
    y_test_bin = (y_test.values >= HIGH_THRESHOLD).astype(int)

    model = clone(winner['model_obj'])
    features = winner['features']
    model.fit(X_train[features], y_train_bin)

    train_scores = _get_classifier_scores(model, X_train[features])
    threshold, _, _, _ = _search_threshold_for_target_recall(
        train_scores,
        y_train_bin,
        target_recall=TARGET_RECALL,
    )

    train_pred = (train_scores >= threshold).astype(int)
    test_scores = _get_classifier_scores(model, X_test[features])
    test_pred = (test_scores >= threshold).astype(int)

    return {
        'model': model,
        'model_name': winner['model_name'],
        'features': features,
        'refinement_changes': winner.get('refinement_changes', {}),
        'threshold': float(threshold),
        'target_recall': TARGET_RECALL,
        'train_metrics': compute_binary_metrics(y_train_bin, train_pred),
        'test_metrics': compute_binary_metrics(y_test_bin, test_pred),
        'balance_summary': balance_summary,
    }


def retrain_full_model(
    X: pd.DataFrame,
    y: pd.Series,
    winner: Dict,
    seed: int,
) -> Dict:
    """Retrain winning full-model configuration on a capped, homogeneously-balanced 85–100 split.

    Training is capped at FULL_RETRAIN_MAX_TRAIN (1,000,000) rows; validation uses a
    separate FULL_RETRAIN_VAL_SIZE (10,000) row set.  Both sets are drawn with
    uniform coverage across the 85–100 identity range.
    """
    mask = (y.values >= HIGH_THRESHOLD) & (y.values <= 100.0)
    all_hi_idx = np.where(mask)[0]
    if len(all_hi_idx) < 2:
        raise ValueError("Need at least 2 rows in 85–100 range for full-model retraining.")

    X_hi = X.iloc[all_hi_idx].reset_index(drop=True)
    y_hi = y.iloc[all_hi_idx].reset_index(drop=True)

    # Draw homogeneous validation set first so it is excluded from training.
    # Cap at half the available data so there is always room for a training set.
    n_val = min(FULL_RETRAIN_VAL_SIZE, max(1, len(all_hi_idx) // 2))
    val_local_idx = _draw_homogeneous_full_subset(
        y_hi.values, n_val, HIGH_THRESHOLD, 100.0, seed=seed,
    )
    # Exclude every original row that appears in the validation draw from the
    # training pool.  np.unique handles the rare case where _draw_homogeneous_full_subset
    # used within-bin replacement (possible only when a bin has very few rows).
    val_unique = np.unique(val_local_idx)
    remaining_local = np.setdiff1d(np.arange(len(y_hi)), val_unique)

    X_remaining = X_hi.iloc[remaining_local].reset_index(drop=True)
    y_remaining = y_hi.iloc[remaining_local].reset_index(drop=True)

    # Draw homogeneous training set from the remaining pool.
    n_train = min(FULL_RETRAIN_MAX_TRAIN, len(remaining_local))
    train_local_idx = _draw_homogeneous_full_subset(
        y_remaining.values, n_train, HIGH_THRESHOLD, 100.0, seed=seed + 1,
    )

    X_train = X_remaining.iloc[train_local_idx].reset_index(drop=True)
    y_train = y_remaining.iloc[train_local_idx].reset_index(drop=True)
    X_val = X_hi.iloc[val_local_idx].reset_index(drop=True)
    y_val = y_hi.iloc[val_local_idx].reset_index(drop=True)

    print(f"  Full model retrain — train: {len(y_train):,} | test: {len(y_val):,} "
          f"(high-identity pool: {len(all_hi_idx):,})")

    model = clone(winner['model_obj'])
    features = winner['features']
    model.fit(X_train[features], y_train.values)

    train_pred = model.predict(X_train[features])
    val_pred = model.predict(X_val[features])

    return {
        'model': model,
        'model_name': winner['model_name'],
        'features': features,
        'chosen_kmer_k': winner.get('chosen_kmer_k'),
        'chosen_kmer_hash_size': winner.get('chosen_kmer_hash_size'),
        'chosen_kmer_config': winner.get('chosen_kmer_config'),
        'refinement_changes': winner.get('refinement_changes', {}),
        'selection_metric': 'heldout_rmse_85_100',
        'train_metrics': compute_regression_metrics(y_train.values, train_pred),
        'test_metrics': compute_regression_metrics(y_val.values, val_pred),
        'dataset_summary': {
            'full_high_identity_total': int(len(all_hi_idx)),
            'train_total': int(len(y_train)),
            'test_total': int(len(y_val)),
            'target_range': [85.0, 100.0],
        },
    }


# =========================================================
# PERSISTENCE
# =========================================================

def _json_default(obj):
    """JSON serializer for numpy scalar/array values used in metadata dumps."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def save_final_artifacts(
    output_dir: str,
    fast_result: Dict,
    full_result: Dict,
    fast_search_summary: Dict,
    full_search_summary: Dict,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    fast_model_path = os.path.join(output_dir, "fast_model.pkl")
    full_model_path = os.path.join(output_dir, "full_model.pkl")
    fast_meta_path = os.path.join(output_dir, "fast_model_metadata.json")
    full_meta_path = os.path.join(output_dir, "full_model_metadata.json")

    joblib.dump(fast_result['model'], fast_model_path)
    joblib.dump(full_result['model'], full_model_path)

    fast_metadata = {
        'selected_model_name': fast_result['model_name'],
        'selected_features': fast_result['features'],
        'selected_hyperparameter_changes': fast_result.get('refinement_changes', {}),
        'threshold': fast_result['threshold'],
        'target_recall': fast_result['target_recall'],
        'train_metrics': fast_result['train_metrics'],
        'test_metrics': fast_result['test_metrics'],
        'dataset_summary': fast_result['balance_summary'],
        'candidate_search': fast_search_summary,
    }
    full_metadata = {
        'selected_model_name': full_result['model_name'],
        'selected_features': full_result['features'],
        'selected_kmer_k': full_result.get('chosen_kmer_k'),
        'selected_kmer_hash_size': full_result.get('chosen_kmer_hash_size'),
        'selected_kmer_config': full_result.get('chosen_kmer_config'),
        'selected_hyperparameter_changes': full_result.get('refinement_changes', {}),
        'selection_metric': full_result['selection_metric'],
        'train_metrics': full_result['train_metrics'],
        'test_metrics': full_result['test_metrics'],
        'dataset_summary': full_result['dataset_summary'],
        'candidate_search': full_search_summary,
    }

    with open(fast_meta_path, 'w', encoding='utf-8') as f:
        json.dump(fast_metadata, f, indent=2, default=_json_default)
    with open(full_meta_path, 'w', encoding='utf-8') as f:
        json.dump(full_metadata, f, indent=2, default=_json_default)

    print(f"  ✓ Saved fast model: {fast_model_path}")
    print(f"  ✓ Saved full model: {full_model_path}")
    print(f"  ✓ Saved fast metadata/stats: {fast_meta_path}")
    print(f"  ✓ Saved full metadata/stats: {full_meta_path}")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train NanoPred using a simplified two-model workflow: "
            "fast/full candidate search, winner-local refinement on 100k subsets, "
            "and capped final retraining."
        )
    )
    parser.add_argument('--input', '-i', default='all_pairs_data.csv', help='Path to training CSV.')
    parser.add_argument(
        '--output-dir',
        '-o',
        default='final_models',
        help='Directory for final artifacts (default: final_models).',
    )
    parser.add_argument('--seed', type=int, default=23, help='Random seed (default: 23).')
    parser.add_argument(
        '--test-fraction',
        type=float,
        default=TEST_FRACTION,
        help='Holdout fraction for final retraining evaluation (default: 0.10).',
    )
    parser.add_argument(
        '--candidate-subset-size',
        type=int,
        default=FAST_CANDIDATE_SUBSET_SIZE,
        help='Candidate-search subset size for both models (default: 10000).',
    )
    parser.add_argument(
        '--n-candidates',
        type=int,
        default=N_RANDOM_CANDIDATES,
        help='Number of random candidates to evaluate per model type (default: 2000).',
    )
    args = parser.parse_args()

    print("=" * 72)
    print("NANOPRED TWO-MODEL TRAINING")
    print("=" * 72)

    # 1) Load + feature expansion
    X, y = load_and_prepare_data(args.input)
    print("\nExpanding features (log/sqrt variants)...")
    X = expand_features(X)
    print(f"  Expanded feature count: {X.shape[1]}")

    base_classifiers = get_base_classifiers(seed=args.seed)
    base_regressors = get_base_regressors(seed=args.seed)
    print(f"\nClassifiers available: {len(base_classifiers)}")
    print(f"Regressors available : {len(base_regressors)}")

    # 2) FAST MODEL candidate search
    print("\n" + "=" * 72)
    print("FAST MODEL: candidate search")
    print("=" * 72)

    X_fast_sub, y_fast_sub = draw_balanced_fast_candidate_subset(
        X,
        y,
        subset_size=args.candidate_subset_size,
        threshold=HIGH_THRESHOLD,
        seed=args.seed,
    )
    y_fast_sub_bin = (y_fast_sub.values >= HIGH_THRESHOLD).astype(int)
    train_idx, eval_idx = _balanced_binary_train_test_split_indices(y_fast_sub_bin, TEST_FRACTION, args.seed)

    X_fast_train = X_fast_sub.iloc[train_idx].reset_index(drop=True)
    y_fast_train = y_fast_sub.iloc[train_idx].reset_index(drop=True)
    X_fast_eval = X_fast_sub.iloc[eval_idx].reset_index(drop=True)
    y_fast_eval = y_fast_sub.iloc[eval_idx].reset_index(drop=True)

    print(f"  Candidate subset: {len(X_fast_sub):,} total | train: {len(X_fast_train):,} | eval: {len(X_fast_eval):,}")

    fast_feature_pool = get_columns_by_prefix(X, [GC_PREFIX, LENGTH_PREFIX, QUALITY_PREFIX])
    if len(fast_feature_pool) < FAST_FEATURE_COUNT:
        raise ValueError(
            f"Not enough GC/length/quality features for fast model: {len(fast_feature_pool)} available."
        )

    fast_candidates = generate_random_candidates(
        models=base_classifiers,
        feature_pool=fast_feature_pool,
        n_features=FAST_FEATURE_COUNT,
        n_candidates=args.n_candidates,
        seed=args.seed,
    )

    fast_winner, fast_candidate_rows = evaluate_and_select_fast_candidate(
        X_fast_train,
        y_fast_train,
        X_fast_eval,
        y_fast_eval,
        fast_candidates,
        target_recall=TARGET_RECALL,
    )

    print(
        f"Fast winner: {fast_winner['model_name']} "
        f"| recall={fast_winner['recall']:.4f} "
        f"| precision={fast_winner['precision']:.4f} "
        f"| threshold={fast_winner['threshold']:.4f} "
        f"| target_met(>{TARGET_RECALL})={fast_winner['target_met']}"
    )

    # 3) FULL MODEL candidate search
    print("\n" + "=" * 72)
    print("FULL MODEL: candidate search")
    print("=" * 72)

    X_full_sub, y_full_sub = draw_full_candidate_subset(
        X,
        y,
        subset_size=args.candidate_subset_size,
        low=HIGH_THRESHOLD,
        high=100.0,
        seed=args.seed,
    )
    X_full_train, y_full_train, X_full_eval, y_full_eval = split_regression_train_test(
        X_full_sub,
        y_full_sub,
        test_fraction=TEST_FRACTION,
        seed=args.seed,
    )

    print(f"  Candidate subset: {len(X_full_sub):,} total | train: {len(X_full_train):,} | eval: {len(X_full_eval):,}")

    full_feature_pool = X.columns.tolist()
    full_kmer_pool = [c for c in full_feature_pool if get_feature_prefix(base_name(c)) == KMER_PREFIX]
    if len(full_feature_pool) < FULL_FEATURE_COUNT:
        raise ValueError(
            f"Not enough features for full model: need {FULL_FEATURE_COUNT}, have {len(full_feature_pool)}."
        )
    if not full_kmer_pool:
        raise ValueError("Full model requires at least one kmer feature, but none were found.")

    full_candidates = generate_random_full_candidates_single_k(
        models=base_regressors,
        feature_pool=full_feature_pool,
        n_features=FULL_FEATURE_COUNT,
        n_candidates=args.n_candidates,
        seed=args.seed + 101,
        required_features=fast_winner['features'],
    )

    full_winner, full_candidate_rows = evaluate_and_select_full_candidate(
        X_full_train,
        y_full_train,
        X_full_eval,
        y_full_eval,
        full_candidates,
    )

    print(
        f"Full winner: {full_winner['model_name']} "
        f"| eval_rmse={full_winner['eval_rmse']:.4f} "
        f"| eval_r2={full_winner['eval_r2']:.4f} "
        f"| chosen_k={full_winner.get('chosen_kmer_k')} "
        f"| chosen_hash={full_winner.get('chosen_kmer_hash_size')}"
    )

    # 4) Winner refinement
    print("\n" + "=" * 72)
    print("WINNER REFINEMENT")
    print("=" * 72)

    refined_fast_winner, fast_refinement_summary = refine_fast_winner(
        X,
        y,
        winner=fast_winner,
        subset_size=REFINEMENT_SUBSET_SIZE,
        test_fraction=TEST_FRACTION,
        seed=args.seed,
    )
    print(
        f"  Fast refinement — train: {fast_refinement_summary['train_size']:,} "
        f"| test: {fast_refinement_summary['eval_size']:,} "
        f"| winner: {refined_fast_winner['model_name']}"
    )

    refined_full_winner, full_refinement_summary = refine_full_winner(
        X,
        y,
        winner=full_winner,
        subset_size=REFINEMENT_SUBSET_SIZE,
        test_fraction=TEST_FRACTION,
        seed=args.seed,
    )
    print(
        f"  Full refinement — train: {full_refinement_summary['train_total']:,} "
        f"| test: {full_refinement_summary['eval_total']:,} "
        f"| winner: {refined_full_winner['model_name']} "
        f"| chosen_k={refined_full_winner.get('chosen_kmer_k')} "
        f"| chosen_hash={refined_full_winner.get('chosen_kmer_hash_size')}"
    )

    # 5) Final retraining
    print("\n" + "=" * 72)
    print("FINAL RETRAINING")
    print("=" * 72)

    fast_result = retrain_fast_model(
        X,
        y,
        winner=refined_fast_winner,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    full_result = retrain_full_model(
        X,
        y,
        winner=refined_full_winner,
        seed=args.seed,
    )

    fast_search_summary = {
        'subset_size': int(len(X_fast_sub)),
        'train_size': int(len(X_fast_train)),
        'eval_size': int(len(X_fast_eval)),
        'n_candidates': int(len(fast_candidate_rows)),
        'selection_rule': f'highest precision with recall > {TARGET_RECALL}; fallback highest recall then precision',
        'winner_eval_metrics': {
            'recall': refined_fast_winner['recall'],
            'precision': refined_fast_winner['precision'],
            'threshold': refined_fast_winner['threshold'],
            'target_met': refined_fast_winner['target_met'],
        },
        'winner_refinement': fast_refinement_summary,
    }
    full_search_summary = {
        'subset_size': int(len(X_full_sub)),
        'train_size': int(len(X_full_train)),
        'eval_size': int(len(X_full_eval)),
        'n_candidates': int(len(full_candidate_rows)),
        'selection_rule': (
            'lowest held-out RMSE on 85–100 candidate subset with all fast-model features '
            'and exactly one chosen kmer (k, hash-size) configuration '
            '(tie: higher R², then lower MAE)'
        ),
        'winner_eval_metrics': {
            'rmse': refined_full_winner['eval_rmse'],
            'r2': refined_full_winner['eval_r2'],
            'mae': refined_full_winner['eval_mae'],
            'chosen_kmer_k': refined_full_winner.get('chosen_kmer_k'),
            'chosen_kmer_hash_size': refined_full_winner.get('chosen_kmer_hash_size'),
            'chosen_kmer_config': refined_full_winner.get('chosen_kmer_config'),
        },
        'winner_refinement': full_refinement_summary,
    }

    save_final_artifacts(
        output_dir=args.output_dir,
        fast_result=fast_result,
        full_result=full_result,
        fast_search_summary=fast_search_summary,
        full_search_summary=full_search_summary,
    )

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Fast model: {os.path.join(args.output_dir, 'fast_model.pkl')}")
    print(f"Full model: {os.path.join(args.output_dir, 'full_model.pkl')}")


if __name__ == "__main__":
    main()
