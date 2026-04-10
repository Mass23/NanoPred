"""
train_model.py - Train a pairwise regression ensemble to predict percent identity
from sequence metrics.

Adapts the pairwise ensemble approach from MACE-laboratory for regression tasks.

Usage:
    python train_model.py --input all_pairs_data.csv --output-dir models/ \
        [--results results.csv] [--seed 23]
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# =========================================================
# DATA LOADING AND PREPARATION
# =========================================================

def load_and_prepare_data(csv_path: str) -> tuple:
    """
    Load all_pairs_data.csv, apply length filtering and remove missing values.

    Filtering rules (applied per pair row):
    - Discard rows where length_min < 900  (i.e. the shorter sequence is < 900 bp)
    - Discard rows where length_max > 2500 (i.e. the longer sequence is > 2500 bp)

    Returns:
        (X, y) where X is a DataFrame of all numeric features and
        y is a Series of real_percent_identity values.
    """
    print(f"Loading data from {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns.")

    # Length filtering
    before = len(df)
    df = df[(df['length_min'] >= 900) & (df['length_max'] <= 2500)]
    print(f"  After length filter (900–2500 bp): {len(df):,} rows "
          f"(removed {before - len(df):,}).")

    # Drop missing values
    before = len(df)
    df = df.dropna()
    print(f"  After dropping NAs: {len(df):,} rows (removed {before - len(df):,}).")

    if len(df) == 0:
        raise ValueError("No data remaining after filtering. Check your CSV file.")

    target = 'real_percent_identity'
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in {csv_path}.")

    y = df[target]

    # Use ALL numeric columns except the target as features
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    feature_cols = [c for c in numeric_cols if c != target]
    X = df[feature_cols]

    print(f"  Features: {len(feature_cols)}, Target: '{target}'")
    print(f"  Target range: [{y.min():.2f}, {y.max():.2f}]  mean={y.mean():.2f}")
    return X, y


# =========================================================
# FEATURE SELECTION
# =========================================================

def select_topn_spearman(X: pd.DataFrame, y: pd.Series, n: int) -> list:
    """
    Select the top-n features by absolute Spearman correlation with the target.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        n: Number of features to select.

    Returns:
        List of column names (top n by |Spearman r|).
    """
    corrs = {}
    for col in X.columns:
        vals = X[col].values
        if np.std(vals) == 0:
            corrs[col] = 0.0
        else:
            r, _ = spearmanr(vals, y.values)
            corrs[col] = abs(r) if not np.isnan(r) else 0.0

    sorted_feats = sorted(corrs, key=corrs.get, reverse=True)
    return sorted_feats[:n]


# =========================================================
# BASE REGRESSORS
# =========================================================

def get_base_regressors(seed: int = 23) -> list:
    """
    Define all base regression models for the ensemble.

    Args:
        seed: Random seed for reproducibility (used for all stochastic models).

    Returns:
        List of (name, model) tuples.
    """
    return [
        # Linear regularised models
        ("Lasso_a1e-3",    Lasso(alpha=1e-3, max_iter=5000, random_state=seed)),
        ("Lasso_a1e-2",    Lasso(alpha=1e-2, max_iter=5000, random_state=seed)),
        ("Lasso_a1e-1",    Lasso(alpha=1e-1, max_iter=5000, random_state=seed)),
        ("Ridge_a1e-1",    Ridge(alpha=1e-1)),
        ("Ridge_a1",       Ridge(alpha=1.0)),
        ("Ridge_a10",      Ridge(alpha=10.0)),
        ("ElasticNet_l1_05", ElasticNet(alpha=1e-2, l1_ratio=0.5, max_iter=5000, random_state=seed)),
        ("ElasticNet_l1_08", ElasticNet(alpha=1e-2, l1_ratio=0.8, max_iter=5000, random_state=seed)),

        # Random Forest
        ("RF_md4",   RandomForestRegressor(n_estimators=100, max_depth=4,  random_state=seed, n_jobs=-1)),
        ("RF_md8",   RandomForestRegressor(n_estimators=200, max_depth=8,  random_state=seed, n_jobs=-1)),
        ("RF_md12",  RandomForestRegressor(n_estimators=300, max_depth=12, random_state=seed, n_jobs=-1)),
        ("RF_md16",  RandomForestRegressor(n_estimators=400, max_depth=16, random_state=seed, n_jobs=-1)),

        # Neural networks
        ("MLP_50_l2",       MLPRegressor(hidden_layer_sizes=(50,),       alpha=1e-2, max_iter=500, random_state=seed)),
        ("MLP_100_nopen",   MLPRegressor(hidden_layer_sizes=(100,),      alpha=0.0,  max_iter=500, random_state=seed)),
        ("MLP_5050_l2",     MLPRegressor(hidden_layer_sizes=(50, 50),    alpha=1e-2, max_iter=500, random_state=seed)),
        ("MLP_100100_l2",   MLPRegressor(hidden_layer_sizes=(100, 100),  alpha=1e-2, max_iter=500, random_state=seed)),

        # Gradient Boosting
        ("GBR_d3_lr01",  GradientBoostingRegressor(n_estimators=200, max_depth=3,  learning_rate=0.1,  random_state=seed)),
        ("GBR_d5_lr01",  GradientBoostingRegressor(n_estimators=200, max_depth=5,  learning_rate=0.1,  random_state=seed)),
        ("GBR_d3_lr005", GradientBoostingRegressor(n_estimators=300, max_depth=3,  learning_rate=0.05, random_state=seed)),
    ]


# =========================================================
# MODEL EVALUATION WITH K-FOLD CV
# =========================================================

def evaluate_models_on_feature_set(
    base_models: list,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_features: int,
    feature_sets: dict,
    seed: int = 23,
) -> list:
    """
    Evaluate all base models on a specific feature set using 3-fold CV.

    Returns:
        List of (name, model, n_features, mean_cv_r2) tuples.
    """
    features = feature_sets[n_features]
    X_sel = X_train[features]

    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    model_results = []

    print(f"  Evaluating base models for top {n_features} features:")
    for name, model in base_models:
        scores = []
        for tr_idx, val_idx in kf.split(X_sel):
            X_tr  = X_sel.iloc[tr_idx]
            X_val = X_sel.iloc[val_idx]
            y_tr  = y_train.iloc[tr_idx]
            y_val = y_train.iloc[val_idx]

            m = clone(model)
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_val)
            scores.append(r2_score(y_val, y_pred))

        mean_r2 = float(np.mean(scores))
        print(f"    {name:25s} CV R² = {mean_r2:.4f}")
        model_results.append((name, model, n_features, mean_r2))

    return model_results


# =========================================================
# ENSEMBLE TRAINING AND EVALUATION
# =========================================================

def _build_oof_matrix(
    top_results: list,
    X: pd.DataFrame,
    y: pd.Series,
    feature_sets: dict,
    n_splits: int = 3,
    seed: int = 23,
) -> np.ndarray:
    """
    Build an out-of-fold prediction matrix for the given models and dataset.

    Args:
        top_results: List of (name, model, n_feat, cv_r2) tuples.
        X: Feature DataFrame.
        y: Target Series.
        feature_sets: Dict mapping n_features -> list of column names.
        n_splits: Number of KFold splits.
        seed: Random seed.

    Returns:
        oof_preds: Array of shape (len(X), len(top_results)).
    """
    top_k = len(top_results)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_preds = np.zeros((len(X), top_k))
    for tr_idx, val_idx in kf.split(X):
        for m_idx, (name, model, n_feat, _) in enumerate(top_results):
            feats = feature_sets[n_feat]
            m = clone(model)
            m.fit(X.iloc[tr_idx][feats], y.iloc[tr_idx])
            oof_preds[val_idx, m_idx] = m.predict(X.iloc[val_idx][feats])
    return oof_preds


def train_and_evaluate_ensemble(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    feature_sets: dict,
    base_models: list,
    top_k: int = 5,
    seed: int = 23,
) -> dict:
    """
    Stage 1: Evaluate base models on 20k train / 10k test, select top-k by CV R².
    Stage 3: Retrain top-k on the entire filtered dataset and combine with a
             Ridge meta-learner trained on out-of-fold predictions from the
             full dataset.

    Args:
        X_train: 20k training features for base model comparison.
        y_train: 20k training target.
        X_test:  10k held-out test features for evaluation.
        y_test:  10k held-out test target.
        X_full:  Entire filtered dataset features (for final retraining).
        y_full:  Entire filtered dataset target.
        feature_sets: Dict mapping n_features -> list of column names.
        base_models:  List of (name, model) tuples.
        top_k:  Number of top base models to include in the ensemble.
        seed:   Random seed.

    Returns:
        dict with trained_models, meta_learner, ensemble_metadata, and metrics.
    """
    # ----- Stage 1a: evaluate all base models on each feature set via CV -----
    all_model_results = []
    for n_features in [10, 20, 30]:
        all_model_results.extend(
            evaluate_models_on_feature_set(
                base_models, X_train, y_train, n_features, feature_sets, seed=seed
            )
        )

    # ----- Stage 1b: select top-k models by CV R² -----
    sorted_results = sorted(all_model_results, key=lambda x: x[3], reverse=True)
    top_results = sorted_results[:top_k]

    print(f"\n  Top {top_k} models selected:")
    for name, _, n_feat, cv_r2 in top_results:
        print(f"    {name:25s} (top {n_feat:2d} feats)  CV R² = {cv_r2:.4f}")

    # ----- Stage 1c: train top-k on 20k set, evaluate individually on 10k -----
    test_base_preds_20k = np.zeros((len(X_test), top_k))
    test_results_per_model = []

    for m_idx, (name, model, n_feat, cv_r2) in enumerate(top_results):
        feats = feature_sets[n_feat]
        m = clone(model)
        m.fit(X_train[feats], y_train)

        y_test_pred = m.predict(X_test[feats])
        test_base_preds_20k[:, m_idx] = y_test_pred

        test_r2   = r2_score(y_test, y_test_pred)
        test_mae  = mean_absolute_error(y_test, y_test_pred)
        test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
        test_results_per_model.append({
            'model_name': name,
            'n_features': n_feat,
            'cv_r2': cv_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'test_rmse': test_rmse,
        })

    # Ensemble evaluation on test set (meta-learner trained on 20k OOF)
    oof_preds_20k = _build_oof_matrix(top_results, X_train, y_train, feature_sets, seed=seed)

    eval_scaler = StandardScaler()
    oof_scaled_20k = eval_scaler.fit_transform(oof_preds_20k)
    eval_meta = Ridge(alpha=1.0)
    eval_meta.fit(oof_scaled_20k, y_train)

    test_base_scaled_20k = eval_scaler.transform(test_base_preds_20k)
    y_ensemble_test = eval_meta.predict(test_base_scaled_20k)

    ensemble_r2   = r2_score(y_test, y_ensemble_test)
    ensemble_mae  = mean_absolute_error(y_test, y_ensemble_test)
    ensemble_rmse = float(np.sqrt(mean_squared_error(y_test, y_ensemble_test)))

    print(f"\n  Ensemble (Ridge meta-learner, evaluated on 10k test set):")
    print(f"    R²   = {ensemble_r2:.4f}")
    print(f"    MAE  = {ensemble_mae:.4f}")
    print(f"    RMSE = {ensemble_rmse:.4f}")

    # ----- Stage 3: retrain top-k models on the ENTIRE filtered dataset -----
    print(f"\n  Retraining top {top_k} models on entire filtered dataset "
          f"({len(X_full):,} rows)...")

    oof_preds_full = _build_oof_matrix(top_results, X_full, y_full, feature_sets, seed=seed)

    # Train the final meta-learner on OOF predictions from the full dataset
    meta_scaler = StandardScaler()
    oof_scaled_full = meta_scaler.fit_transform(oof_preds_full)
    meta_learner = Ridge(alpha=1.0)
    meta_learner.fit(oof_scaled_full, y_full)

    # Fit the final base models on the complete filtered dataset
    trained_models = []
    for name, model, n_feat, cv_r2 in top_results:
        feats = feature_sets[n_feat]
        m = clone(model)
        m.fit(X_full[feats], y_full)
        trained_models.append((name, m, n_feat, feats))
        print(f"    ✓ {name} retrained on full data (top {n_feat} feats)")

    return {
        'trained_models': trained_models,
        'meta_learner': meta_learner,
        'meta_scaler': meta_scaler,
        'test_results_per_model': test_results_per_model,
        'ensemble_metrics': {
            'r2': ensemble_r2,
            'mae': ensemble_mae,
            'rmse': ensemble_rmse,
        },
        'top_results': [(name, n_feat, cv_r2) for name, _, n_feat, cv_r2 in top_results],
    }


# =========================================================
# SAVING MODELS AND RESULTS
# =========================================================

def save_models(output_dir: str, result: dict) -> None:
    """
    Save trained base models, feature sets, and ensemble metadata to disk.

    Layout:
        output_dir/
            <ModelName>_top<N>.pkl          – trained base model
            <ModelName>_top<N>_features.pkl – list of selected feature names
            ensemble_metadata.pkl           – which models/features used + weights
    """
    os.makedirs(output_dir, exist_ok=True)

    for name, model, n_feat, feats in result['trained_models']:
        safe_name = name.replace(' ', '_')
        model_path = os.path.join(output_dir, f"{safe_name}_top{n_feat}.pkl")
        feats_path  = os.path.join(output_dir, f"{safe_name}_top{n_feat}_features.pkl")
        joblib.dump(model, model_path)
        joblib.dump(feats,  feats_path)

    # Save meta-learner
    joblib.dump(result['meta_learner'], os.path.join(output_dir, 'meta_learner.pkl'))
    joblib.dump(result['meta_scaler'],  os.path.join(output_dir, 'meta_scaler.pkl'))

    # Save ensemble metadata
    ensemble_metadata = {
        'models': [(name, n_feat) for name, _, n_feat, _ in result['trained_models']],
        'ensemble_metrics': result['ensemble_metrics'],
        'top_results': result['top_results'],
    }
    joblib.dump(ensemble_metadata, os.path.join(output_dir, 'ensemble_metadata.pkl'))

    print(f"  ✓ Models saved to {output_dir}/")


def save_results(results_path: str, result: dict) -> None:
    """
    Save per-model test results plus a final ensemble row to a CSV.
    """
    rows = list(result['test_results_per_model'])

    # Add ensemble row
    em = result['ensemble_metrics']
    rows.append({
        'model_name': 'Ensemble_Ridge',
        'n_features': 'N/A',
        'cv_r2': float('nan'),
        'test_r2':  em['r2'],
        'test_mae': em['mae'],
        'test_rmse': em['rmse'],
    })

    df = pd.DataFrame(rows, columns=['model_name', 'n_features', 'cv_r2',
                                     'test_r2', 'test_mae', 'test_rmse'])
    df.to_csv(results_path, index=False)
    print(f"  ✓ Results saved to {results_path}")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a pairwise regression ensemble to predict percent identity."
    )
    parser.add_argument(
        '--input', '-i',
        default='all_pairs_data.csv',
        help='Path to all_pairs_data.csv (default: all_pairs_data.csv).',
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
        '--train-size',
        type=int,
        default=20000,
        help='Number of samples to use for base model training (default: 20000).',
    )
    parser.add_argument(
        '--test-size',
        type=int,
        default=10000,
        help='Number of samples to hold out for base model evaluation (default: 10000).',
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top base models to include in the ensemble (default: 5).',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=23,
        help='Random seed (default: 23).',
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PAIRWISE REGRESSION ENSEMBLE TRAINING")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load and prepare data
    # ------------------------------------------------------------------
    print("\n1. Loading and preparing data...")
    X, y = load_and_prepare_data(args.input)

    # ------------------------------------------------------------------
    # 2. Train / test split (fixed 20k train, 10k test)
    # ------------------------------------------------------------------
    print("\n2. Splitting into train/test sets...")
    rng = np.random.default_rng(args.seed)
    n = len(X)
    need = args.train_size + args.test_size
    if n < need:
        raise ValueError(
            f"Not enough data after filtering: {n:,} rows available, "
            f"need at least {need:,} ({args.train_size:,} train + {args.test_size:,} test)."
        )
    all_idx = rng.permutation(n)
    train_idx = all_idx[:args.train_size]
    test_idx  = all_idx[args.train_size:args.train_size + args.test_size]

    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test  = X.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test  = y.iloc[test_idx].reset_index(drop=True)

    print(f"  Train: {len(X_train):,} rows   Test: {len(X_test):,} rows   "
          f"Full (for final ensemble): {n:,} rows")

    # ------------------------------------------------------------------
    # 3. Feature selection (Spearman correlation on training set)
    # ------------------------------------------------------------------
    print("\n3. Selecting features via Spearman correlation...")
    feature_sets = {}
    for n_feat in [10, 20, 30]:
        feature_sets[n_feat] = select_topn_spearman(X_train, y_train, n_feat)
        print(f"  Top {n_feat}: {feature_sets[n_feat]}")

    # ------------------------------------------------------------------
    # 4. Define base regressors
    # ------------------------------------------------------------------
    print("\n4. Defining base regressors...")
    base_models = get_base_regressors(seed=args.seed)
    print(f"  {len(base_models)} base models defined.")

    # ------------------------------------------------------------------
    # 5. Train and evaluate
    # ------------------------------------------------------------------
    print("\n5. Evaluating base models and training ensemble...")
    result = train_and_evaluate_ensemble(
        X_train, y_train, X_test, y_test,
        X, y,           # full filtered dataset – used for final retraining
        feature_sets, base_models,
        top_k=args.top_k,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # 6. Save models and results
    # ------------------------------------------------------------------
    print("\n6. Saving models and results...")
    save_models(args.output_dir, result)
    save_results(args.results, result)

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nPer-model test performance:")
    print(f"  {'Model':<25}  {'Feats':>5}  {'CV R²':>8}  {'Test R²':>8}  "
          f"{'MAE':>8}  {'RMSE':>8}")
    print("  " + "-" * 72)
    for row in result['test_results_per_model']:
        print(
            f"  {row['model_name']:<25}  {row['n_features']:>5}  "
            f"{row['cv_r2']:>8.4f}  {row['test_r2']:>8.4f}  "
            f"{row['test_mae']:>8.4f}  {row['test_rmse']:>8.4f}"
        )

    em = result['ensemble_metrics']
    print(f"\nEnsemble (Ridge meta-learner):")
    print(f"  Test R²   = {em['r2']:.4f}")
    print(f"  Test MAE  = {em['mae']:.4f}")
    print(f"  Test RMSE = {em['rmse']:.4f}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
