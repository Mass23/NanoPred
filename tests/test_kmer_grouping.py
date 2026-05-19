import unittest
import sys
import types


def _install_train_model_import_stubs() -> None:
    if "numpy" not in sys.modules:
        np_stub = types.ModuleType("numpy")
        np_stub.ndarray = object
        np_stub.integer = int
        np_stub.floating = float
        np_stub.random = types.SimpleNamespace(Generator=object, default_rng=lambda *args, **kwargs: None)
        sys.modules["numpy"] = np_stub
    if "pandas" not in sys.modules:
        pd_stub = types.ModuleType("pandas")
        pd_stub.DataFrame = object
        pd_stub.Series = object
        sys.modules["pandas"] = pd_stub
    if "joblib" not in sys.modules:
        sys.modules["joblib"] = types.ModuleType("joblib")

    sklearn_stub = sys.modules.setdefault("sklearn", types.ModuleType("sklearn"))

    base_stub = types.ModuleType("sklearn.base")
    base_stub.clone = lambda x: x
    sys.modules["sklearn.base"] = base_stub
    sklearn_stub.base = base_stub

    ensemble_stub = types.ModuleType("sklearn.ensemble")
    linear_stub = types.ModuleType("sklearn.linear_model")
    metrics_stub = types.ModuleType("sklearn.metrics")
    nn_stub = types.ModuleType("sklearn.neural_network")

    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

    for name in [
        "HistGradientBoostingClassifier",
        "HistGradientBoostingRegressor",
        "RandomForestClassifier",
        "RandomForestRegressor",
    ]:
        setattr(ensemble_stub, name, _Dummy)
    for name in [
        "LinearRegression",
        "LogisticRegression",
        "Ridge",
        "RidgeClassifier",
        "SGDClassifier",
    ]:
        setattr(linear_stub, name, _Dummy)
    for name in ["MLPClassifier", "MLPRegressor"]:
        setattr(nn_stub, name, _Dummy)

    metrics_stub.mean_absolute_error = lambda *args, **kwargs: 0.0
    metrics_stub.mean_squared_error = lambda *args, **kwargs: 0.0
    metrics_stub.r2_score = lambda *args, **kwargs: 0.0

    sys.modules["sklearn.ensemble"] = ensemble_stub
    sys.modules["sklearn.linear_model"] = linear_stub
    sys.modules["sklearn.metrics"] = metrics_stub
    sys.modules["sklearn.neural_network"] = nn_stub
    sklearn_stub.ensemble = ensemble_stub
    sklearn_stub.linear_model = linear_stub
    sklearn_stub.metrics = metrics_stub
    sklearn_stub.neural_network = nn_stub


_install_train_model_import_stubs()

import train_model


class TestKmerGrouping(unittest.TestCase):
    def test_extract_kmer_k_uses_second_to_last_numeric_token(self):
        self.assertEqual(train_model.extract_kmer_k("kmer_5_hashjaccard_64"), 5)
        self.assertEqual(train_model.extract_kmer_k("kmer_foo_7_256__log"), 7)
        self.assertIsNone(train_model.extract_kmer_k("kmer_bucket_5"))
        self.assertIsNone(train_model.extract_kmer_k("kmer_5x_hashjaccard_64"))
        self.assertIsNone(train_model.extract_kmer_k("quality_hash_64"))

    def test_extract_kmer_hash_size_uses_last_numeric_token(self):
        self.assertEqual(train_model.extract_kmer_hash_size("kmer_5_hashjaccard_64"), 64)
        self.assertEqual(train_model.extract_kmer_hash_size("kmer_foo_7_256__log"), 256)
        self.assertIsNone(train_model.extract_kmer_hash_size("kmer_bucket_5"))
        self.assertIsNone(train_model.extract_kmer_hash_size("kmer_5x_hashjaccard_64"))
        self.assertIsNone(train_model.extract_kmer_hash_size("quality_hash_64"))

    def test_full_candidate_generation_constrains_to_single_k_hash_and_required_core(self):
        feature_pool = [
            "gc_mean",
            "length_min",
            "quality_hash_64",
            "quality_hash_128",
            "gc_std",
            "kmer_3_hashjaccard_64",
            "kmer_3_hashjaccard_128",
            "kmer_5_hashjaccard_64",
            "kmer_5_hashjaccard_256",
            "kmer_5_hashjaccard_128__sqrt",
        ]
        candidates = train_model.generate_random_full_candidates_single_k(
            models=[("dummy", object())],
            feature_pool=feature_pool,
            n_features=5,
            n_candidates=25,
            seed=23,
            required_features=["gc_mean", "length_min"],
        )

        self.assertTrue(candidates)
        for candidate in candidates:
            chosen_k = candidate["chosen_kmer_k"]
            chosen_hash = candidate["chosen_kmer_hash_size"]
            self.assertIn("gc_mean", candidate["features"])
            self.assertIn("length_min", candidate["features"])
            kmer_features = [
                feat for feat in candidate["features"]
                if train_model.get_feature_prefix(train_model.base_name(feat)) == train_model.KMER_PREFIX
            ]
            self.assertTrue(kmer_features)
            self.assertTrue(
                all(train_model.extract_kmer_k(train_model.base_name(feat)) == chosen_k for feat in kmer_features)
            )
            self.assertTrue(
                all(
                    train_model.extract_kmer_hash_size(train_model.base_name(feat)) == chosen_hash
                    for feat in kmer_features
                )
            )
            self.assertEqual(
                candidate["chosen_kmer_config"],
                {"k_value": chosen_k, "hash_size": chosen_hash},
            )

    def test_full_candidate_generation_reports_debug_when_no_valid_group(self):
        feature_pool = [
            "gc_mean",
            "length_min",
            "quality_hash_64",
            "kmer_hashjaccard",
            "kmer_noise_feature",
        ]
        with self.assertRaisesRegex(ValueError, "sample: .*Parsed k/hash groups:"):
            train_model.generate_random_full_candidates_single_k(
                models=[("dummy", object())],
                feature_pool=feature_pool,
                n_features=4,
                n_candidates=5,
                seed=23,
            )


if __name__ == "__main__":
    unittest.main()
