from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    from lightgbm import LGBMClassifier
    from lightgbm.basic import LightGBMError
except Exception:  # pragma: no cover
    LGBMClassifier = None
    LightGBMError = None


@dataclass
class TrainOutput:
    model_name: str
    model: object
    metrics: Dict[str, float]
    report_text: str
    confusion: np.ndarray


LGBM_PROFILE_CHOICES = ("baseline", "regularized", "large", "wide")
MODEL_SET_CHOICES = ("all", "lightgbm")
MODEL_ARCHITECTURE_CHOICES = (
    "flat",
    "hierarchical_attribution",
    "pairwise_ovo_attribution",
    "binary_expert_ensemble",
)


def lightgbm_profile_params(profile: str) -> Dict[str, object]:
    if profile not in LGBM_PROFILE_CHOICES:
        raise ValueError(f"Unknown LightGBM profile '{profile}'. Expected one of: {', '.join(LGBM_PROFILE_CHOICES)}")

    base: Dict[str, object] = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced",
        "verbose": -1,
    }
    profiles: Dict[str, Dict[str, object]] = {
        "baseline": {},
        "regularized": {
            "n_estimators": 700,
            "learning_rate": 0.04,
            "min_child_samples": 80,
            "reg_alpha": 0.2,
            "reg_lambda": 2.0,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
        },
        "large": {
            "n_estimators": 1000,
            "learning_rate": 0.03,
            "min_child_samples": 40,
        },
        "wide": {
            "n_estimators": 700,
            "learning_rate": 0.04,
            "num_leaves": 63,
            "min_child_samples": 20,
        },
    }
    out = base.copy()
    out.update(profiles[profile])
    return out


def _lgbm_classifier(
    multiclass: bool,
    lightgbm_device: str,
    lightgbm_profile: str,
    random_state: int = 42,
) -> object:
    if LGBMClassifier is None:
        raise RuntimeError("LightGBM is not available.")
    params = lightgbm_profile_params(lightgbm_profile)
    params["objective"] = "multiclass" if multiclass else "binary"
    params["random_state"] = random_state
    if lightgbm_device != "cpu":
        params["device_type"] = lightgbm_device
    return LGBMClassifier(**params)


def _sanitize_feature_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(X)


def _is_lgbm_model(model: object) -> bool:
    return LGBMClassifier is not None and isinstance(model, LGBMClassifier)


def _is_lgbm_gpu_model(model: object) -> bool:
    if not _is_lgbm_model(model):
        return False
    params = model.get_params()
    return str(params.get("device_type", params.get("device", "cpu"))).lower() == "gpu"


def _is_lgbm_error(exc: Exception) -> bool:
    return LightGBMError is not None and isinstance(exc, LightGBMError)


def _cpu_lgbm_clone(model: object) -> object:
    if not _is_lgbm_model(model):
        return model
    params = model.get_params()
    params.pop("device_type", None)
    params.pop("device", None)
    return LGBMClassifier(**params)


def _fit_estimator(model: object, X: np.ndarray, y: np.ndarray, context: str) -> object:
    X = _sanitize_feature_matrix(X)
    try:
        return model.fit(X, y)
    except Exception as exc:
        if _is_lgbm_gpu_model(model) and _is_lgbm_error(exc):
            print(f"[LightGBM fallback] {context}: GPU training failed; retrying on CPU. Error: {exc}")
            cpu_model = _cpu_lgbm_clone(model)
            return cpu_model.fit(X, y)
        raise


def _predict_proba(model: object, X: np.ndarray, groups: Sequence[str] | None = None) -> np.ndarray | None:
    if hasattr(model, "predict_proba_with_context"):
        return model.predict_proba_with_context(X, groups)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return None


def _predict(model: object, X: np.ndarray, groups: Sequence[str] | None = None) -> np.ndarray:
    if hasattr(model, "predict_with_context"):
        return model.predict_with_context(X, groups)
    return model.predict(X)


def _fit(model: object, X: np.ndarray, y: np.ndarray, groups: Sequence[str] | None = None) -> object:
    if hasattr(model, "fit_with_context"):
        return model.fit_with_context(X, y, groups)
    return _fit_estimator(model, X, y, context=model.__class__.__name__)


class ThresholdCalibratedBinaryClassifier:
    def __init__(self, base_model: object, enabled: bool = False):
        self.base_model = base_model
        self.enabled = enabled
        self.threshold = 0.5

    @property
    def feature_importances_(self) -> np.ndarray:
        return getattr(self.base_model, "feature_importances_", np.asarray([]))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ThresholdCalibratedBinaryClassifier":
        X = _sanitize_feature_matrix(X)
        if self.enabled and len(np.unique(y)) == 2 and min(np.bincount(y.astype(int))) >= 10:
            idx = np.arange(len(y))
            train_idx, cal_idx = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)
            self.base_model = _fit_estimator(
                self.base_model,
                X[train_idx],
                y[train_idx],
                context="threshold calibration base model",
            )
            proba = self.base_model.predict_proba(X[cal_idx])[:, 1]
            best_threshold = 0.5
            best_score = -1.0
            for threshold in np.linspace(0.05, 0.95, 181):
                pred = (proba >= threshold).astype(int)
                score = f1_score(y[cal_idx], pred, average="macro")
                if score > best_score:
                    best_score = score
                    best_threshold = float(threshold)
            self.threshold = best_threshold
        self.base_model = _fit_estimator(self.base_model, X, y, context="threshold calibrated binary model")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.base_model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)


class HierarchicalAttributionClassifier:
    def __init__(self, label_names: List[str], lightgbm_device: str, lightgbm_profile: str):
        self.label_names = label_names
        self.lightgbm_device = lightgbm_device
        self.lightgbm_profile = lightgbm_profile
        self.primary_model: object | None = None
        self.expert_model: object | None = None
        self.group_names: List[str] = []
        self.adm_idx = self._label_idx("ADM")
        self.vqdm_idx = self._label_idx("VQDM")
        self.fallback_model: object | None = None

    def _label_idx(self, target: str) -> int | None:
        for idx, label in enumerate(self.label_names):
            if label.lower() == target.lower():
                return idx
        return None

    @property
    def feature_importances_(self) -> np.ndarray:
        vals = []
        for model in (self.primary_model, self.expert_model, self.fallback_model):
            if model is not None and hasattr(model, "feature_importances_"):
                vals.append(np.asarray(model.feature_importances_, dtype=float))
        return np.mean(vals, axis=0) if vals else np.asarray([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HierarchicalAttributionClassifier":
        if self.adm_idx is None or self.vqdm_idx is None or len(np.unique(y)) < 3:
            self.fallback_model = _lgbm_classifier(True, self.lightgbm_device, self.lightgbm_profile)
            self.fallback_model = _fit_estimator(self.fallback_model, X, y, context="hierarchical attribution fallback")
            return self

        group_labels = []
        for label_id in y:
            if int(label_id) in {self.adm_idx, self.vqdm_idx}:
                group_labels.append("ADM_or_VQDM")
            else:
                group_labels.append(self.label_names[int(label_id)])
        self.group_names = sorted(set(group_labels))
        group_to_id = {name: idx for idx, name in enumerate(self.group_names)}
        y_group = np.asarray([group_to_id[g] for g in group_labels], dtype=np.int64)
        self.primary_model = _lgbm_classifier(len(np.unique(y_group)) > 2, self.lightgbm_device, self.lightgbm_profile)
        self.primary_model = _fit_estimator(self.primary_model, X, y_group, context="hierarchical attribution primary")

        expert_mask = np.isin(y, [self.adm_idx, self.vqdm_idx])
        y_expert = (y[expert_mask] == self.vqdm_idx).astype(int)
        self.expert_model = _lgbm_classifier(False, self.lightgbm_device, self.lightgbm_profile, random_state=43)
        self.expert_model = _fit_estimator(
            self.expert_model,
            X[expert_mask],
            y_expert,
            context="hierarchical attribution ADM-VQDM expert",
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.fallback_model is not None:
            return self.fallback_model.predict_proba(X)
        if self.primary_model is None or self.expert_model is None:
            raise RuntimeError("HierarchicalAttributionClassifier is not fitted.")

        primary = self.primary_model.predict_proba(X)
        expert = self.expert_model.predict_proba(X)
        out = np.zeros((X.shape[0], len(self.label_names)), dtype=np.float64)
        for group_idx, group_name in enumerate(self.group_names):
            group_prob = primary[:, group_idx]
            if group_name == "ADM_or_VQDM":
                out[:, self.adm_idx] += group_prob * expert[:, 0]
                out[:, self.vqdm_idx] += group_prob * expert[:, 1]
            else:
                label_idx = self._label_idx(group_name)
                if label_idx is not None:
                    out[:, label_idx] += group_prob
        return out / (out.sum(axis=1, keepdims=True) + 1e-12)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


class PairwiseOVOAttributionClassifier:
    def __init__(self, label_names: List[str], lightgbm_device: str, lightgbm_profile: str):
        self.label_names = label_names
        self.lightgbm_device = lightgbm_device
        self.lightgbm_profile = lightgbm_profile
        self.models: Dict[tuple[int, int], object] = {}
        self.fallback_model: object | None = None

    @property
    def feature_importances_(self) -> np.ndarray:
        vals = [np.asarray(m.feature_importances_, dtype=float) for m in self.models.values() if hasattr(m, "feature_importances_")]
        if self.fallback_model is not None and hasattr(self.fallback_model, "feature_importances_"):
            vals.append(np.asarray(self.fallback_model.feature_importances_, dtype=float))
        return np.mean(vals, axis=0) if vals else np.asarray([])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PairwiseOVOAttributionClassifier":
        classes = sorted(int(c) for c in np.unique(y))
        if len(classes) < 3:
            self.fallback_model = _lgbm_classifier(False, self.lightgbm_device, self.lightgbm_profile)
            self.fallback_model = _fit_estimator(self.fallback_model, X, y, context="pairwise OVO fallback")
            return self
        for a, b in combinations(classes, 2):
            mask = np.isin(y, [a, b])
            y_pair = (y[mask] == b).astype(int)
            model = _lgbm_classifier(False, self.lightgbm_device, self.lightgbm_profile, random_state=100 + a * 10 + b)
            model = _fit_estimator(model, X[mask], y_pair, context=f"pairwise OVO attribution {a}-vs-{b}")
            self.models[(a, b)] = model
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.fallback_model is not None:
            return self.fallback_model.predict_proba(X)
        out = np.zeros((X.shape[0], len(self.label_names)), dtype=np.float64)
        for (a, b), model in self.models.items():
            proba = model.predict_proba(X)
            out[:, a] += proba[:, 0]
            out[:, b] += proba[:, 1]
        return out / (out.sum(axis=1, keepdims=True) + 1e-12)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


class GeneratorExpertBinaryClassifier:
    def __init__(self, lightgbm_device: str, lightgbm_profile: str, blend: float = 0.65, calibrate_threshold: bool = False):
        self.lightgbm_device = lightgbm_device
        self.lightgbm_profile = lightgbm_profile
        self.blend = blend
        self.global_model = ThresholdCalibratedBinaryClassifier(
            _lgbm_classifier(False, lightgbm_device, lightgbm_profile), enabled=calibrate_threshold
        )
        self.experts: Dict[str, object] = {}

    @property
    def feature_importances_(self) -> np.ndarray:
        vals = []
        if hasattr(self.global_model, "feature_importances_"):
            vals.append(np.asarray(self.global_model.feature_importances_, dtype=float))
        vals.extend(np.asarray(m.feature_importances_, dtype=float) for m in self.experts.values() if hasattr(m, "feature_importances_"))
        return np.mean(vals, axis=0) if vals else np.asarray([])

    def fit_with_context(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Sequence[str] | None = None,
    ) -> "GeneratorExpertBinaryClassifier":
        X = _sanitize_feature_matrix(X)
        self.global_model.fit(X, y)
        if groups is None:
            return self
        group_arr = np.asarray(groups)
        for group in sorted(set(group_arr.tolist())):
            mask = group_arr == group
            if mask.sum() < 20 or len(np.unique(y[mask])) < 2:
                continue
            model = _lgbm_classifier(False, self.lightgbm_device, self.lightgbm_profile, random_state=200 + len(self.experts))
            model = _fit_estimator(model, X[mask], y[mask], context=f"binary generator expert {group}")
            self.experts[str(group)] = model
        return self

    def predict_proba_with_context(self, X: np.ndarray, groups: Sequence[str] | None = None) -> np.ndarray:
        global_proba = self.global_model.predict_proba(X)
        if groups is None:
            return global_proba
        out = global_proba.copy()
        for i, group in enumerate(groups):
            expert = self.experts.get(str(group))
            if expert is None:
                continue
            out[i] = (1.0 - self.blend) * global_proba[i] + self.blend * expert.predict_proba(X[i : i + 1])[0]
        return out

    def predict_with_context(self, X: np.ndarray, groups: Sequence[str] | None = None) -> np.ndarray:
        return np.argmax(self.predict_proba_with_context(X, groups), axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba_with_context(X, None)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_with_context(X, None)


def build_models(
    multiclass: bool,
    label_names: List[str],
    lightgbm_device: str = "cpu",
    lightgbm_profile: str = "baseline",
    model_set: str = "all",
    model_architecture: str = "flat",
    calibrate_threshold: bool = False,
) -> Dict[str, object]:
    if model_set not in MODEL_SET_CHOICES:
        raise ValueError(f"Unknown model set '{model_set}'. Expected one of: {', '.join(MODEL_SET_CHOICES)}")
    if model_architecture not in MODEL_ARCHITECTURE_CHOICES:
        raise ValueError(
            f"Unknown model architecture '{model_architecture}'. "
            f"Expected one of: {', '.join(MODEL_ARCHITECTURE_CHOICES)}"
        )

    models: Dict[str, object] = {}
    if LGBMClassifier is None:
        if model_set == "lightgbm" or model_architecture != "flat":
            raise RuntimeError("LightGBM is not available for the requested model configuration.")
    else:
        if model_architecture == "hierarchical_attribution" and multiclass:
            models["lightgbm_hierarchical"] = HierarchicalAttributionClassifier(label_names, lightgbm_device, lightgbm_profile)
            return models
        if model_architecture == "pairwise_ovo_attribution" and multiclass:
            models["lightgbm_pairwise_ovo"] = PairwiseOVOAttributionClassifier(label_names, lightgbm_device, lightgbm_profile)
            return models
        if model_architecture == "binary_expert_ensemble" and not multiclass:
            models["lightgbm_binary_expert_ensemble"] = GeneratorExpertBinaryClassifier(
                lightgbm_device, lightgbm_profile, calibrate_threshold=calibrate_threshold
            )
            return models

        lgbm = _lgbm_classifier(multiclass, lightgbm_device, lightgbm_profile)
        if not multiclass and calibrate_threshold:
            lgbm = ThresholdCalibratedBinaryClassifier(lgbm, enabled=True)
        models["lightgbm"] = lgbm

    if model_set == "lightgbm":
        if "lightgbm" not in models:
            raise RuntimeError("LightGBM is not available, but --model-set lightgbm was requested.")
        return models

    if model_architecture == "flat":
        models["random_forest"] = RandomForestClassifier(
            n_estimators=400, random_state=42, class_weight="balanced_subsample", n_jobs=-1
        )
        models["logreg"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    return models


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }
    unique_classes = np.unique(y_true)
    if y_proba is not None and len(unique_classes) == 2:
        pos_col = 1 if y_proba.shape[1] > 1 else 0
        out["auc"] = float(roc_auc_score(y_true, y_proba[:, pos_col]))
    return out


def train_and_select(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    label_names: List[str],
    lightgbm_device: str = "cpu",
    lightgbm_profile: str = "baseline",
    model_set: str = "all",
    model_architecture: str = "flat",
    calibrate_threshold: bool = False,
    train_groups: Sequence[str] | None = None,
    val_groups: Sequence[str] | None = None,
) -> Tuple[TrainOutput, List[TrainOutput]]:
    X_train = _sanitize_feature_matrix(X_train)
    X_val = _sanitize_feature_matrix(X_val)
    multiclass = len(np.unique(y_train)) > 2
    candidates = build_models(
        multiclass=multiclass,
        label_names=label_names,
        lightgbm_device=lightgbm_device,
        lightgbm_profile=lightgbm_profile,
        model_set=model_set,
        model_architecture=model_architecture,
        calibrate_threshold=calibrate_threshold,
    )
    results: List[TrainOutput] = []

    candidate_items = list(candidates.items())
    for name, model in tqdm(candidate_items, desc="Training candidate models", leave=False):
        model = _fit(model, X_train, y_train, train_groups)
        pred = _predict(model, X_val, val_groups)
        proba = _predict_proba(model, X_val, val_groups)
        metrics = evaluate_predictions(y_val, pred, proba)
        report = classification_report(y_val, pred, target_names=label_names, digits=4)
        cm = confusion_matrix(y_val, pred)
        results.append(
            TrainOutput(
                model_name=name,
                model=model,
                metrics=metrics,
                report_text=report,
                confusion=cm,
            )
        )

    results = sorted(results, key=lambda x: (x.metrics.get("macro_f1", 0.0), x.metrics.get("accuracy", 0.0)), reverse=True)
    return results[0], results


def feature_importance_df(model: object, feature_names: List[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(model.feature_importances_, dtype=float)
        if vals.size == len(feature_names):
            out = pd.DataFrame({"feature": feature_names, "importance": vals})
            return out.sort_values("importance", ascending=False).reset_index(drop=True)
    return pd.DataFrame({"feature": feature_names, "importance": np.zeros(len(feature_names), dtype=float)})
