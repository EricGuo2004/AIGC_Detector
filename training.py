from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None


@dataclass
class TrainOutput:
    model_name: str
    model: object
    metrics: Dict[str, float]
    report_text: str
    confusion: np.ndarray


def build_models(multiclass: bool) -> Dict[str, object]:
    models: Dict[str, object] = {}
    if LGBMClassifier is not None:
        objective = "multiclass" if multiclass else "binary"
        models["lightgbm"] = LGBMClassifier(
            objective=objective,
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )

    models["random_forest"] = RandomForestClassifier(
        n_estimators=400, random_state=42, class_weight="balanced_subsample", n_jobs=-1
    )
    # Keep LogisticRegression args minimal for broad sklearn version compatibility.
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
) -> Tuple[TrainOutput, List[TrainOutput]]:
    multiclass = len(np.unique(y_train)) > 2
    candidates = build_models(multiclass=multiclass)
    results: List[TrainOutput] = []

    candidate_items = list(candidates.items())
    for name, model in tqdm(candidate_items, desc="Training candidate models", leave=False):
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
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
        out = pd.DataFrame({"feature": feature_names, "importance": vals})
        return out.sort_values("importance", ascending=False).reset_index(drop=True)
    return pd.DataFrame({"feature": feature_names, "importance": np.zeros(len(feature_names), dtype=float)})
