"""Train and compare credit-risk models, track everything in MLflow, promote the best one.

Dataset: German Credit (OpenML "credit-g"), 1,000 loan applicants labelled good/bad.
Selection uses cross-validation on the training split only; the test split is touched once per model.
"""
import mlflow
import numpy as np
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_NAME = "credit-risk"
# The dataset's own cost matrix: approving a bad applicant costs 5x rejecting a good one.
COST_FN, COST_FP = 5, 1
# Protected attributes (sex/marital status, nationality) are excluded from the features.
EXCLUDED = ["personal_status", "foreign_worker"]

CANDIDATES = {
    "logistic_regression": LogisticRegression(max_iter=2000, C=0.5),
    "random_forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3, random_state=0, n_jobs=-1),
    "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, random_state=0),
}


def load_data():
    X, y = fetch_openml("credit-g", version=1, as_frame=True, return_X_y=True)
    X = X.drop(columns=EXCLUDED)
    X = X.astype({c: str for c in X.select_dtypes(exclude="number").columns})
    return X, (y == "bad").astype(int)  # 1 = default risk


def build_pipeline(clf):
    pre = ColumnTransformer([
        ("num", StandardScaler(), make_column_selector(dtype_include="number")),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), make_column_selector(dtype_exclude="number")),
    ])
    return make_pipeline(pre, clf)


def total_cost(y, proba, threshold):
    flagged = proba >= threshold
    return COST_FN * np.sum((y == 1) & ~flagged) + COST_FP * np.sum((y == 0) & flagged)


def best_threshold(y, proba):
    """Decision threshold minimising the business cost (not 0.5, since errors are asymmetric)."""
    grid = np.round(np.linspace(0.05, 0.95, 91), 2)
    return float(grid[np.argmin([total_cost(y, proba, t) for t in grid])])


def evaluate(y, proba, threshold):
    pred = proba >= threshold
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "cost_per_applicant": total_cost(y, proba, threshold) / len(y),
    }


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    mlflow.set_experiment(MODEL_NAME)

    # Reference point: approve everyone.
    baseline_cost = total_cost(y_test.to_numpy(), np.zeros(len(y_test)), 0.5) / len(y_test)
    results = []
    for name, clf in CANDIDATES.items():
        with mlflow.start_run(run_name=name) as run:
            pipe = build_pipeline(clf)
            oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
            threshold = best_threshold(y_train.to_numpy(), oof)
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]

            metrics = {f"test_{k}": v for k, v in evaluate(y_test.to_numpy(), proba, threshold).items()}
            metrics["cv_roc_auc"] = roc_auc_score(y_train, oof)
            metrics["baseline_cost_per_applicant"] = baseline_cost
            params = {k: v for k, v in clf.get_params().items() if isinstance(v, (int, float, str, bool))}
            mlflow.log_params({"model": name, "threshold": threshold, "excluded_features": ",".join(EXCLUDED), **params})
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                pipe, artifact_path="model",
                signature=infer_signature(X_train, proba),
                metadata={"threshold": threshold},
            )
            results.append((metrics["cv_roc_auc"], name, run.info.run_id, metrics))

    print(f"\n{'model':<24}{'cv_auc':>8}{'test_auc':>10}{'recall':>8}{'cost/app':>10}")
    for cv_auc, name, _, m in sorted(results, reverse=True):
        print(f"{name:<24}{cv_auc:>8.3f}{m['test_roc_auc']:>10.3f}{m['test_recall']:>8.2f}{m['test_cost_per_applicant']:>10.3f}")
    print(f"{'approve everyone':<24}{'':>8}{'':>10}{0:>8.2f}{baseline_cost:>10.3f}")

    _, best_name, best_run, _ = max(results)  # chosen on CV AUC, never on the test set
    version = mlflow.register_model(f"runs:/{best_run}/model", MODEL_NAME)
    MlflowClient().set_registered_model_alias(MODEL_NAME, "production", version.version)
    print(f"\nRegistered {best_name} as {MODEL_NAME} v{version.version} (alias: production)")


if __name__ == "__main__":
    main()
