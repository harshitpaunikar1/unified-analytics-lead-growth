"""
Lead scoring model for unified analytics and lead growth platform.
Backtested on historical data with threshold optimization and action recommendations.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        roc_auc_score, precision_recall_curve, f1_score, average_precision_score,
    )
    from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class LeadFeatures:
    lead_id: str
    channel: str
    industry: str
    title: str
    company_size: str
    num_page_views: int
    num_email_opens: int
    num_form_fills: int
    days_since_created: int
    num_demo_requests: int
    has_phone: bool
    has_linkedin: bool
    raw_score: float


@dataclass
class ScoredLead:
    lead_id: str
    score: float
    mql_probability: float
    sql_probability: float
    action: str
    segment: str


class LeadScoreFeatureEngineer:
    """Engineers engagement and firmographic features for scoring."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "num_page_views" in df.columns and "days_since_created" in df.columns:
            df["daily_page_views"] = (
                df["num_page_views"] / df["days_since_created"].replace(0, 1)
            )
        if "num_email_opens" in df.columns and "num_form_fills" in df.columns:
            df["engagement_index"] = df["num_email_opens"] * 0.3 + df["num_form_fills"] * 2.0
        if "num_demo_requests" in df.columns:
            df["high_intent"] = (df["num_demo_requests"] > 0).astype(int)
        if "title" in df.columns:
            df["is_decision_maker"] = df["title"].str.lower().str.contains(
                r"\b(ceo|cto|cfo|vp|director|head|founder)\b", na=False
            ).astype(int)
        if "company_size" in df.columns:
            size_map = {"micro": 1, "small": 2, "medium": 3, "large": 4, "enterprise": 5}
            df["company_size_num"] = df["company_size"].str.lower().map(size_map).fillna(2)
        return df


class LeadScoringModel:
    """
    Predicts MQL and SQL conversion probability from behavioral and firmographic features.
    """

    MQL_THRESHOLD = 0.35
    SQL_THRESHOLD = 0.60

    def __init__(self, numeric_features: List[str], categorical_features: List[str]):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.engineer = LeadScoreFeatureEngineer()
        self._model: Optional[Pipeline] = None
        self._mql_threshold = self.MQL_THRESHOLD
        self._sql_threshold = self.SQL_THRESHOLD
        self._feature_names: List[str] = []

    def _preprocessor(self):
        transformers = []
        if self.numeric_features:
            transformers.append(("num", StandardScaler(), self.numeric_features))
        if self.categorical_features:
            transformers.append(("cat", OneHotEncoder(handle_unknown="ignore",
                                                        sparse_output=False),
                                  self.categorical_features))
        return ColumnTransformer(transformers=transformers, remainder="drop")

    def fit(self, df: pd.DataFrame, target_col: str = "converted_mql") -> Dict[str, Any]:
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn required.")
        df = self.engineer.transform(df)
        num_cols = [c for c in self.numeric_features if c in df.columns]
        cat_cols = [c for c in self.categorical_features if c in df.columns]
        df_clean = df.dropna(subset=[target_col])
        for col in num_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        for col in cat_cols:
            df_clean[col] = df_clean[col].fillna("unknown")

        X = df_clean[num_cols + cat_cols]
        y = df_clean[target_col].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        prep = self._preprocessor()
        est = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                          max_depth=4, random_state=42)
        self._model = Pipeline([("preprocessor", prep), ("model", est)])
        self._model.fit(X_train, y_train)

        y_prob = self._model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_prob))
        ap = float(average_precision_score(y_test, y_prob))

        self._mql_threshold, self._sql_threshold = self._optimize_thresholds(y_test, y_prob)

        try:
            cat_names = list(prep.named_transformers_["cat"].get_feature_names_out(cat_cols))
        except Exception:
            cat_names = []
        self._feature_names = num_cols + cat_names

        return {
            "auc": round(auc, 4),
            "average_precision": round(ap, 4),
            "mql_threshold": round(self._mql_threshold, 3),
            "sql_threshold": round(self._sql_threshold, 3),
            "training_rows": len(X_train),
        }

    def _optimize_thresholds(self, y_true: pd.Series,
                              y_prob: np.ndarray) -> Tuple[float, float]:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-9)
        if len(thresholds) > 0:
            best_idx = int(np.argmax(f1_scores[:-1]))
            mql_threshold = float(thresholds[best_idx])
        else:
            mql_threshold = self.MQL_THRESHOLD
        sql_threshold = min(mql_threshold + 0.25, 0.85)
        return mql_threshold, sql_threshold

    def predict_score(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        df = self.engineer.transform(df)
        num_cols = [c for c in self.numeric_features if c in df.columns]
        cat_cols = [c for c in self.categorical_features if c in df.columns]
        probs = self._model.predict_proba(df[num_cols + cat_cols])[:, 1]
        scores = np.round(probs * 100, 1)
        result = df[["lead_id"] if "lead_id" in df.columns else []].copy().reset_index(drop=True)
        result["lead_score"] = scores
        result["mql_probability"] = np.round(probs, 4)
        result["is_mql"] = (probs >= self._mql_threshold).astype(int)
        result["is_sql"] = (probs >= self._sql_threshold).astype(int)
        result["segment"] = pd.cut(
            probs,
            bins=[0, 0.25, 0.5, 0.75, 1.0],
            labels=["cold", "warm", "hot", "ready_to_buy"],
        ).astype(str)
        result["recommended_action"] = result.apply(self._recommend_action, axis=1)
        return result

    def _recommend_action(self, row) -> str:
        prob = row.get("mql_probability", 0)
        if prob >= self._sql_threshold:
            return "assign_to_sales"
        if prob >= self._mql_threshold:
            return "nurture_high_touch"
        if prob >= 0.20:
            return "email_sequence"
        return "content_retargeting"

    def feature_importance(self) -> Optional[pd.DataFrame]:
        if self._model is None:
            return None
        est = self._model.named_steps["model"]
        if not hasattr(est, "feature_importances_"):
            return None
        imp = est.feature_importances_
        names = self._feature_names[:len(imp)]
        return (pd.DataFrame({"feature": names, "importance": imp})
                .sort_values("importance", ascending=False)
                .head(10)
                .reset_index(drop=True))

    def backtest(self, historical_df: pd.DataFrame,
                  date_col: str = "created_at",
                  target_col: str = "converted_mql") -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        scored = self.predict_score(historical_df)
        if target_col in historical_df.columns:
            scored["actual"] = historical_df[target_col].values
            precision_by_segment = (
                scored.groupby("segment")
                .apply(lambda x: (x["actual"] == 1).mean() if len(x) > 0 else 0)
                .reset_index(name="precision")
            )
            return precision_by_segment
        return scored


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(42)
    n = 2000

    channels = ["organic_search", "paid_search", "email", "social", "referral"]
    industries = ["software", "manufacturing", "retail", "healthcare", "finance"]
    titles = ["CEO", "Manager", "Director", "VP", "Analyst", "Engineer"]
    sizes = ["micro", "small", "medium", "large", "enterprise"]

    df = pd.DataFrame({
        "lead_id": [f"lead_{i:05d}" for i in range(n)],
        "channel": rng.choice(channels, n),
        "industry": rng.choice(industries, n),
        "title": rng.choice(titles, n),
        "company_size": rng.choice(sizes, n),
        "num_page_views": rng.integers(0, 50, n),
        "num_email_opens": rng.integers(0, 20, n),
        "num_form_fills": rng.integers(0, 5, n),
        "days_since_created": rng.integers(1, 90, n),
        "num_demo_requests": rng.integers(0, 3, n),
        "has_phone": rng.integers(0, 2, n).astype(bool),
        "has_linkedin": rng.integers(0, 2, n).astype(bool),
        "raw_score": rng.uniform(0, 100, n),
    })
    signal = (df["num_demo_requests"] * 3 + df["num_form_fills"] * 2 + df["num_email_opens"])
    df["converted_mql"] = (signal + rng.normal(0, 2, n) > signal.median()).astype(int)

    model = LeadScoringModel(
        numeric_features=["num_page_views", "num_email_opens", "num_form_fills",
                           "days_since_created", "num_demo_requests", "raw_score"],
        categorical_features=["channel", "industry", "title", "company_size"],
    )

    metrics = model.fit(df, target_col="converted_mql")
    print("Training metrics:", metrics)

    scored = model.predict_score(df.head(10))
    print("\nTop scored leads:")
    print(scored[["lead_score", "mql_probability", "is_mql", "segment", "recommended_action"]]
          .to_string(index=False))

    fi = model.feature_importance()
    if fi is not None:
        print("\nTop 5 features:")
        print(fi.head(5).to_string(index=False))

    print("\nBacktest by segment:")
    bt = model.backtest(df, target_col="converted_mql")
    print(bt.to_string(index=False))
