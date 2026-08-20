import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.linear_model import RidgeCV, LogisticRegression, LassoCV
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


class FeatureSelector:
    """
    Unified Feature Selection Engine for driverfind.
    Supports Filter, Embedded, and Wrapper selection modes with an explicit audit trail log.
    """
    def __init__(
        self,
        mode: str = "wrapper",              # Options: "filter", "embedded", "wrapper"
        target_type: str = "continuous",    # Options: "continuous", "categorical"
        n_features_to_select: int = 3,
        corr_threshold: float = 0.85        # Redundancy cutoff for Filter mode
    ):
        self.mode = mode.lower()
        self.target_type = target_type
        self.k = n_features_to_select
        self.corr_threshold = corr_threshold
        self.audit_log: List[Dict[str, Any]] = []

    def fit_select(
        self, 
        df: pd.DataFrame, 
        target_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes feature selection and returns:
        1. Pruned DataFrame (Selected features + Target)
        2. Full Audit DataFrame dissecting every feature's fate
        """
        self.audit_log.clear()
        
        # Ensure target column exists
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

        X = df.drop(columns=[target_col]).astype(np.float32)
        y = df[target_col].values

        # Route to appropriate feature selection mode
        if self.mode == "filter":
            retained = self._dissect_filter(X, y)
        elif self.mode == "embedded":
            retained = self._dissect_embedded(X, y)
        elif self.mode == "wrapper":
            retained = self._dissect_wrapper(X, y)
        else:
            raise ValueError(f"Unknown selection mode '{self.mode}'. Choose from 'filter', 'embedded', or 'wrapper'.")

        # Record retained features in audit log
        for feat in retained:
            self.audit_log.append({
                "feature": feat,
                "status": "Retained",
                "stage": f"Final Selection ({self.mode.upper()})",
                "elimination_order": 0,
                "metric_value": "N/A",
                "reason": "Isolated as a primary business driver"
            })

        # Format Audit Trail DataFrame
        audit_df = pd.DataFrame(self.audit_log).sort_values(
            by=["status", "elimination_order"], ascending=[True, True]
        ).reset_index(drop=True)

        pruned_df = pd.concat([X[retained], pd.Series(y, name=target_col, index=df.index)], axis=1)
        return pruned_df, audit_df
    
    # 1. FILTER MODE (Pairwise Correlation -> Mutual Information)
    def _dissect_filter(self, X: pd.DataFrame, y: np.ndarray) -> List[str]:
        current_cols = list(X.columns)
        order = 1

        # Phase A: Pairwise Redundancy Sweep
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop_corr = set()
        for col in upper_tri.columns:
            redundant_with = upper_tri.index[upper_tri[col] > self.corr_threshold].tolist()
            if redundant_with:
                to_drop_corr.add(col)
                parent_feat = redundant_with[0]
                r_val = round(float(corr_matrix.loc[col, parent_feat]), 2)
                self.audit_log.append({
                    "feature": col,
                    "status": "Eliminated",
                    "stage": "Filter (Redundancy)",
                    "elimination_order": order,
                    "metric_value": f"r = {r_val}",
                    "reason": f"Redundant metric (> {int(self.corr_threshold*100)}% aligned with '{parent_feat}')"
                })
                order += 1

        remaining = [c for c in current_cols if c not in to_drop_corr]

        # Phase B: Mutual Information Importance Cutoff
        mi_func = mutual_info_classif if self.target_type == "categorical" else mutual_info_regression
        scores = mi_func(X[remaining].values, y, random_state=42)
        score_dict = dict(zip(remaining, scores))

        sorted_feats = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
        retained = [f[0] for f in sorted_feats[:self.k]]

        for feat, score in sorted_feats[self.k:]:
            self.audit_log.append({
                "feature": feat,
                "status": "Eliminated",
                "stage": "Filter (Mutual Info)",
                "elimination_order": order,
                "metric_value": f"MI = {round(float(score), 4)}",
                "reason": "Low univariate association with target"
            })
            order += 1

        return retained

    # 2. EMBEDDED MODE (L1 Lasso Penalty Coefficient Shrinkage)
    def _dissect_embedded(self, X: pd.DataFrame, y: np.ndarray) -> List[str]:
        order = 1
        if self.target_type == "categorical":
            model = LogisticRegression(penalty='l1', solver='saga', max_iter=200, random_state=42)
        else:
            model = LassoCV(cv=3, random_state=42)

        model.fit(X.values, y)
        coefs = np.abs(model.coef_).ravel()
        sorted_feats = sorted(dict(zip(X.columns, coefs)).items(), key=lambda item: item[1], reverse=True)

        retained = [f[0] for f in sorted_feats[:self.k]]
        for feat, weight in sorted_feats[self.k:]:
            self.audit_log.append({
                "feature": feat,
                "status": "Eliminated",
                "stage": "Embedded (Lasso)",
                "elimination_order": order,
                "metric_value": f"Weight = {round(float(weight), 4)}",
                "reason": "Shrunk to ~0 by L1 penalty (Low predictive power)"
            })
            order += 1

        return retained

    # 3. WRAPPER MODE (Iterative Recursive Feature Elimination)
    def _dissect_wrapper(self, X: pd.DataFrame, y: np.ndarray) -> List[str]:
        current_features = list(X.columns)
        order = 1

        while len(current_features) > self.k:
            model = LogisticRegression(max_iter=100) if self.target_type == "categorical" else RidgeCV()
            model.fit(X[current_features].values, y)

            coefs = np.abs(model.coef_).ravel() if hasattr(model, "coef_") else model.feature_importances_
            importances = dict(zip(current_features, coefs))
            
            # Identify weakest remaining feature
            weakest_feat, weight = min(importances.items(), key=lambda x: x[1])

            self.audit_log.append({
                "feature": weakest_feat,
                "status": "Eliminated",
                "stage": f"Wrapper (RFE Round {order})",
                "elimination_order": order,
                "metric_value": f"Rel Impact = {round(float(weight), 4)}",
                "reason": "Weakest relative contributor in model fit"
            })

            current_features.remove(weakest_feat)
            order += 1

        return current_features