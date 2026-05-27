"""
CatBoost classifier for teacher attrition prediction.

Model predicts probability of teacher leaving next year.
Uses 2022 data for training, 2023 data for validation/testing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
)
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings("ignore")


class TeacherAttritionClassifier:
    """CatBoost classifier for predicting teacher attrition."""

    def __init__(self, random_state: int = 42, threshold: float = 0.1):
        """
        Initialize the classifier.

        Args:
            random_state: Random seed for reproducibility
            threshold: Probability threshold for positive class prediction
        """
        self.random_state = random_state
        self.threshold = threshold
        self.model = None
        self.feature_importance = None
        self.categorical_features = None
        self.numerical_features = None
        self.id_column = "id"
        self.target_column = "is_out_next_year"
        self.year_column = "year"

    def _select_features(
        self, df: pd.DataFrame
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Select features for the model.

        Excludes: id, year, is_out_next_year, and text columns with too many unique values
        Keeps categorical features separate.

        Args:
            df: Input dataframe

        Returns:
            Tuple of (all_features, categorical_features, numerical_features)
        """
        # Columns to exclude
        exclude_cols = {
            self.id_column,
            self.year_column,
            self.target_column,
            "subjects",  # Too many unique values (923)
            "speciality",  # Too many unique values (60) and too many missing (20865)
            "academic_degree",  # Too many missing (52815)
            "main_position",  # Covered by main_position_cat
            "category_qual",  # Covered by category_qual_cat and has many missing (12085)
            "parallels",  # Covered by min/max_parallel
            "n_unique_subjects_log",  # Log transform of n_unique_subjects
            "pkpp_prog_types_2023",  # Data leakage - only in 2023
            "pkpp_total_courses",  # 2023 specific
            "pkpp_unique_types",  # 2023 specific
            "school_num",  # School ID, not a predictive feature
            "is_critical",  # Boolean, not useful in this context
            "category_due",  # Has too many missing values
        }

        # Categorical features (kept as is)
        categorical_features = [
            "sex",
            "class_manager",
            "main_position_cat",
            "academic_degree_cat",
            "category_qual_cat",
            "category_active",
            "studied_in_moscow_any",
            "has_higher_edu",
            "has_prestigious_university",
            "school_known",
        ]

        # Numerical features
        numerical_features = [
            "age",
            "experience_age_all",
            "experience_age_pedagogic",
            "experience_age_other",
            "experience_age_pedagogic_vin",
            "category_qual_rank",
            "n_unique_subjects",
            "n_unique_parallel",
            "min_parallel",
            "max_parallel",
            "buildings_num_filled",
            "attrition_rate_2022",
            "load_vs_district",
            "school_size_deviation",
        ]

        # Filter to existing columns in dataframe
        categorical_features = [f for f in categorical_features if f in df.columns]
        numerical_features = [f for f in numerical_features if f in df.columns]

        # Extra safety: remove any feature that contains 'pkpp' (noisy/leaky)
        categorical_features = [f for f in categorical_features if "pkpp" not in f]
        numerical_features = [f for f in numerical_features if "pkpp" not in f]

        all_features = categorical_features + numerical_features

        self.categorical_features = categorical_features
        self.numerical_features = numerical_features

        return all_features, categorical_features, numerical_features

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values with forward filling strategy.

        Args:
            df: Input dataframe

        Returns:
            Dataframe with missing values handled
        """
        df = df.copy()

        # For categorical features: fill with 0 (for binary/float categorical), then convert to string
        for col in self.categorical_features:
            # First fill NaN with 0 for binary features or mode for others
            if df[col].isna().sum() > 0:
                # For binary features (0/1), use 0 as default
                if set(df[col].dropna().unique()).issubset({0.0, 1.0, 0, 1}):
                    df[col] = df[col].fillna(0)
                else:
                    mode_val = df[col].mode()
                    fill_val = mode_val[0] if len(mode_val) > 0 else 0
                    df[col] = df[col].fillna(fill_val)
            
            # Convert to string AFTER filling NaN
            df[col] = df[col].astype(str)
            
            # Additional safety: replace any remaining 'nan' strings with '0'
            df[col] = df[col].replace({'nan': '0'})

        # For numerical features: fill with median
        for col in self.numerical_features:
            if df[col].isna().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)

        return df

    def train(
        self, df: pd.DataFrame, train_year: int = 2022
    ) -> Dict[str, float]:
        """
        Train the model on data from a specific year.

        Args:
            df: Input dataframe with all data
            train_year: Year to use for training (default 2022)

        Returns:
            Dictionary with training metrics
        """
        # Select features
        features, cat_features, num_features = self._select_features(df)

        # Filter training data
        train_df = df[df[self.year_column] == train_year].copy()

        # Handle missing values
        train_df = self._handle_missing_values(train_df)

        # Prepare X and y
        X_train = train_df[features]
        y_train = train_df[self.target_column]

        print(f"Training data shape: {X_train.shape}")
        print(f"Class distribution:\n{y_train.value_counts(normalize=True)}")

        # Train CatBoost model
        self.model = CatBoostClassifier(
            iterations=300,
            learning_rate=0.1,
            max_depth=6,
            loss_function="Logloss",
            random_state=self.random_state,
            verbose=50,
            cat_features=cat_features,
        )

        self.model.fit(X_train, y_train)

        # Store feature importance
        self.feature_importance = pd.DataFrame(
            {
                "feature": features,
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        # Save feature importance to disk for dashboard usage
        try:
            fi_path = Path(__file__).parent.parent / "data" / "feature_importance.csv"
            self.feature_importance.to_csv(fi_path, index=False)
        except Exception:
            pass

        # Save trained model to disk for reuse
        try:
            model_path = Path(__file__).parent / "classifier_model.cbm"
            self.model.save_model(str(model_path))
        except Exception:
            pass

        print("\nTop 10 Most Important Features:")
        print(self.feature_importance.head(10))

        return {"status": "Model trained successfully"}

    def evaluate(self, df: pd.DataFrame, eval_year: int = 2023) -> Dict[str, float]:
        """
        Evaluate model on data from a specific year.

        Args:
            df: Input dataframe with all data
            eval_year: Year to use for evaluation (default 2023)

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        features = self.categorical_features + self.numerical_features

        # Filter evaluation data
        eval_df = df[df[self.year_column] == eval_year].copy()

        # Handle missing values
        eval_df = self._handle_missing_values(eval_df)

        # Prepare X and y
        X_eval = eval_df[features]
        y_eval = eval_df[self.target_column]

        # Get predictions
        y_pred_proba = self.model.predict_proba(X_eval)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)

        # Calculate metrics
        metrics = {
            "year": eval_year,
            "n_samples": len(y_eval),
            "n_positive": y_eval.sum(),
            "positive_rate": y_eval.sum() / len(y_eval),
            "precision": precision_score(y_eval, y_pred, zero_division=0),
            "recall": recall_score(y_eval, y_pred, zero_division=0),
            "f1_score": f1_score(y_eval, y_pred, zero_division=0),
            "f2_score": self._f2_score(y_eval, y_pred),
            "roc_auc": roc_auc_score(y_eval, y_pred_proba),
        }

        print(f"\n=== Evaluation Results for {eval_year} ===")
        print(f"Samples: {metrics['n_samples']}")
        print(f"Positive cases: {metrics['n_positive']} ({metrics['positive_rate']:.1%})")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"F2-Score: {metrics['f2_score']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")

        print(f"\nClassification Report:")
        print(
            classification_report(
                y_eval,
                y_pred,
                target_names=["Staying", "Leaving"],
                zero_division=0,
            )
        )

        return metrics

    def predict(self, df: pd.DataFrame, predict_year: int = 2024) -> pd.DataFrame:
        """
        Make predictions on data from a specific year.

        Args:
            df: Input dataframe with all data
            predict_year: Year to make predictions for (default 2024)

        Returns:
            Dataframe with id, actual data columns, and predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        features = self.categorical_features + self.numerical_features

        # Filter data for prediction year
        pred_df = df[df[self.year_column] == predict_year].copy()

        # Handle missing values
        pred_df = self._handle_missing_values(pred_df)

        # Prepare X
        X_pred = pred_df[features]

        # Get predictions
        y_pred_proba = self.model.predict_proba(X_pred)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)

        # Create result dataframe
        result_df = pred_df[[self.id_column]].copy()
        result_df["attrition_probability"] = y_pred_proba
        result_df["predicted_attrition"] = y_pred
        result_df["risk_level"] = pd.cut(
            y_pred_proba,
            bins=[0, 0.2, 0.5, 1.0],
            labels=["Low", "Medium", "High"],
        )

        print(
            f"\n=== Predictions for {predict_year} ===\nTotal predictions: {len(result_df)}"
        )
        print(f"High risk (prob > 0.5): {(result_df['attrition_probability'] > 0.5).sum()}")
        print(
            f"Medium risk (0.2 < prob <= 0.5): {((result_df['attrition_probability'] > 0.2) & (result_df['attrition_probability'] <= 0.5)).sum()}"
        )
        print(f"Low risk (prob <= 0.2): {(result_df['attrition_probability'] <= 0.2).sum()}")

        return result_df

    @staticmethod
    def _f2_score(y_true, y_pred, beta: float = 2.0) -> float:
        """
        Calculate F-beta score (emphasizes recall).

        Args:
            y_true: True labels
            y_pred: Predicted labels
            beta: Beta parameter (default 2 for F2-score)

        Returns:
            F-beta score
        """
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        if precision + recall == 0:
            return 0.0

        f_beta = (1 + beta**2) * (precision * recall) / (
            (beta**2 * precision) + recall
        )
        return f_beta


def main():
    """Main entry point for model training and evaluation."""
    # Load data
    data_path = Path(__file__).parent.parent / "data" / "teachers_ml_yearly_enriched.csv"

    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)

    print(f"Data shape: {df.shape}")
    print(f"Data types:\n{df.dtypes}\n")

    # Initialize and train classifier
    classifier = TeacherAttritionClassifier()

    print("=" * 50)
    print("TRAINING")
    print("=" * 50)
    classifier.train(df, train_year=2022)

    print("\n" + "=" * 50)
    print("EVALUATION ON 2023 DATA")
    print("=" * 50)
    eval_metrics = classifier.evaluate(df, eval_year=2023)

    print("\n" + "=" * 50)
    print("PREDICTIONS FOR 2024")
    print("=" * 50)
    predictions_2024 = classifier.predict(df, predict_year=2024)

    # Save predictions
    output_path = Path(__file__).parent.parent / "data" / "teachers_2024_attrition_predictions.csv"
    predictions_2024.to_csv(output_path, index=False)
    print(f"\nPredictions saved to {output_path}")

    return classifier, eval_metrics, predictions_2024


if __name__ == "__main__":
    classifier, metrics, predictions = main()
