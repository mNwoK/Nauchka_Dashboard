"""
Module for loading attrition predictions and integrating them into the dashboard.
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict


class PredictionLoader:
    """Load and manage attrition predictions for dashboard display."""

    _predictions_cache: Optional[pd.DataFrame] = None
    _cache_year: Optional[int] = None

    @classmethod
    def load_predictions(cls, year: int = 2024) -> pd.DataFrame:
        """
        Load attrition predictions for a specific year.

        Args:
            year: Year to load predictions for (default 2024)

        Returns:
            DataFrame with predictions including id, attrition_probability, predicted_attrition, risk_level
        """
        # Check cache
        if cls._predictions_cache is not None and cls._cache_year == year:
            return cls._predictions_cache

        # Determine file path based on year
        data_dir = Path(__file__).parent.parent / "data"

        if year == 2024:
            file_path = data_dir / "teachers_dashboard_2024_with_predictions.csv"
        else:
            file_path = data_dir / f"teachers_2024_attrition_predictions.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {file_path}")

        # Load predictions
        df = pd.read_csv(file_path)

        # Cache the results
        cls._predictions_cache = df
        cls._cache_year = year

        return df

    @classmethod
    def get_prediction_for_teacher(cls, teacher_id: str, year: int = 2024) -> Optional[Dict]:
        """
        Get attrition prediction for a specific teacher.

        Args:
            teacher_id: Teacher ID
            year: Year (default 2024)

        Returns:
            Dict with prediction info or None if not found
        """
        predictions = cls.load_predictions(year)

        result = predictions[predictions["id"] == teacher_id]
        if result.empty:
            return None

        row = result.iloc[0]
        return {
            "id": row["id"],
            "attrition_probability": float(row["attrition_probability"]),
            "predicted_attrition": int(row["predicted_attrition"]),
            "risk_level": row["risk_level"],
        }

    @classmethod
    def get_risk_statistics(cls, year: int = 2024) -> Dict:
        """
        Get overall risk statistics.

        Args:
            year: Year (default 2024)

        Returns:
            Dict with risk statistics
        """
        predictions = cls.load_predictions(year)

        high_risk = (predictions["attrition_probability"] > 0.5).sum()
        medium_risk = (
            (predictions["attrition_probability"] > 0.2)
            & (predictions["attrition_probability"] <= 0.5)
        ).sum()
        low_risk = (predictions["attrition_probability"] <= 0.2).sum()

        return {
            "total": len(predictions),
            "high_risk": high_risk,
            "medium_risk": medium_risk,
            "low_risk": low_risk,
            "avg_probability": float(predictions["attrition_probability"].mean()),
            "max_probability": float(predictions["attrition_probability"].max()),
            "min_probability": float(predictions["attrition_probability"].min()),
        }

    @classmethod
    def clear_cache(cls):
        """Clear the cached predictions."""
        cls._predictions_cache = None
        cls._cache_year = None
