import numpy as np
import pandas as pd
import streamlit as st

from app.config import DATA_PATH


REQUIRED_COLUMNS = {"Возраст", "Пол"}


@st.cache_data
def load_data(path=DATA_PATH) -> pd.DataFrame:
    if path.exists():
        data = pd.read_excel(path)
        missing_columns = REQUIRED_COLUMNS.difference(data.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"В файле данных нет обязательных колонок: {missing}")
        return data

    return create_demo_data()


def create_demo_data(rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    genders = rng.choice(["Женский", "Мужской"], size=rows, p=[0.72, 0.28])
    ages = np.clip(rng.normal(loc=44, scale=10, size=rows).round(), 22, 67).astype(int)
    risk = rng.choice(["Низкий", "Средний", "Высокий"], size=rows, p=[0.58, 0.29, 0.13])
    training = rng.choice(["Проходит", "Запланировано", "Нет"], size=rows, p=[0.45, 0.32, 0.23])

    return pd.DataFrame(
        {
            "id": np.arange(1, rows + 1),
            "Пол": genders,
            "Возраст": ages,
            "Риск увольнения": risk,
            "Повышение квалификации": training,
            "Нагрузка часов": rng.integers(16, 34, size=rows),
        }
    )


def filter_data(df: pd.DataFrame, gender: list[str], age_range: tuple[int, int]) -> pd.DataFrame:
    return df[
        df["Пол"].isin(gender)
        & df["Возраст"].between(age_range[0], age_range[1])
    ]

