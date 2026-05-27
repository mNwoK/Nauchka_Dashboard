import numpy as np
import pandas as pd
import streamlit as st
import traceback

from app.auth import get_school_id, is_school_director
from app.config import DATA_PATH, EDUCATION_DATA_PATH


REQUIRED_COLUMNS = {"Возраст", "Пол"}
EDUCATION_REQUIRED_COLUMNS = {"id"}
SCHOOL_ID_COLUMNS = ("school_num", "ID образовательной организации")
TEACHER_ID_COLUMN = "id"


@st.cache_data
def load_data(path=DATA_PATH) -> pd.DataFrame:
    if path.exists():
        data = read_table(path)
        data = normalize_teacher_columns(data)
        missing_columns = REQUIRED_COLUMNS.difference(data.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"В файле данных нет обязательных колонок: {missing}")
        return data

    return create_demo_data()


def read_table(path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path)


def normalize_teacher_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "Возраст" not in data.columns and "age" in data.columns:
        data["Возраст"] = data["age"]

    if "Пол" not in data.columns and "sex" in data.columns:
        data["Пол"] = data["sex"].map({0: "Женский", 1: "Мужской"}).fillna(data["sex"])

    return data


@st.cache_data
def load_education_data(path=EDUCATION_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=[TEACHER_ID_COLUMN])

    data = pd.read_excel(path)
    missing_columns = EDUCATION_REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"В файле образования нет обязательных колонок: {missing}")
    return data


def get_available_school_ids(df: pd.DataFrame) -> list[int]:
    school_column = get_school_id_column(df)
    if school_column is None:
        return []

    school_ids = pd.to_numeric(df[school_column], errors="coerce").dropna().astype(int)
    return sorted(school_ids.unique().tolist())


def apply_role_scope(df: pd.DataFrame) -> pd.DataFrame:
    if not is_school_director():
        return df

    school_id = get_school_id()
    school_column = get_school_id_column(df)
    if school_id is None or school_column is None:
        return df.iloc[0:0]

    school_ids = pd.to_numeric(df[school_column], errors="coerce")
    return df[school_ids.eq(school_id)]


def get_school_id_column(df: pd.DataFrame) -> str | None:
    for column in SCHOOL_ID_COLUMNS:
        if column in df.columns:
            return column
    return None


def apply_education_role_scope(education_df: pd.DataFrame, teachers_df: pd.DataFrame) -> pd.DataFrame:
    if not is_school_director():
        return education_df

    scoped_teachers = apply_role_scope(teachers_df)
    allowed_teacher_ids = scoped_teachers[TEACHER_ID_COLUMN].dropna().unique()
    return education_df[education_df[TEACHER_ID_COLUMN].isin(allowed_teacher_ids)]


def create_demo_data(rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    genders = rng.choice(["Женский", "Мужской"], size=rows, p=[0.72, 0.28])
    ages = np.clip(rng.normal(loc=44, scale=10, size=rows).round(), 22, 67).astype(int)
    risk = rng.choice(["Низкий", "Средний", "Высокий"], size=rows, p=[0.58, 0.29, 0.13])
    training = rng.choice(["Проходит", "Запланировано", "Нет"], size=rows, p=[0.45, 0.32, 0.23])
    school_ids = rng.choice([101, 478, 523, 1207], size=rows)

    return pd.DataFrame(
        {
            "id": np.arange(1, rows + 1),
            "school_num": school_ids,
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
