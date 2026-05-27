from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SOURCE_PATH = DATA_DIR / "teachers_dz5.csv"
DZ8_PATH = BASE_DIR / "classifier" / "teachers_dz8.csv"
ML_PATH = DATA_DIR / "teachers_ml_yearly.csv"
ML_ENRICHED_PATH = DATA_DIR / "teachers_ml_2022_2023_enriched.csv"
DASHBOARD_YEAR = 2023
DASHBOARD_PATH = DATA_DIR / f"teachers_dashboard_{DASHBOARD_YEAR}.csv"
ENRICHED_ML_YEARS = (2022, 2023)

ID_COLUMNS = ["id"]
YEAR_COLUMNS = ["id", "year"]
LIST_COLUMNS = {"subject", "parallel"}
MAX_COLUMNS = {"class_manager", "is_critical"}
TARGET_COLUMN = "is_out_next_year"
JOIN_COLUMNS = ["id", "year"]
RECOMPUTED_COLUMNS = {
    "n_unique_subjects",
    "min_parallel",
    "max_parallel",
    "n_unique_parallel",
    "n_unique_subjects_log",
}


def recompute_is_out_next_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    years = sorted(df["year"].dropna().unique().tolist())
    ids_by_year = {
        year: set(df.loc[df["year"].eq(year), "id"].dropna())
        for year in years
    }

    df[TARGET_COLUMN] = 0
    for year in years:
        next_year = year + 1
        if next_year not in ids_by_year:
            continue

        year_mask = df["year"].eq(year)
        df.loc[year_mask, TARGET_COLUMN] = (
            ~df.loc[year_mask, "id"].isin(ids_by_year[next_year])
        ).astype(int)

    return df


def unique_join(values: pd.Series) -> str:
    unique_values = values.dropna().drop_duplicates().tolist()
    try:
        unique_values = sorted(unique_values, key=lambda value: float(value))
    except (TypeError, ValueError):
        unique_values = sorted(unique_values, key=lambda value: str(value))
    return " | ".join(map(str, unique_values))


def collapse(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    sort_columns = [column for column in ["id", "year"] if column in df.columns]
    df = df.sort_values(sort_columns)

    scalar_columns = [
        column
        for column in df.columns
        if column
        not in set(group_columns).union(LIST_COLUMNS).union(MAX_COLUMNS).union(RECOMPUTED_COLUMNS)
    ]
    base = df.drop_duplicates(group_columns, keep="last")[group_columns + scalar_columns]

    subjects = (
        df.groupby(group_columns)["subject"]
        .agg(unique_join)
        .rename("subjects")
        .reset_index()
    )
    parallels = (
        df.groupby(group_columns)["parallel"]
        .agg(unique_join)
        .rename("parallels")
        .reset_index()
    )
    counts = (
        df.groupby(group_columns)
        .agg(
            n_unique_subjects=("subject", "nunique"),
            min_parallel=("parallel", "min"),
            max_parallel=("parallel", "max"),
            n_unique_parallel=("parallel", "nunique"),
        )
        .reset_index()
    )
    max_columns = [column for column in MAX_COLUMNS if column in df.columns]
    max_flags = df.groupby(group_columns)[max_columns].max().reset_index() if max_columns else None

    collapsed = base.merge(subjects, on=group_columns, how="left")
    collapsed = collapsed.merge(parallels, on=group_columns, how="left")
    collapsed = collapsed.merge(counts, on=group_columns, how="left")
    if max_flags is not None:
        collapsed = collapsed.merge(max_flags, on=group_columns, how="left")
    collapsed["n_unique_subjects_log"] = np.log1p(collapsed["n_unique_subjects"])

    return collapsed.sort_values(group_columns)


def collapse_by_teacher_year(df: pd.DataFrame) -> pd.DataFrame:
    return collapse(df, YEAR_COLUMNS)


def collapse_dashboard_year(df: pd.DataFrame, year: int = DASHBOARD_YEAR) -> pd.DataFrame:
    year_df = df[df["year"].eq(year)].copy()
    return collapse(year_df, ID_COLUMNS)


def enrich_ml_data(ml_df: pd.DataFrame, dz8_path: Path = DZ8_PATH) -> pd.DataFrame:
    enriched_df = ml_df[ml_df["year"].isin(ENRICHED_ML_YEARS)].copy()
    if not dz8_path.exists():
        return enriched_df

    dz8_df = pd.read_csv(dz8_path)
    new_columns = [
        column
        for column in dz8_df.columns
        if column not in enriched_df.columns and column not in JOIN_COLUMNS
    ]
    dz8_features = dz8_df[JOIN_COLUMNS + new_columns].drop_duplicates(JOIN_COLUMNS)

    return enriched_df.merge(
        dz8_features,
        on=JOIN_COLUMNS,
        how="left",
        validate="one_to_one",
    )


def main() -> None:
    df = pd.read_csv(SOURCE_PATH)
    df = recompute_is_out_next_year(df)
    dashboard_df = collapse_dashboard_year(df)
    ml_df = collapse_by_teacher_year(df)
    ml_enriched_df = enrich_ml_data(ml_df)

    dashboard_df.to_csv(DASHBOARD_PATH, index=False)
    ml_df.to_csv(ML_PATH, index=False)
    ml_enriched_df.to_csv(ML_ENRICHED_PATH, index=False)

    print(f"Dashboard table: {DASHBOARD_PATH} {dashboard_df.shape}")
    print(f"ML table: {ML_PATH} {ml_df.shape}")
    print(f"Enriched ML table: {ML_ENRICHED_PATH} {ml_enriched_df.shape}")


if __name__ == "__main__":
    main()
