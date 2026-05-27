from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DASHBOARD_YEAR = 2024
DATA_PATH = DATA_DIR / f"teachers_dashboard_{DASHBOARD_YEAR}.csv"
EDUCATION_DATA_PATH = DATA_DIR / "Образование.xlsx"
