import streamlit as st

from app.charts import (
    render_age_boxplot,
    render_age_distribution,
    render_gender_distribution,
)
from app.config import DATA_PATH
from app.data import filter_data, load_data


st.set_page_config(page_title="Отчет", layout="wide")


def render_metrics(df):
    risk_count = (
        df["Риск увольнения"].eq("Высокий").sum()
        if "Риск увольнения" in df.columns
        else 0
    )
    training_share = (
        df["Повышение квалификации"].eq("Проходит").mean() * 100
        if "Повышение квалификации" in df.columns and len(df) > 0
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Сотрудников", value=f"{len(df):,}".replace(",", " "))
    with col2:
        st.metric(label="В зоне риска", value=f"{risk_count:,}".replace(",", " "))
    with col3:
        st.metric(label="Средний возраст", value=f"{df['Возраст'].mean():.0f}")
    with col4:
        st.metric(label="Проходят ПК", value=f"{training_share:.1f}%")


def render_filters(df):
    with st.sidebar:
        st.header("Настройка фильтров")

        with st.form("filter_form"):
            gender_options = sorted(df["Пол"].dropna().unique())
            selected_gender = st.multiselect(
                "Выберите пол",
                options=gender_options,
                default=gender_options,
            )
            min_age = int(df["Возраст"].min())
            max_age = int(df["Возраст"].max())
            selected_age = st.slider("Возраст", min_age, max_age, (min_age, max_age))

            submitted = st.form_submit_button(label="Применить фильтры")

    if submitted:
        st.success("Фильтры применены")

    return filter_data(df, selected_gender, selected_age)


def main() -> None:
    st.title("📊 Динамика кадров")

    try:
        df = load_data()
    except ValueError as error:
        st.error(str(error))
        st.stop()

    if not DATA_PATH.exists():
        st.info("Файл data/dataset.csv не найден, поэтому используется демонстрационный набор данных.")

    filtered_df = render_filters(df)

    st.subheader("Ключевые показатели")
    render_metrics(filtered_df)

    st.markdown("---")

    tab1, tab2 = st.tabs(["Базовые", "Расширенные"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Распределение по возрасту")
            render_age_distribution(filtered_df)
        with col2:
            render_gender_distribution(filtered_df)

    with tab2:
        render_age_boxplot(filtered_df)
        st.dataframe(filtered_df, use_container_width=True)


if __name__ == "__main__":
    main()

