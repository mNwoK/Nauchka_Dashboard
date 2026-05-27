import streamlit as st
import traceback

from app.auth import (
    get_role_label,
    get_school_id,
    is_school_director,
    require_role,
    reset_role,
)
from app.charts import (
    render_age_boxplot,
    render_age_distribution,
    render_gender_distribution,
)
from app.config import DATA_PATH
from app.data import (
    apply_education_role_scope,
    apply_role_scope,
    filter_data,
    load_data,
    load_education_data,
)
from classifier.predictions import PredictionLoader
import plotly.express as px
import pandas as pd


st.set_page_config(page_title="Отчет", layout="wide")


def render_metrics(df):
    average_age = f"{df['Возраст'].mean():.0f}" if len(df) > 0 else "0"
    if "is_critical" in df.columns:
        risk_count = df["is_critical"].sum()
    elif "Риск увольнения" in df.columns:
        risk_count = df["Риск увольнения"].eq("Высокий").sum()
    else:
        risk_count = 0
    out_next_year = df["is_out_next_year"].sum() if "is_out_next_year" in df.columns else 0
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
        st.metric(label="Средний возраст", value=average_age)
    with col4:
        if "is_out_next_year" in df.columns:
            st.metric(label="Уходят в след. году", value=f"{out_next_year:,}".replace(",", " "))
        else:
            st.metric(label="Проходят ПК", value=f"{training_share:.1f}%")


def render_filters(df):
    if df.empty:
        st.sidebar.info("Нет данных для фильтрации.")
        return df

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


def render_role_sidebar(role_label: str, school_id: int | None) -> None:
    with st.sidebar:
        st.caption(f"Роль: {role_label}")
        if school_id is not None:
            st.caption(f"Школа №{school_id}")
        if st.button("Сменить роль", use_container_width=True):
            reset_role()
            st.switch_page("main.py")


def main() -> None:
    if not require_role():
        return

    try:
        df = load_data()
        education_df = load_education_data()
        print(f"DEBUG: Loaded data - df shape: {df.shape}, education_df shape: {education_df.shape}")
        print(f"DEBUG: Available years in df: {sorted(df['year'].unique()) if 'year' in df.columns else 'No year column'}")
    except ValueError as error:
        st.error(str(error))
        print(f"ERROR loading data: {str(error)}")
        print(traceback.format_exc())
        st.stop()

    if not DATA_PATH.exists():
        st.info("Файл data/dataset.csv не найден, поэтому используется демонстрационный набор данных.")

    scoped_df = apply_role_scope(df)
    scoped_education_df = apply_education_role_scope(education_df, df)
    role_label = get_role_label()
    school_id = get_school_id()
    render_role_sidebar(role_label, school_id)

    if is_school_director():
        st.title(f"📊 Динамика кадров школы №{school_id}")
        st.caption("Доступны только сотрудники выбранной школы.")
    else:
        st.title("📊 Динамика кадров по всем школам")
        st.caption("Сводный отчет для директора департамента образования.")

    if scoped_df.empty:
        st.warning("Нет данных для выбранной роли и школы.")
        return

    filtered_df = render_filters(scoped_df)

    if filtered_df.empty:
        st.warning("По выбранным фильтрам нет данных.")
        return

    st.subheader("Ключевые показатели")
    render_metrics(filtered_df)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Базовые", "Расширенные", "Кадры"])
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Распределение по возрасту")
            render_age_distribution(filtered_df)
        with col2:
            render_gender_distribution(filtered_df)

        # Show risk share chart (from 2024 predictions if available)
        try:
            preds = PredictionLoader.load_predictions(2024)
            print(f"DEBUG: Loaded predictions shape: {preds.shape}")
            print(f"DEBUG: Prediction columns: {list(preds.columns)}")
            print(f"DEBUG: Prediction risk_level distribution: {preds['risk_level'].value_counts().to_dict()}")
            # Filter predictions to match current role and school scope
            # Get 2024 data from scoped_df and join with predictions
            scoped_2024 = scoped_df[scoped_df['year'] == 2024][['id']]
            print(f"DEBUG: Scoped 2024 data count: {len(scoped_2024)}")
            if not scoped_2024.empty:
                risk_preds = preds[preds['id'].isin(scoped_2024['id'])]
                print(f"DEBUG: Filtered predictions count: {len(risk_preds)}")
                if len(risk_preds) > 0:
                    # compute counts
                    risk_counts = risk_preds['risk_level'].value_counts().reindex(['High','Medium','Low']).fillna(0)
                    fig = px.pie(values=risk_counts.values, names=risk_counts.index, title='Доля по уровням риска (2024)')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info('После фильтрации по роли/школе нет предсказаний за 2024 год.')
                    print(f"DEBUG: No predictions after filtering by role/school. scoped_2024 IDs: {scoped_2024['id'].tolist()[:10]}")
            else:
                st.info('Нет данных за 2024 год для выбранной роли и школы.')
                print(f"DEBUG: No 2024 data found for role/school. scoped_df shape: {scoped_df.shape}")
                if 'year' in scoped_df.columns:
                    print(f"DEBUG: Available years in scoped_df: {sorted(scoped_df['year'].unique())}")
                print(f"DEBUG: scoped_df head:\n{scoped_df.head()}")
        except Exception as e:
            error_msg = f'Предсказания 2024 недоступны для визуализации риска: {str(e)}'
            st.error(error_msg)
            print(f"ERROR in risk share chart: {error_msg}")
            print(traceback.format_exc())

    with tab2:
        render_age_boxplot(filtered_df)
        st.dataframe(filtered_df, use_container_width=True)

    with tab3:
        st.markdown('### Список учителей со средним и высоким риском (2024)')
        try:
            preds = PredictionLoader.load_predictions(2024)
            print(f"DEBUG: Loaded predictions for risk list, shape: {preds.shape}")
            risk_df = preds[preds['risk_level'].isin(['Medium', 'High'])].copy()
            print(f"DEBUG: Risk df count (Medium+High): {len(risk_df)}")
            print(f'DEBUG: scoped_df length: {len(scoped_df)}')
            # Merge with scoped_df to show additional columns (if same ids exist)
            scoped_2024_for_merge = scoped_df[scoped_df['year'] == 2024]
            print(f"DEBUG: Scoped 2024 df for merge count: {len(scoped_2024_for_merge)}")
            merged = scoped_2024_for_merge.merge(risk_df, on='id', how='inner')
            print(f"DEBUG: Merged df count: {len(merged)}")
            # Show selected columns
            cols = ['id', 'school_num', 'age', 'sex', 'attrition_probability', 'risk_level']
            display_cols = [c for c in cols if c in merged.columns]
            print(f"DEBUG: Display columns: {display_cols}")
            st.dataframe(merged[display_cols].sort_values('attrition_probability', ascending=False),
                         use_container_width=True)
        except Exception as e:
            error_msg = f'Нет предсказаний для отображения списка рискованных учителей: {str(e)}'
            st.error(error_msg)
            print(f"ERROR in risk list: {error_msg}")
            print(traceback.format_exc())

        # Top features importance
        st.markdown("### Топ признаков, влияющих на риск увольнения")
        try:
            fi_path = DATA_PATH.parent / 'feature_importance.csv'
            fi = pd.read_csv(fi_path)
            fi_top = fi.head(10)
            fig2 = px.bar(fi_top, x='importance', y='feature', orientation='h', title='Top 10 features')
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            error_msg = f'Feature importance недоступна: {str(e)}'
            st.error(error_msg)
            print(f"ERROR in feature importance: {error_msg}")
            print(traceback.format_exc())

        # List of teachers with Medium and High risk



if __name__ == "__main__":
    main()
