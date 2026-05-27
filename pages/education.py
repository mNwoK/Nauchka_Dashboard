import streamlit as st
import traceback
import plotly.express as px

from app.auth import (
    get_role_label,
    get_school_id,
    is_school_director,
    require_role,
    reset_role,
)
from app.config import DATA_PATH, EDUCATION_DATA_PATH
from app.data import (
    apply_education_role_scope,
    apply_role_scope,
    load_education_data,
    load_data,
)
import pandas as pd


st.set_page_config(page_title="Образование", layout="wide")


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
    except ValueError as error:
        st.error(str(error))
        print(f"ERROR loading data: {str(error)}")
        print(traceback.format_exc())
        st.stop()

    scoped_df = apply_role_scope(df)
    scoped_education_df = apply_education_role_scope(education_df, df)
    role_label = get_role_label()
    school_id = get_school_id()
    render_role_sidebar(role_label, school_id)

    if is_school_director():
        st.title(f"📚 Образование школы №{school_id}")
        st.caption("Доступны только сотрудники выбранной школы.")
    else:
        st.title("📚 Образование по всем школам")
        st.caption("Сводный отчет для директора департамента образования.")

    if scoped_education_df.empty:
        st.warning("Нет данных об образовании для выбранной роли и школы.")
        return

    # Charts for education metrics
    st.markdown("### 📊 Показатели образования")
    col1, col2 = st.columns(2)
    
    with col1:
        # Доля людей с активной научной степенью
        if 'category_active' in scoped_df.columns:
            active_count = (scoped_df['category_active'] == 'активно').sum()
            total_count = len(scoped_df)
            active_share = active_count / total_count * 100 if total_count > 0 else 0
            fig_active = px.pie(values=[active_count, total_count - active_count], 
                               names=['Активная степень', 'Нет активной степени'],
                               title=f'Доля людей с активной степенью: {active_share:.1f}%')
            st.plotly_chart(fig_active, use_container_width=True)
        else:
            st.info("Колонка 'category_active' не найдена в данных об образовании.")
    
    with col2:
        # Доля людей с высшем образованием
        # Assuming column name for higher education might be 'Высшее образование' or similar
        # Let's try common variations
        higher_ed_cols = [col for col in scoped_df.columns if 'высш' in col.lower() or 'higher' in col.lower() or 'education' in col.lower()]
        if higher_ed_cols:
            higher_ed_col = higher_ed_cols[0]
            print(f"DEBUG: higher_ed_col {higher_ed_col}")
            # Assuming values like 'Да'/'Нет' or boolean
            higher_count = scoped_df[higher_ed_col].notna().sum()  # Simplistic, adjust if needed
            total_count = len(scoped_df)
            higher_share = higher_count / total_count * 100 if total_count > 0 else 0
            fig_higher = px.pie(values=[higher_count, total_count - higher_count], 
                               names=['Высшее образование', 'Нет высшего образования'],
                               title=f'Доля людей с высшим образованием: {higher_share:.1f}%')
            st.plotly_chart(fig_higher, use_container_width=True)
        else:
            st.info("Колонка с информацией о высшем образовании не найдена.")
    
    st.markdown("---")
    
    # Table with people without higher education
    st.markdown("### 📋 Список людей без высшего образования")
    if higher_ed_cols:
        higher_ed_col = higher_ed_cols[0]
        # Filter for those without higher education (assuming NaN or specific value means no)
        # We'll consider NaN as no higher education for simplicity
        without_higher = scoped_df[scoped_df[higher_ed_col].isna()]
        if not without_higher.empty:
            st.dataframe(without_higher, use_container_width=True)
        else:
            st.info("Все люди имеют высшее образование (или данные отсутствуют).")
    else:
        st.info("Не удалось определить колонку высшего образования для фильтрации.")
    
    st.markdown("---")

    st.markdown("### 📋 Все записи об образовании")
    st.dataframe(scoped_education_df, use_container_width=True)


if __name__ == "__main__":
    main()