import streamlit as st

from app.auth import (
    ROLE_DEPARTMENT_DIRECTOR,
    ROLE_SCHOOL_DIRECTOR,
    get_role_label,
    get_school_id,
    reset_role,
)
from app.data import get_available_school_ids, load_data


def show_role_gate() -> None:
    st.title("Вход в аналитическую панель")
    st.write("Выберите роль, чтобы открыть доступный уровень данных.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Директор департамента образования", use_container_width=True):
            st.session_state.user_role = ROLE_DEPARTMENT_DIRECTOR
            st.session_state.pop("school_id", None)
            st.rerun()

    with col2:
        if st.button("Директор школы", use_container_width=True):
            st.session_state.user_role = ROLE_SCHOOL_DIRECTOR
            st.rerun()


def show_school_selector() -> None:
    st.title("Вход директора школы")
    st.write("Введите номер школы, чтобы открыть данные только по своей образовательной организации.")

    try:
        df = load_data()
    except ValueError as error:
        st.error(str(error))
        return

    available_school_ids = get_available_school_ids(df)
    positive_school_ids = [school_id for school_id in available_school_ids if school_id > 0]
    default_school_id = positive_school_ids[0] if positive_school_ids else 1

    with st.form("school_login_form"):
        school_id = st.number_input(
            "Номер школы",
            min_value=1,
            step=1,
            value=default_school_id,
        )
        submitted = st.form_submit_button("Войти", use_container_width=True)

    if submitted:
        school_id = int(school_id)
        if positive_school_ids and school_id not in positive_school_ids:
            st.error("Школа с таким номером не найдена в данных.")
            return

        st.session_state.school_id = school_id
        st.rerun()


def show() -> None:
    if "user_role" not in st.session_state:
        show_role_gate()
        return

    if st.session_state.user_role == ROLE_SCHOOL_DIRECTOR and get_school_id() is None:
        show_school_selector()
        return

    role_label = get_role_label()
    school_id = get_school_id()

    if school_id is None:
        st.title("Панель управления департамента образования")
        st.caption(role_label)
    else:
        st.title(f"Панель управления школы №{school_id}")
        st.caption(role_label)

    with st.sidebar:
        st.caption(f"Роль: {role_label}")
        if school_id is not None:
            st.caption(f"Школа №{school_id}")
        if st.button("Сменить роль", use_container_width=True):
            reset_role()
            st.switch_page("main.py")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 👥 Динамика кадров")
        if school_id is None:
            st.write("Сводная аналитика по педагогам всех школ")
        else:
            st.write("Кадровая аналитика по педагогам вашей школы")
        if st.button("Открыть отчет", key="open_report"):
            st.switch_page("pages/report.py")

        st.markdown("### 🎓 О школе")
        st.write("Общая информация о школе")
        st.button("В разработке...", disabled=True, key="school_stub")

    with col2:
        st.markdown("### 📚 Обучение")
        st.write("Повышение квалификации и переподготовка сотрудников")
        st.button("В разработке...", disabled=True, key="training_stub")

    with col3:
        st.markdown("### ⏱️ Нагрузка")
        st.write("Количество уроков и часов у сотрудника")
        st.button("В разработке...", disabled=True, key="workload_stub")
