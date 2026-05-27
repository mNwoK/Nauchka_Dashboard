import streamlit as st


ROLE_DEPARTMENT_DIRECTOR = "department_director"
ROLE_SCHOOL_DIRECTOR = "school_director"

ROLE_LABELS = {
    ROLE_DEPARTMENT_DIRECTOR: "Директор департамента образования",
    ROLE_SCHOOL_DIRECTOR: "Директор школы",
}


def get_role() -> str | None:
    return st.session_state.get("user_role")


def get_role_label() -> str:
    role = get_role()
    return ROLE_LABELS.get(role, "Роль не выбрана")


def get_school_id() -> int | None:
    return st.session_state.get("school_id")


def is_department_director() -> bool:
    return get_role() == ROLE_DEPARTMENT_DIRECTOR


def is_school_director() -> bool:
    return get_role() == ROLE_SCHOOL_DIRECTOR


def reset_role() -> None:
    st.session_state.pop("user_role", None)
    st.session_state.pop("school_id", None)


def require_role() -> bool:
    if get_role() is not None:
        return True

    st.warning("Сначала выберите роль на главной странице.")
    if st.button("Перейти к выбору роли", use_container_width=True):
        st.switch_page("main.py")
    return False
