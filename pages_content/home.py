import streamlit as st


def show() -> None:
    st.title("Панель управления")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 👥 Динамика кадров")
        st.write("Общие сведения и движение сотрудников")
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

