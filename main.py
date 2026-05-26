import streamlit as st

from pages_content import home


st.set_page_config(page_title="Панель управления", layout="wide")


def main() -> None:
    home.show()


if __name__ == "__main__":
    main()

