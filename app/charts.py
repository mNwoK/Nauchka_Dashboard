import pandas as pd
import plotly.express as px
import streamlit as st


def render_age_distribution(df: pd.DataFrame) -> None:
    age_dist = df["Возраст"].value_counts().sort_index()
    st.bar_chart(age_dist)


def render_gender_distribution(df: pd.DataFrame) -> None:
    gender_counts = df["Пол"].value_counts()
    fig = px.pie(
        values=gender_counts.values,
        names=gender_counts.index,
        hole=0.5,
        title="Распределение сотрудников по полу",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_age_boxplot(df: pd.DataFrame) -> None:
    fig = px.box(
        df,
        x="Пол",
        y="Возраст",
        points="outliers",
        title="Анализ выбросов по возрасту",
    )
    st.plotly_chart(fig, use_container_width=True)

