import streamlit as st


st.header("Команда ЛИФТ")
x = st.slider("Выбирите значение")
st.write(x, "squared is", x * x)
