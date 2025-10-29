import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(layout="centered")
st.title("Interactive Sliders with Addable Bar")

# --- Sliders for first three variables ---
var1 = st.slider("Variable 1", 1, 10, 5)
var2 = st.slider("Variable 2", 1, 10, 5)
var3 = st.slider("Variable 3", 1, 10, 5)

# --- Textbox for fourth variable ---
if 'var4' not in st.session_state:
    st.session_state.var4 = None

var4_input = st.text_input("Enter Variable 4 value", "")

if st.button("Add Variable 4"):
    st.session_state.var4 = float(var4_input)

variables = ['Variable 1', 'Variable 2', 'Variable 3']
values = [var1, var2, var3]

if st.session_state.var4 is not None:
    variables.append('Variable 4')
    values.append(st.session_state.var4)

data = pd.DataFrame({
    'Variable': variables,
    'Value': values
})

# --- Bar chart ---
bar_chart = alt.Chart(data).mark_bar().encode(
    x='Variable',
    y='Value',
    color='Variable'
).properties(
    width=400,
    height=300
)

st.altair_chart(bar_chart, use_container_width=True)
