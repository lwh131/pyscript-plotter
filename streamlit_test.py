import streamlit as st
import numpy as np
import plotly.graph_objects as go
import torch, joblib
from reconstruction_logic import analyze_profile

st.set_page_config(page_title="Cable Route Planner", layout="wide")

st.title("Cable Route Planner — Surrogate Model Demo")

# Sidebar controls
terrain_size = st.sidebar.slider("Terrain Size (m)", 50, 500, 200)
terrain_height = st.sidebar.slider("Terrain Height (m)", 5, 50, 20)
noise_scale = st.sidebar.slider("Noise Scale", 1, 10, 3)

# Generate terrain
x = np.linspace(0, terrain_size, 200)
y = np.linspace(0, terrain_size, 200)
X, Y = np.meshgrid(x, y)
Z = terrain_height * np.sin(X / (10 * noise_scale)) * np.cos(Y / (10 * noise_scale))

fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="Viridis", showscale=False)])
fig.update_layout(
    height=600,
    margin=dict(l=0, r=0, b=0, t=30),
    scene=dict(
        xaxis_title="East (m)",
        yaxis_title="North (m)",
        zaxis_title="Depth (m)"
    )
)

st.subheader("Terrain Preview")
st.plotly_chart(fig, use_container_width=True)

# Run surrogate analysis
if st.button("Calculate Cable Spans"):
    st.info("Running surrogate model...")
    # Placeholder: you'd call your real model here
    results = analyze_profile(Z)

    st.success("Cable profile calculated successfully!")
    st.metric("Max Span Height", f"{results['max_span']:.2f} m")
    st.metric("Total Cable Length", f"{results['cable_length']:.1f} m")
