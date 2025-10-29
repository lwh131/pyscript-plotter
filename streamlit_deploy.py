import streamlit as st
import numpy as np
import plotly.graph_objects as go
from reconstruction import reconstruct
from simulation_utils import generate_seabed


PROFILE_LENGTH = 500
CABLE_VISUAL_DIAMETER = 8
ELEVATION_OFFSET_SCALE = 0.05

st.set_page_config(page_title="Cable Route Planner", layout="wide")

st.title("Cable Route Planner — Surrogate Model Demo")

# --- Session State Initialization ---
# Initialize session state variables if they don't exist. This is key to
# persisting the calculation result across reruns. Inspired by streamlit_toy.py.
if 'predicted_cable_z' not in st.session_state:
    st.session_state.predicted_cable_z = None
if 'last_run_params' not in st.session_state:
    st.session_state.last_run_params = None


# --- Sidebar Controls ---
st.sidebar.header("Terrain Controls")
terrain_size = st.sidebar.slider("Terrain Width and Depth (m)", 50, 2*PROFILE_LENGTH, PROFILE_LENGTH, 10)
terrain_height = st.sidebar.slider("Terrain Height (m)", 1.0, 5.0, 2.5, 0.5)
noise_scale = st.sidebar.slider("Noise Scale", 0.5, 3.0, 2.0, 0.25)
elevation_offset = terrain_height * ELEVATION_OFFSET_SCALE
# Group the current parameters to check if they have changed
current_params = (terrain_size, terrain_height, noise_scale)

# --- Invalidation Logic ---
# If the terrain parameters have changed, the old calculation is no longer valid.
# Set the predicted result to None to clear it from the plot.
if current_params != st.session_state.last_run_params:
    st.session_state.predicted_cable_z = None
    st.session_state.last_run_params = None # Clear params until new calculation

# --- Data Generation ---
# This part runs on every script rerun, so it always reflects the current slider values.
x = np.linspace(0, terrain_size, PROFILE_LENGTH)
y = np.linspace(0, terrain_size, PROFILE_LENGTH)
X, Y = np.meshgrid(x, y)
Z = terrain_height * (np.sin(X / (15 * noise_scale)) + np.cos(Y / (10 * noise_scale)))

# Define the cable route (a diagonal line across the terrain)
route_x = x
route_y = y
# Extract the elevation profile along this diagonal route for calculation
seabed_profile_1d = Z.diagonal()

# --- Dictionaries for model ---
cable_dict = {
    'EA': 9336351.79727941,
    'EI': 93363.5179727941,
    'submerged_weight': 71.14523286308311,
    'residual_lay_tension': 9033.41008549494
    }
seabed_dict = {
    'profile_length': PROFILE_LENGTH,
    'total_window_length': 128,       # Base 2
    'centre_window_length': 74,       # Must be less than total_window_length
    'overlap_length': 20,             # Define the overlap
    'k_foundation': 1e5
}


# --- Calculation Trigger ---
st.sidebar.header("Analysis")
if st.sidebar.button("Calculate Cable Spans"):
    with st.spinner("Running surrogate model..."):
        # FIX: Use the visualized seabed profile, not randomly generated data
        predicted_z = reconstruct(route_x, seabed_profile_1d, cable_dict, seabed_dict)
        print(predicted_z)
        # Store the result and the parameters used for this run in the session state
        st.session_state.predicted_cable_z = predicted_z
        st.session_state.last_run_params = current_params

    st.sidebar.success("Calculation complete!")
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Max Seabed Height", f"{seabed_profile_1d.max():.2f} m")
    col2.metric("Max Cable Height", f"{predicted_z.max():.2f} m")

    # Rerun to ensure the plot updates immediately with the new state
    st.rerun()


# --- Visualization ---
st.subheader("Interactive 3D Terrain")

# Create the base figure with the terrain surface
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale="deep", showscale=False)])

# Add the planned route line to the 3D plot
fig.add_trace(go.Scatter3d(
    x=route_x, y=route_y, z=seabed_profile_1d + elevation_offset, # Use dynamic offset
    mode='lines', line=dict(color='red', width=CABLE_VISUAL_DIAMETER/2, dash='dash'), name='Planned Route'
))

# Conditionally add the calculated cable profile to the 3D plot
# This trace will only be added if a valid calculation exists in the session state.
if st.session_state.predicted_cable_z is not None:
    fig.add_trace(go.Scatter3d(
        x=route_x, y=route_y, z=st.session_state.predicted_cable_z,
        mode='lines', line=dict(color='cyan', width=CABLE_VISUAL_DIAMETER), name='Calculated Cable'
    ))

# Update layout for better appearance
fig.update_layout(
    height=600, margin=dict(l=0, r=0, b=0, t=30),
    scene=dict(
        xaxis_title="East (m)",
        yaxis_title="North (m)",
        zaxis_title="Depth (m)",
        aspectmode='data' # Ensures proportions are realistic
        ),
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

# Display the figure in the Streamlit app
st.plotly_chart(fig, width='stretch')