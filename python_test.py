import plotly.graph_objects as go
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State

# Load elevation data
z_data = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/api_docs/mt_bruno_elevation.csv')

# Create 3D surface
surface = go.Surface(z=z_data.values, colorscale='Viridis')
fig = go.Figure(data=[surface])
fig.update_layout(
    title="3D Seabed Terrain Explorer",
    scene=dict(
        xaxis_title="Distance East (m)",
        yaxis_title="Distance North (m)",
        zaxis_title="Depth (m)"
    ),
    template='plotly_dark',
    margin=dict(l=0, r=0, b=0, t=50)
)

# Initialize Dash app
app = Dash(__name__)
app.title = "3D Terrain Path Explorer"

app.layout = html.Div(
    style={
        "backgroundColor": "#111",
        "color": "white",
        "fontFamily": "Arial, sans-serif",
        "textAlign": "center",
        "padding": "10px"
    },
    children=[
        html.H2("3D Terrain Path Explorer", style={"marginBottom": "10px"}),
        
        dcc.Graph(
            id='terrain',
            figure=fig,
            style={
                'height': '80vh',
                'width': '100%',
                'border': '2px solid #333',
                'borderRadius': '10px'
            }
        ),
        
        html.Div(
            id='output',
            style={
                'fontSize': '20px',
                'marginTop': '20px',
                'color': '#00FFFF',  # bright cyan for visibility
                'fontWeight': 'bold'
            },
            children="Click on the surface to begin drawing a path."
        )
    ]
)

@app.callback(
    Output('terrain', 'figure'),
    Output('output', 'children'),
    Input('terrain', 'clickData'),
    State('terrain', 'figure'),
    prevent_initial_call=True
)
def display_click_data(clickData, current_fig):
    fig = go.Figure(current_fig)

    if not clickData:
        return fig, "Click on the surface to begin drawing a path."

    # Extract clicked coordinates
    x = clickData['points'][0]['x']
    y = clickData['points'][0]['y']
    z = clickData['points'][0]['z']

    # Add or update path
    if len(fig.data) > 1:
        xs, ys, zs = list(fig.data[1].x) + [x], list(fig.data[1].y) + [y], list(fig.data[1].z) + [z]
        fig.data[1].x, fig.data[1].y, fig.data[1].z = xs, ys, zs
    else:
        fig.add_trace(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='lines+markers',
            marker=dict(size=5, color='red'),
            line=dict(color='red', width=4),
            name='Path'
        ))

    # Compute average height
    path_trace = fig.data[1]
    avg_z = sum(path_trace.z) / len(path_trace.z)

    # Print to console (for debug)
    print(f"Clicked ({x:.2f}, {y:.2f}, {z:.2f}) | Avg Height = {avg_z:.2f}")

    return fig, f"Average Height (Depth): {avg_z:.2f} m"

if __name__ == '__main__':
    app.run(debug=True)
