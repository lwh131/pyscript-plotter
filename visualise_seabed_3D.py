# # This script has been reworked to use a multi-layered Perlin noise method.
# # It now requires the 'perlin-noise' library.
# # You can install it by running: pip install perlin-noise

# import numpy as np
# import plotly.graph_objects as go
# from perlin_noise import PerlinNoise


# class SeabedGenerator:
#     """
#     Interactive 3D seabed generator using multiple layers of Perlin noise
#     to create complex and realistic underwater terrain.
#     The scale is in meters, with each grid point representing one meter.
#     """

#     def __init__(self, width=100, height=100, base_depth=50, seed=None):
#         """
#         Initialize the seabed generator.

#         Args:
#             width (int): Width of the seabed grid in meters (1 point per meter).
#             height (int): Height of the seabed grid in meters (1 point per meter).
#             base_depth (float): The average depth of the seabed in meters.
#             seed (int): Random seed for reproducibility.
#         """
#         self.width = width
#         self.height = height
#         self.base_depth = base_depth
#         # Use provided seed or generate a random one
#         self.seed = seed if seed is not None else np.random.randint(0, 10000)

#         # Create coordinate system (meters)
#         # Each point in the grid corresponds to one meter.
#         self.x_m = np.linspace(0, width, width)
#         self.y_m = np.linspace(0, height, height)
#         self.X, self.Y = np.meshgrid(self.x_m, self.y_m)

#         # Generate seabed terrain
#         self.depth_map = self._generate_seabed()

#     def _generate_seabed(self):
#         """
#         Generates a 3D seabed using multiple layers of Perlin noise.
#         This is a 2D adaptation of the provided 1D profile generator.
#         """
#         # Define the different layers of noise, from large features to fine details
#         # The amplitude controls the vertical variation in meters.
#         amplitude = 50

#         noise_layers = {
#             'layer1': {'octaves': 1, 'amplitude': amplitude, 'seed_offset': 0},
#             'layer2': {'octaves': 2, 'amplitude': amplitude/2, 'seed_offset': 1},
#             'layer3': {'octaves': 3, 'amplitude': amplitude/3, 'seed_offset': 2},
#             'layer4': {'octaves': 5, 'amplitude': amplitude/5, 'seed_offset': 3},
#             'layer5': {'octaves': 8, 'amplitude': amplitude/8, 'seed_offset': 4},
#             'layer6': {'octaves': 12, 'amplitude': amplitude/12, 'seed_offset': 5},
#             'layer7': {'octaves': 18, 'amplitude': amplitude/18, 'seed_offset': 6},
#             'layer8': {'octaves': 30, 'amplitude': amplitude/30, 'seed_offset': 7},
#             'layer9': {'octaves': 40, 'amplitude': amplitude/40, 'seed_offset': 8},
#             'layer10': {'octaves': 80, 'amplitude': amplitude/80, 'seed_offset': 9},
#             'layer11': {'octaves': 150, 'amplitude': amplitude/150, 'seed_offset': 10},
#             'layer12': {'octaves': 300, 'amplitude': amplitude/300, 'seed_offset': 11}
#         }

#         # Initialize a 2D grid for the final heightmap
#         world = np.zeros((self.height, self.width))

#         # Generate and combine all noise layers
#         for layer_name, params in noise_layers.items():
#             # Each layer gets its own noise generator with a unique seed
#             noise = PerlinNoise(octaves=params['octaves'], seed=self.seed + params['seed_offset'])

#             # Create a 2D noise map for the current layer
#             layer_map = np.array([
#                 [noise([i / self.height, j / self.width]) for j in range(self.width)]
#                 for i in range(self.height)
#             ])

#             # Add the scaled layer to the main world map
#             world += layer_map * params['amplitude']

#         # Center the terrain variation around zero
#         # world = world - world.mean()

#         # Set the final depth map. Depth is negative, so we subtract the base_depth.
#         # The noise `world` adds variation (positive makes it shallower, negative deeper).
#         final_depth_map = -self.base_depth + world

#         return final_depth_map


# def create_seabed_viewer(seabed_gen):
#     """
#     Create a highly interactive 3D seabed visualization with Plotly.
#     """
#     X, Y, Z = seabed_gen.X, seabed_gen.Y, seabed_gen.depth_map
#     surface = go.Surface(
#         x=X, y=Y, z=Z, colorscale='deep', showscale=True,
#         colorbar=dict(title="Depth (m)"),
#         lighting=dict(ambient=0.4, diffuse=0.8, specular=0.1, roughness=0.8),
#         lightposition=dict(x=1000, y=1000, z=2000), # Adjusted light position for meter scale
#         hovertemplate="<b>Position</b><br>" +
#                       "X: %{x:.0f} m<br>" +
#                       "Y: %{y:.0f} m<br>" +
#                       "Depth: %{z:.1f} m<br>" +
#                       "<extra></extra>"
#     )
#     fig = go.Figure(data=[surface])
#     fig.update_layout(
#         title=dict(text="3D Seabed Terrain Explorer", x=0.5, font=dict(size=18, color='white')),
#         scene=dict(
#             xaxis=dict(title="Distance East (m)", showgrid=True, gridcolor='lightblue'),
#             yaxis=dict(title="Distance North (m)", showgrid=True, gridcolor='lightblue'),
#             zaxis=dict(title="Depth (m)", showgrid=True, gridcolor='lightblue', autorange='reversed'),
#             camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
#             aspectratio=dict(x=1, y=1, z=0.4)
#         ),
#         template='plotly_dark'
#     )
#     fig.add_annotation(
#         text="🖱️ Drag to rotate • 🔍 Scroll to zoom • 📱 Double-click to reset view",
#         xref="paper", yref="paper", x=0.5, y=0.02, showarrow=False,
#         font=dict(size=12, color="darkblue"), bgcolor="rgba(255,255,255,0.8)",
#         bordercolor="darkblue", borderwidth=1
#     )
#     return fig


# def main():
#     """Main function to demonstrate the interactive seabed generator."""
#     print("🌊 Generating complex 3D seabed terrain using Perlin noise...")
#     seabed = SeabedGenerator(
#         width=600,          # The number of points (and meters) in the X direction
#         height=600,         # The number of points (and meters) in the Y direction
#         base_depth=150,     # The average depth of the seabed in meters
#         seed=123            # Use a specific seed for reproducible terrain
#     )

#     depths = seabed.depth_map
#     print(f"📊 Seabed Statistics:")
#     print(f"   Depth range: {depths.min():.1f} to {depths.max():.1f} m")
#     print(f"   Average depth: {depths.mean():.1f} m")
#     print(f"   Terrain area: {seabed.width} × {seabed.height} m")

#     print("\n🚀 Launching interactive 3D seabed explorer...")
#     fig_3d = create_seabed_viewer(seabed)
#     fig_3d.show()

#     print("\n✅ Interactive visualization launched!")


# if __name__ == "__main__":
#     main()