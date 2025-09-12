import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sqrt, pi
from mpl_toolkits.mplot3d import Axes3D

def dimensional_recursion_explorer():
    st.header("Dimensional Recursion Explorer")
    st.markdown("Model dimensionality progression (1D string → 2D membrane → 3D torus) using recursive functions and SymPy projections scaled by φ.")

    # Calculate golden ratio
    phi = (1 + sqrt(5)) / 2
    phi_val = float(phi.evalf())

    # Input parameters
    recursion_depth = st.slider("Recursion Depth", 1, 8, 4)
    base_scale = st.slider("Base Scale", 0.1, 2.0, 1.0, step=0.1)
    projection_type = st.selectbox("Projection Type", ["Linear", "Golden Ratio", "Fibonacci"])

    # Recursive function for dimensional progression
    def dimensional_recursion(level, max_level, scale, center=(0, 0, 0)):
        if level > max_level:
            return []

        points = []

        if level == 1:  # 1D string
            # Create a 1D line segment
            length = scale * phi_val ** level
            x_vals = np.linspace(center[0] - length/2, center[0] + length/2, 50)
            y_vals = np.full_like(x_vals, center[1])
            z_vals = np.full_like(x_vals, center[2])
            points.extend(list(zip(x_vals, y_vals, z_vals)))

        elif level == 2:  # 2D membrane
            # Create a 2D circular membrane
            radius = scale * phi_val ** level
            theta = np.linspace(0, 2*np.pi, 100)
            x_vals = center[0] + radius * np.cos(theta)
            y_vals = center[1] + radius * np.sin(theta)
            z_vals = np.full_like(x_vals, center[2])
            points.extend(list(zip(x_vals, y_vals, z_vals)))

        else:  # 3D torus and higher
            # Create a 3D torus-like structure
            R = scale * phi_val ** level  # Major radius
            r = scale * phi_val ** (level-1)  # Minor radius

            u = np.linspace(0, 2*np.pi, 50)
            v = np.linspace(0, 2*np.pi, 50)
            U, V = np.meshgrid(u, v)

            x_vals = center[0] + (R + r * np.cos(V)) * np.cos(U)
            y_vals = center[1] + (R + r * np.cos(V)) * np.sin(U)
            z_vals = center[2] + r * np.sin(V)

            # Sample points from the surface
            for i in range(0, len(x_vals), 5):  # Sample every 5th point
                for j in range(0, len(x_vals[i]), 5):
                    points.append((x_vals[i,j], y_vals[i,j], z_vals[i,j]))

        # Recursive calls for sub-structures
        if level < max_level:
            # Create child structures at golden ratio positions
            if projection_type == "Golden Ratio":
                child_scale = scale * phi_val
                offsets = [(child_scale, 0, 0), (-child_scale, 0, 0),
                          (0, child_scale, 0), (0, -child_scale, 0),
                          (0, 0, child_scale), (0, 0, -child_scale)]
            elif projection_type == "Fibonacci":
                child_scale = scale * 1.618
                fib_angle = level * 137.5 * np.pi / 180  # Fibonacci angle
                offsets = [(child_scale * np.cos(fib_angle), child_scale * np.sin(fib_angle), 0),
                          (child_scale * np.cos(fib_angle + np.pi), child_scale * np.sin(fib_angle + np.pi), 0)]
            else:  # Linear
                child_scale = scale * 0.8
                offsets = [(child_scale, 0, 0), (-child_scale, 0, 0)]

            for offset in offsets[:min(level+1, len(offsets))]:  # Limit children based on level
                child_center = (center[0] + offset[0], center[1] + offset[1], center[2] + offset[2])
                child_points = dimensional_recursion(level + 1, max_level, child_scale * 0.7, child_center)
                points.extend(child_points)

        return points

    # Generate the recursive structure
    all_points = dimensional_recursion(1, recursion_depth, base_scale)

    # Separate points by dimension for visualization
    dim1_points = []
    dim2_points = []
    dim3_points = []

    current_level = 1
    points_per_level = []
    temp_points = []

    for point in all_points:
        temp_points.append(point)
        if len(temp_points) >= 50:  # Approximate points per level
            points_per_level.append(temp_points)
            temp_points = []

    if temp_points:
        points_per_level.append(temp_points)

    # Assign dimensions
    for i, level_points in enumerate(points_per_level):
        if i == 0:
            dim1_points = level_points
        elif i == 1:
            dim2_points = level_points
        else:
            dim3_points.extend(level_points)

    # Create visualization
    fig = plt.figure(figsize=(15, 10))

    # 3D plot
    ax1 = fig.add_subplot(221, projection='3d')

    if dim1_points:
        x1, y1, z1 = zip(*dim1_points)
        ax1.plot(x1, y1, z1, 'r-', linewidth=2, label='1D Strings', alpha=0.8)

    if dim2_points:
        x2, y2, z2 = zip(*dim2_points)
        ax2_scatter = ax1.scatter(x2, y2, z2, c='b', s=10, alpha=0.6, label='2D Membranes')

    if dim3_points:
        x3, y3, z3 = zip(*dim3_points)
        ax3_scatter = ax1.scatter(x3, y3, z3, c='g', s=5, alpha=0.4, label='3D Tori')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Recursive Dimensional Structure')
    ax1.legend()

    # Set equal aspect ratio
    max_range = max([ax1.get_xlim()[1]-ax1.get_xlim()[0],
                     ax1.get_ylim()[1]-ax1.get_ylim()[0],
                     ax1.get_zlim()[1]-ax1.get_zlim()[0]]) / 2.0
    mid_x = (ax1.get_xlim()[1]+ax1.get_xlim()[0]) * 0.5
    mid_y = (ax1.get_ylim()[1]+ax1.get_ylim()[0]) * 0.5
    mid_z = (ax1.get_zlim()[1]+ax1.get_zlim()[0]) * 0.5
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    # 2D projections
    ax2 = fig.add_subplot(222)
    if dim1_points:
        x1, y1, z1 = zip(*dim1_points)
        ax2.plot(x1, y1, 'r-', alpha=0.8)
    if dim2_points:
        x2, y2, z2 = zip(*dim2_points)
        ax2.scatter(x2, y2, c='b', s=10, alpha=0.6)
    if dim3_points:
        x3, y3, z3 = zip(*dim3_points)
        ax3_scatter = ax2.scatter(x3, y3, c='g', s=5, alpha=0.4)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection')
    ax2.grid(True)
    ax2.set_aspect('equal')

    # Scale analysis
    ax3 = fig.add_subplot(223)
    scales = [base_scale * (phi_val ** i) for i in range(recursion_depth)]
    ax3.plot(range(1, recursion_depth + 1), scales, 'o-', color='purple')
    ax3.set_xlabel('Recursion Level')
    ax3.set_ylabel('Scale Factor')
    ax3.set_title('Recursive Scaling')
    ax3.grid(True)

    # Volume analysis
    ax4 = fig.add_subplot(224)
    volumes = [scale ** level for level, scale in enumerate(scales, 1)]
    ax4.plot(range(1, recursion_depth + 1), volumes, 's-', color='orange')
    ax4.set_xlabel('Dimension Level')
    ax4.set_ylabel('Relative Volume')
    ax4.set_title('Volume Progression')
    ax4.grid(True)

    plt.tight_layout()
    st.pyplot(fig)

    # Statistics
    st.subheader("Recursion Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Recursion Depth", recursion_depth)
    with col2:
        st.metric("Total Points", len(all_points))
    with col3:
        st.metric("Golden Ratio φ", f"{phi_val:.6f}")
    with col4:
        st.metric("Projection Type", projection_type)

    # Dimensional analysis
    st.subheader("Dimensional Analysis")
    st.write(f"**1D Structures (Strings):** {len(dim1_points)} points")
    st.write(f"**2D Structures (Membranes):** {len(dim2_points)} points")
    st.write(f"**3D Structures (Tori):** {len(dim3_points)} points")

    # Scaling explanation
    st.markdown(f"""
    **Recursive Dimensional Progression:**
    - **Level 1 (1D):** Linear strings representing fundamental vibrations
    - **Level 2 (2D):** Circular membranes representing wave propagation
    - **Level 3+ (3D):** Toroidal structures representing field topologies
    - **Scaling:** Each level scales by φ = {phi_val:.6f} for optimal resonance
    - **Projection:** {projection_type} method used for sub-structure positioning

    This recursive model demonstrates how complex field structures emerge from
    simple dimensional progressions, with the golden ratio providing optimal scaling.
    """)
