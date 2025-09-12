import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import sqrt

def triadic_phenomena_mapper():
    st.header("Triadic Phenomena Mapper")
    st.markdown("Visualize triadic structures (e.g., three spatial dimensions, force unifications) via trefoil lobes, using Matplotlib to overlay Fibonacci knots on 3D grids.")

    # Calculate golden ratio
    phi = (1 + sqrt(5)) / 2
    phi_val = float(phi.evalf())

    # Input parameters
    structure_type = st.selectbox("Triadic Structure", ["Spatial Dimensions", "Force Unifications", "Quantum States", "Consciousness Fields"])
    scale_factor = st.slider("Scale Factor", 0.5, 3.0, 1.5, step=0.1)
    rotation_angle = st.slider("Rotation Angle", 0, 360, 120, step=15)
    fibonacci_order = st.slider("Fibonacci Order", 3, 8, 5)

    # Generate Fibonacci sequence for knot parameters
    fib_seq = [1, 1]
    for i in range(2, fibonacci_order + 2):
        fib_seq.append(fib_seq[i-1] + fib_seq[i-2])

    # Create 3D grid
    grid_size = 20
    x = np.linspace(-scale_factor, scale_factor, grid_size)
    y = np.linspace(-scale_factor, scale_factor, grid_size)
    z = np.linspace(-scale_factor, scale_factor, grid_size)
    X, Y, Z = np.meshgrid(x, y, z)

    # Generate triadic structures
    if structure_type == "Spatial Dimensions":
        # Three orthogonal planes representing spatial dimensions
        structure_1 = np.abs(X)  # X-dimension
        structure_2 = np.abs(Y)  # Y-dimension
        structure_3 = np.abs(Z)  # Z-dimension

    elif structure_type == "Force Unifications":
        # Electromagnetic, weak, and strong forces
        r = np.sqrt(X**2 + Y**2 + Z**2)
        structure_1 = np.exp(-r) * np.cos(3 * np.arctan2(Y, X))  # Electromagnetic
        structure_2 = np.exp(-r/phi_val) * np.sin(4 * np.arctan2(Z, X))  # Weak
        structure_3 = np.exp(-r/phi_val**2) * np.cos(5 * np.arctan2(Y, Z))  # Strong

    elif structure_type == "Quantum States":
        # Three quantum states (ground, excited, ionized)
        r = np.sqrt(X**2 + Y**2 + Z**2)
        theta = np.arctan2(Y, X)
        phi_angle = np.arctan2(np.sqrt(X**2 + Y**2), Z)

        structure_1 = np.exp(-2*r) * (1 + np.cos(theta))  # Ground state
        structure_2 = np.exp(-r/phi_val) * np.sin(2*theta) * np.cos(phi_angle)  # Excited state
        structure_3 = np.exp(-r/(2*phi_val)) * np.sin(3*theta) * np.sin(2*phi_angle)  # Ionized state

    else:  # Consciousness Fields
        # Three consciousness aspects (perception, cognition, emotion)
        r = np.sqrt(X**2 + Y**2 + Z**2)
        theta = np.arctan2(Y, X)
        phi_angle = np.arctan2(np.sqrt(X**2 + Y**2), Z)

        structure_1 = np.sin(2*theta) * np.exp(-r)  # Perception
        structure_2 = np.cos(3*phi_angle) * np.exp(-r/phi_val)  # Cognition
        structure_3 = np.sin(4*theta) * np.cos(2*phi_angle) * np.exp(-r/phi_val**2)  # Emotion

    # Generate Fibonacci knot overlay
    t = np.linspace(0, 2*np.pi, 200)
    p = fib_seq[-2]  # Second to last Fibonacci number
    q = fib_seq[-1]  # Last Fibonacci number

    # Apply rotation
    rotation_rad = np.radians(rotation_angle)
    R_torus = scale_factor * 0.8
    r_torus = scale_factor * 0.3

    # Parametric equations for rotated torus knot
    x_knot = (R_torus + r_torus * np.cos(q * t)) * np.cos(p * t + rotation_rad)
    y_knot = (R_torus + r_torus * np.cos(q * t)) * np.sin(p * t + rotation_rad)
    z_knot = r_torus * np.sin(q * t)

    # Create visualization
    fig = plt.figure(figsize=(16, 12))

    # 3D visualization
    ax1 = fig.add_subplot(221, projection='3d')

    # Plot triadic structures as isosurfaces (using scatter for simplicity)
    threshold = 0.3
    mask1 = structure_1 > threshold
    mask2 = structure_2 > threshold
    mask3 = structure_3 > threshold

    ax1.scatter(X[mask1], Y[mask1], Z[mask1], c='red', alpha=0.1, s=1, label=f'{structure_type} Component 1')
    ax1.scatter(X[mask2], Y[mask2], Z[mask2], c='blue', alpha=0.1, s=1, label=f'{structure_type} Component 2')
    ax1.scatter(X[mask3], Y[mask3], Z[mask3], c='green', alpha=0.1, s=1, label=f'{structure_type} Component 3')

    # Overlay Fibonacci knot
    ax1.plot(x_knot, y_knot, z_knot, 'black', linewidth=3, alpha=0.8, label=f'({p},{q}) Fibonacci Knot')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'Triadic {structure_type} with Fibonacci Knot')
    ax1.legend()
    ax1.set_xlim([-scale_factor, scale_factor])
    ax1.set_ylim([-scale_factor, scale_factor])
    ax1.set_zlim([-scale_factor, scale_factor])

    # 2D projections
    ax2 = fig.add_subplot(222)
    ax2.contourf(X[:, :, grid_size//2], Y[:, :, grid_size//2], structure_1[:, :, grid_size//2], levels=20, cmap='Reds', alpha=0.7)
    ax2.contourf(X[:, :, grid_size//2], Y[:, :, grid_size//2], structure_2[:, :, grid_size//2], levels=20, cmap='Blues', alpha=0.7)
    ax2.contourf(X[:, :, grid_size//2], Y[:, :, grid_size//2], structure_3[:, :, grid_size//2], levels=20, cmap='Greens', alpha=0.7)
    ax2.plot(x_knot, y_knot, 'black', linewidth=2, alpha=0.8)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Cross-section analysis
    ax3 = fig.add_subplot(223)
    center_idx = grid_size // 2
    x_line = X[center_idx, center_idx, :]
    struct1_line = structure_1[center_idx, center_idx, :]
    struct2_line = structure_2[center_idx, center_idx, :]
    struct3_line = structure_3[center_idx, center_idx, :]

    ax3.plot(x_line, struct1_line, 'r-', linewidth=2, label='Component 1')
    ax3.plot(x_line, struct2_line, 'b-', linewidth=2, label='Component 2')
    ax3.plot(x_line, struct3_line, 'g-', linewidth=2, label='Component 3')
    ax3.set_xlabel('Z Position')
    ax3.set_ylabel('Field Strength')
    ax3.set_title('Z-Axis Cross-Section')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Trefoil lobe analysis
    ax4 = fig.add_subplot(224)

    # Calculate lobe volumes (simplified)
    lobe1_volume = np.sum(structure_1 > threshold)
    lobe2_volume = np.sum(structure_2 > threshold)
    lobe3_volume = np.sum(structure_3 > threshold)

    lobes = ['Component 1', 'Component 2', 'Component 3']
    volumes = [lobe1_volume, lobe2_volume, lobe3_volume]
    colors = ['red', 'blue', 'green']

    bars = ax4.bar(lobes, volumes, color=colors, alpha=0.7)
    ax4.set_ylabel('Relative Volume')
    ax4.set_title('Trefoil Lobe Volumes')
    ax4.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, volume in zip(bars, volumes):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{volume}', ha='center', va='bottom')

    plt.tight_layout()
    st.pyplot(fig)

    # Analysis metrics
    st.subheader("Triadic Structure Analysis")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Structure Type", structure_type)
    with col2:
        st.metric("Scale Factor", f"{scale_factor:.1f}")
    with col3:
        st.metric("Fibonacci Order", fibonacci_order)
    with col4:
        st.metric("Golden Ratio φ", f"{phi_val:.6f}")

    # Detailed analysis
    st.subheader("Component Analysis")

    # Calculate interaction strengths
    interaction_12 = np.corrcoef(structure_1.flatten(), structure_2.flatten())[0, 1]
    interaction_13 = np.corrcoef(structure_1.flatten(), structure_3.flatten())[0, 1]
    interaction_23 = np.corrcoef(structure_2.flatten(), structure_3.flatten())[0, 1]

    st.write(f"**Component 1-2 Interaction:** {interaction_12:.3f}")
    st.write(f"**Component 1-3 Interaction:** {interaction_13:.3f}")
    st.write(f"**Component 2-3 Interaction:** {interaction_23:.3f}")

    # Symmetry analysis
    symmetry_score = (abs(np.sum(structure_1) - np.sum(structure_2)) +
                     abs(np.sum(structure_2) - np.sum(structure_3)) +
                     abs(np.sum(structure_1) - np.sum(structure_3))) / 3

    st.write(f"**Triadic Symmetry Score:** {symmetry_score:.3f}")

    # Fibonacci knot properties
    knot_length = np.sum(np.sqrt(np.diff(x_knot)**2 + np.diff(y_knot)**2 + np.diff(z_knot)**2))
    st.write(f"**Knot Length:** {knot_length:.2f}")
    st.write(f"**Knot Type:** ({p},{q}) - {'Trefoil' if p==2 and q==3 else 'Complex'}")

    st.markdown(f"""
    **Triadic Structure Theory:**
    - **{structure_type}**: Three interconnected components forming a unified system
    - **Trefoil Topology**: Three-lobed structure representing fundamental triadic relationships
    - **Fibonacci Overlay**: ({p},{q}) knot providing structural coherence
    - **Golden Ratio Scaling**: φ = {phi_val:.6f} ensures optimal component interactions

    **Physical Interpretations:**
    - **Spatial Dimensions**: X, Y, Z coordinates in 3D space
    - **Force Unifications**: Electromagnetic, weak, and strong nuclear forces
    - **Quantum States**: Ground, excited, and continuum states
    - **Consciousness Fields**: Perception, cognition, and emotional processing

    **Mathematical Properties:**
    - **Symmetry**: Balanced distribution of energy across three components
    - **Interactions**: Correlated field strengths between components
    - **Topology**: Non-trivial connectivity through trefoil geometry
    - **Scaling**: Self-similar patterns at different size scales
    """)

    # Interactive exploration
    st.subheader("Interactive Exploration")
    st.markdown("Adjust the parameters above to explore how different triadic structures emerge and interact with Fibonacci topology.")
