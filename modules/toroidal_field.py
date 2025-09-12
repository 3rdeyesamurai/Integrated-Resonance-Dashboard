import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import sqrt

def toroidal_field_spiral_drawer():
    st.header("Toroidal Field Spiral Drawer")
    st.markdown("Use Matplotlib to draw golden spirals (φ-scaling) tracing toroidal fields, simulating applications like DNA helices or magnetic fields with vector field plots.")

    # Calculate golden ratio
    phi = (1 + sqrt(5)) / 2
    phi_val = float(phi.evalf())

    # Input parameters
    field_type = st.selectbox("Field Type", ["DNA Double Helix", "Magnetic Field", "Electromagnetic Wave", "Vortex Field"])
    spiral_turns = st.slider("Spiral Turns", 1, 8, 3)
    torus_major_radius = st.slider("Torus Major Radius", 1.0, 5.0, 3.0, step=0.1)
    torus_minor_radius = st.slider("Torus Minor Radius", 0.1, 1.5, 0.8, step=0.1)
    spiral_pitch = st.slider("Spiral Pitch", 0.1, 2.0, phi_val, step=0.1)
    field_strength = st.slider("Field Strength", 0.1, 3.0, 1.0, step=0.1)

    # Create toroidal coordinate system
    u = np.linspace(0, 2*np.pi, 100)  # Toroidal angle
    v = np.linspace(0, 2*np.pi, 100)  # Poloidal angle

    U, V = np.meshgrid(u, v)

    # Parametric equations for torus
    X_torus = (torus_major_radius + torus_minor_radius * np.cos(V)) * np.cos(U)
    Y_torus = (torus_major_radius + torus_minor_radius * np.cos(V)) * np.sin(U)
    Z_torus = torus_minor_radius * np.sin(V)

    # Generate golden spiral on torus
    t_spiral = np.linspace(0, spiral_turns * 2 * np.pi, 500)

    # Golden spiral scaling
    r_spiral = torus_minor_radius * phi_val ** (t_spiral / (2 * np.pi))

    # Map spiral to torus surface
    u_spiral = t_spiral  # Toroidal angle follows spiral parameter
    v_spiral = np.arctan2(spiral_pitch * t_spiral, r_spiral)  # Poloidal angle

    # Parametric equations for spiral on torus
    X_spiral = (torus_major_radius + r_spiral * np.cos(v_spiral)) * np.cos(u_spiral)
    Y_spiral = (torus_major_radius + r_spiral * np.cos(v_spiral)) * np.sin(u_spiral)
    Z_spiral = r_spiral * np.sin(v_spiral)

    # Generate field vectors based on type
    if field_type == "DNA Double Helix":
        # Two intertwined spirals (like DNA)
        t_dna = np.linspace(0, spiral_turns * 2 * np.pi, 300)

        # Primary spiral
        r_dna1 = torus_minor_radius * phi_val ** (t_dna / (2 * np.pi))
        u_dna1 = t_dna
        v_dna1 = np.arctan2(spiral_pitch * t_dna, r_dna1) + np.pi/6  # Offset for double helix

        X_dna1 = (torus_major_radius + r_dna1 * np.cos(v_dna1)) * np.cos(u_dna1)
        Y_dna1 = (torus_major_radius + r_dna1 * np.cos(v_dna1)) * np.sin(u_dna1)
        Z_dna1 = r_dna1 * np.sin(v_dna1)

        # Secondary spiral (complementary)
        r_dna2 = torus_minor_radius * phi_val ** (t_dna / (2 * np.pi))
        u_dna2 = t_dna
        v_dna2 = np.arctan2(spiral_pitch * t_dna, r_dna2) - np.pi/6  # Offset in opposite direction

        X_dna2 = (torus_major_radius + r_dna2 * np.cos(v_dna2)) * np.cos(u_dna2)
        Y_dna2 = (torus_major_radius + r_dna2 * np.cos(v_dna2)) * np.sin(u_dna2)
        Z_dna2 = r_dna2 * np.sin(v_dna2)

        # Field vectors (tangential to spirals)
        field_vectors = None

    elif field_type == "Magnetic Field":
        # Toroidal magnetic field (like tokamak)
        # Field vectors point in toroidal direction
        field_U = np.ones_like(U)  # Toroidal component
        field_V = np.zeros_like(V)  # No poloidal component

        # Convert to Cartesian coordinates
        field_X = -field_strength * np.sin(U)  # Toroidal direction
        field_Y = field_strength * np.cos(U)
        field_Z = np.zeros_like(U)

        field_vectors = (X_torus[::5, ::5], Y_torus[::5, ::5], Z_torus[::5, ::5],
                        field_X[::5, ::5], field_Y[::5, ::5], field_Z[::5, ::5])

    elif field_type == "Electromagnetic Wave":
        # Circularly polarized EM wave
        # Field rotates in poloidal direction
        field_U = np.zeros_like(U)
        field_V = field_strength * np.ones_like(V)  # Poloidal direction

        # Convert to Cartesian
        field_X = -field_strength * np.sin(U) * np.sin(V)
        field_Y = field_strength * np.cos(U) * np.sin(V)
        field_Z = field_strength * np.cos(V)

        field_vectors = (X_torus[::5, ::5], Y_torus[::5, ::5], Z_torus[::5, ::5],
                        field_X[::5, ::5], field_Y[::5, ::5], field_Z[::5, ::5])

    else:  # Vortex Field
        # Vortex field with golden ratio scaling
        field_U = field_strength * np.cos(phi_val * V)
        field_V = field_strength * np.sin(phi_val * U)

        # Convert to Cartesian
        field_X = -field_strength * np.sin(U) * np.cos(phi_val * V) - field_strength * np.cos(U) * np.sin(phi_val * U) * np.sin(V)
        field_Y = field_strength * np.cos(U) * np.cos(phi_val * V) - field_strength * np.sin(U) * np.sin(phi_val * U) * np.sin(V)
        field_Z = field_strength * np.cos(V) * np.sin(phi_val * U)

        field_vectors = (X_torus[::5, ::5], Y_torus[::5, ::5], Z_torus[::5, ::5],
                        field_X[::5, ::5], field_Y[::5, ::5], field_Z[::5, ::5])

    # Create visualization
    fig = plt.figure(figsize=(16, 12))

    # 3D field visualization
    ax1 = fig.add_subplot(221, projection='3d')

    # Plot torus surface (semi-transparent)
    ax1.plot_surface(X_torus, Y_torus, Z_torus, alpha=0.1, color='gray')

    # Plot golden spiral
    if field_type == "DNA Double Helix":
        ax1.plot(X_dna1, Y_dna1, Z_dna1, 'b-', linewidth=3, label='DNA Strand 1')
        ax1.plot(X_dna2, Y_dna2, Z_dna2, 'r-', linewidth=3, label='DNA Strand 2')
        # Add connecting bars (simplified DNA structure)
        for i in range(0, len(X_dna1), 20):
            ax1.plot([X_dna1[i], X_dna2[i]], [Y_dna1[i], Y_dna2[i]], [Z_dna1[i], Z_dna2[i]],
                    'g-', alpha=0.6)
    else:
        ax1.plot(X_spiral, Y_spiral, Z_spiral, 'gold', linewidth=4, label='Golden Spiral')

    # Add field vectors if available
    if field_vectors:
        ax1.quiver(field_vectors[0], field_vectors[1], field_vectors[2],
                  field_vectors[3], field_vectors[4], field_vectors[5],
                  length=0.3, normalize=True, color='purple', alpha=0.6, label='Field Vectors')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title(f'{field_type} on Toroidal Surface')
    ax1.legend()
    ax1.set_xlim([-torus_major_radius-1, torus_major_radius+1])
    ax1.set_ylim([-torus_major_radius-1, torus_major_radius+1])
    ax1.set_zlim([-torus_minor_radius-1, torus_minor_radius+1])

    # 2D projection
    ax2 = fig.add_subplot(222)
    ax2.plot(X_spiral, Y_spiral, 'gold', linewidth=3, label='Golden Spiral')
    ax2.plot(X_torus[:, 0], Y_torus[:, 0], 'gray', alpha=0.3, label='Torus Outline')
    ax2.plot(X_torus[:, -1], Y_torus[:, -1], 'gray', alpha=0.3)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Spiral parameter analysis
    ax3 = fig.add_subplot(223)
    ax3.plot(t_spiral, r_spiral, 'gold', linewidth=2, label='Spiral Radius')
    ax3.axhline(y=torus_minor_radius, color='gray', linestyle='--', alpha=0.7, label='Torus Minor Radius')
    ax3.set_xlabel('Spiral Parameter t')
    ax3.set_ylabel('Radius r')
    ax3.set_title('Golden Spiral Scaling')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Field strength analysis
    ax4 = fig.add_subplot(224)

    if field_type == "DNA Double Helix":
        # Plot distance between DNA strands
        distance = np.sqrt((X_dna1 - X_dna2)**2 + (Y_dna1 - Y_dna2)**2 + (Z_dna1 - Z_dna2)**2)
        ax4.plot(t_dna[:len(distance)], distance, 'green', linewidth=2, label='Strand Distance')
        ax4.set_ylabel('Distance')
        ax4.set_title('DNA Strand Separation')
    else:
        # Plot field magnitude variation
        field_magnitude = np.sqrt(field_X**2 + field_Y**2 + field_Z**2)
        center_idx = len(field_magnitude) // 2
        ax4.plot(U[center_idx, :], field_magnitude[center_idx, :], 'purple', linewidth=2, label='Field Magnitude')
        ax4.set_ylabel('Field Strength')
        ax4.set_title('Field Magnitude Variation')

    ax4.set_xlabel('Toroidal Angle θ')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Analysis metrics
    st.subheader("Toroidal Field Analysis")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Field Type", field_type)
    with col2:
        st.metric("Spiral Turns", spiral_turns)
    with col3:
        st.metric("Major Radius", f"{torus_major_radius:.1f}")
    with col4:
        st.metric("Golden Ratio φ", f"{phi_val:.6f}")

    # Detailed analysis
    st.subheader("Spiral Characteristics")

    # Calculate spiral properties
    spiral_length = np.sum(np.sqrt(np.diff(X_spiral)**2 + np.diff(Y_spiral)**2 + np.diff(Z_spiral)**2))
    st.write(f"**Spiral Length:** {spiral_length:.2f}")
    st.write(f"**Spiral Pitch:** {spiral_pitch:.2f}")
    st.write(f"**Growth Rate:** φ = {phi_val:.6f} per turn")

    # Field analysis
    st.subheader("Field Properties")

    if field_type == "DNA Double Helix":
        avg_distance = np.mean(distance)
        st.write(f"**Average Strand Distance:** {avg_distance:.3f}")
        st.write("**DNA Structure:** Double helical configuration with golden ratio scaling")
    elif field_vectors:
        avg_field_strength = np.mean(np.sqrt(field_X**2 + field_Y**2 + field_Z**2))
        st.write(f"**Average Field Strength:** {avg_field_strength:.3f}")
        st.write(f"**Field Configuration:** {field_type} with toroidal topology")

    # Torus geometry
    st.subheader("Toroidal Geometry")

    torus_surface_area = 4 * np.pi**2 * torus_major_radius * torus_minor_radius
    torus_volume = 2 * np.pi**2 * torus_major_radius * torus_minor_radius**2

    st.write(f"**Surface Area:** {torus_surface_area:.2f}")
    st.write(f"**Volume:** {torus_volume:.2f}")
    st.write(f"**Aspect Ratio:** {torus_major_radius/torus_minor_radius:.2f}")

    st.markdown(f"""
    **Toroidal Field Spiral Theory:**
    - **Golden Spiral Scaling**: φ = {phi_val:.6f} provides optimal growth rate for natural structures
    - **Toroidal Topology**: Fundamental geometry for field confinement and energy flow
    - **Spiral Parameterization**: Maps logarithmic spiral to toroidal surface
    - **Field Vector Fields**: Represent different physical field configurations

    **Physical Interpretations:**
    - **DNA Double Helix**: Genetic information storage with golden ratio structural scaling
    - **Magnetic Fields**: Plasma confinement in tokamak fusion devices
    - **Electromagnetic Waves**: Polarized light propagation in complex media
    - **Vortex Fields**: Fluid dynamics and quantum vortex structures

    **Mathematical Properties:**
    - **Self-Similarity**: Golden ratio scaling creates fractal-like patterns
    - **Topological Invariants**: Conserved quantities in field configurations
    - **Symmetry Breaking**: Spiral structures break rotational symmetry
    - **Energy Optimization**: Golden ratio provides minimal energy configurations

    **Applications:**
    - **Molecular Biology**: DNA structure and protein folding patterns
    - **Fusion Physics**: Magnetic field design for plasma confinement
    - **Optics**: Polarization control in laser systems
    - **Fluid Dynamics**: Vortex formation in turbulent flows
    """)

    # Interactive exploration
    st.subheader("Interactive Exploration")
    st.markdown("Adjust the torus parameters and field type to explore how golden spirals manifest different physical phenomena on toroidal surfaces.")
