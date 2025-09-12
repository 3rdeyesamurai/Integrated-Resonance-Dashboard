import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import symbols, sqrt, pi, cos, sin

def scalar_resonance_index_computer():
    st.header("Scalar Resonance Index Computer")
    st.markdown("Calculate the scalar resonance index R = 2 |B1 B2| / (|B1|² + |B2|²) for tonal flow modes and visualize resonance strength on a color-coded torus.")

    # Input parameters
    B1_magnitude = st.slider("B1 Magnitude", 0.1, 2.0, 1.0, step=0.1)
    B2_magnitude = st.slider("B2 Magnitude", 0.1, 2.0, 0.8, step=0.1)
    phase_difference = st.slider("Phase Difference (degrees)", 0, 360, 90, step=15)
    frequency_ratio = st.slider("Frequency Ratio", 1.0, 3.0, 2.0, step=0.1)
    torus_resolution = st.slider("Torus Resolution", 20, 100, 50, step=10)

    # Convert phase to radians
    phi_rad = phase_difference * np.pi / 180

    # Calculate scalar resonance index
    numerator = 2 * B1_magnitude * B2_magnitude
    denominator = B1_magnitude**2 + B2_magnitude**2
    R = numerator / denominator

    st.metric("Scalar Resonance Index R", f"{R:.4f}")

    # Create torus for visualization
    u = np.linspace(0, 2*np.pi, torus_resolution)
    v = np.linspace(0, 2*np.pi, torus_resolution)
    U, V = np.meshgrid(u, v)

    # Torus parameters
    R_major = 3.0  # Major radius
    r_minor = 1.0  # Minor radius

    # Parametric equations for torus
    X = (R_major + r_minor * np.cos(V)) * np.cos(U)
    Y = (R_major + r_minor * np.cos(V)) * np.sin(U)
    Z = r_minor * np.sin(V)

    # Calculate resonance field on torus surface
    # Use angular position for resonance calculation
    resonance_field = np.zeros_like(U)

    for i in range(len(u)):
        for j in range(len(v)):
            # Calculate local resonance based on position
            theta = U[i,j]
            phi = V[i,j]

            # Complex amplitudes with phase
            B1_complex = B1_magnitude * np.exp(1j * theta)
            B2_complex = B2_magnitude * np.exp(1j * (frequency_ratio * theta + phi_rad))

            # Local resonance index
            local_R = 2 * abs(B1_complex * B2_complex) / (abs(B1_complex)**2 + abs(B2_complex)**2)
            resonance_field[i,j] = local_R

    # Create visualization
    fig = plt.figure(figsize=(15, 10))

    # 3D torus with resonance coloring
    ax1 = fig.add_subplot(221, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(resonance_field),
                           rstride=1, cstride=1, alpha=0.8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Resonance Field on Torus')
    plt.colorbar(surf, ax=ax1, shrink=0.5, label='Resonance Index R')

    # Set equal aspect ratio
    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    mid_z = (Z.max()+Z.min()) * 0.5
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    # 2D projection of resonance field
    ax2 = fig.add_subplot(222)
    im = ax2.imshow(resonance_field, extent=[0, 2*np.pi, 0, 2*np.pi],
                   origin='lower', cmap='viridis', aspect='equal')
    ax2.set_xlabel('θ (azimuthal angle)')
    ax2.set_ylabel('φ (polar angle)')
    ax2.set_title('Resonance Field Projection')
    plt.colorbar(im, ax=ax2, label='Resonance Index R')

    # Resonance vs phase difference
    ax3 = fig.add_subplot(223)
    phase_range = np.linspace(0, 2*np.pi, 100)
    resonance_vs_phase = []

    for phi in phase_range:
        B1_comp = B1_magnitude * np.exp(1j * 0)
        B2_comp = B2_magnitude * np.exp(1j * phi)
        R_local = 2 * abs(B1_comp * B2_comp) / (abs(B1_comp)**2 + abs(B2_comp)**2)
        resonance_vs_phase.append(R_local)

    ax3.plot(phase_range * 180 / np.pi, resonance_vs_phase, 'b-', linewidth=2)
    ax3.axvline(x=phase_difference, color='r', linestyle='--', label=f'Current: {phase_difference}°')
    ax3.set_xlabel('Phase Difference (degrees)')
    ax3.set_ylabel('Resonance Index R')
    ax3.set_title('Resonance vs Phase Difference')
    ax3.grid(True)
    ax3.legend()

    # Frequency ratio analysis
    ax4 = fig.add_subplot(224)
    ratio_range = np.linspace(1.0, 3.0, 50)
    resonance_vs_ratio = []

    for ratio in ratio_range:
        B1_comp = B1_magnitude * np.exp(1j * 0)
        B2_comp = B2_magnitude * np.exp(1j * (ratio * np.pi/2 + phi_rad))
        R_local = 2 * abs(B1_comp * B2_comp) / (abs(B1_comp)**2 + abs(B2_comp)**2)
        resonance_vs_ratio.append(R_local)

    ax4.plot(ratio_range, resonance_vs_ratio, 'g-', linewidth=2)
    ax4.axvline(x=frequency_ratio, color='r', linestyle='--', label=f'Current: {frequency_ratio}')
    ax4.set_xlabel('Frequency Ratio')
    ax4.set_ylabel('Resonance Index R')
    ax4.set_title('Resonance vs Frequency Ratio')
    ax4.grid(True)
    ax4.legend()

    plt.tight_layout()
    st.pyplot(fig)

    # Detailed analysis
    st.subheader("Resonance Analysis")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("B1 Magnitude", f"{B1_magnitude:.2f}")
    with col2:
        st.metric("B2 Magnitude", f"{B2_magnitude:.2f}")
    with col3:
        st.metric("Phase Diff", f"{phase_difference}°")
    with col4:
        st.metric("Freq Ratio", f"{frequency_ratio:.1f}")

    # Resonance classification
    if R > 0.8:
        resonance_level = "High Resonance"
        resonance_color = "🟢"
    elif R > 0.5:
        resonance_level = "Medium Resonance"
        resonance_color = "🟡"
    else:
        resonance_level = "Low Resonance"
        resonance_color = "🔴"

    st.subheader(f"Resonance Classification: {resonance_color} {resonance_level}")

    # Field statistics
    st.subheader("Field Statistics")
    st.write(f"**Mean Resonance:** {np.mean(resonance_field):.4f}")
    st.write(f"**Max Resonance:** {np.max(resonance_field):.4f}")
    st.write(f"**Min Resonance:** {np.min(resonance_field):.4f}")
    st.write(f"**Resonance Variance:** {np.var(resonance_field):.6f}")

    # Theoretical explanation
    st.markdown(f"""
    **Scalar Resonance Index Theory:**
    - **R = 2|B₁B₂| / (|B₁|² + |B₂|²)** measures coupling strength between modes
    - **High R (>0.8):** Strong coherent resonance, optimal energy transfer
    - **Medium R (0.5-0.8):** Moderate coupling with some energy exchange
    - **Low R (<0.5):** Weak coupling, minimal resonance effects

    **Torus Visualization:**
    - Color-coded surface shows local resonance strength
    - Phase differences create standing wave patterns on the torus
    - Frequency ratios determine the spatial distribution of resonance zones

    **Applications:**
    - Quantum field theory resonance calculations
    - Musical instrument coupling analysis
    - Neural synchronization modeling
    - Electromagnetic field interactions
    """)
