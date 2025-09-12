import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sympy import sqrt

def phyllotaxis_pattern_generator():
    st.header("Phyllotaxis Pattern Generator")
    st.markdown("Generate phyllotaxis patterns using Fibonacci growth scaled by the golden ratio φ.")

    # Calculate golden ratio
    phi = (1 + sqrt(5)) / 2
    phi_val = float(phi.evalf())

    # Input parameters
    n_seeds = st.slider("Number of Seeds/Points", 50, 500, 200, step=25)
    angle_offset = st.slider("Angle Offset (degrees)", 0, 360, 137, step=1)
    scale_factor = st.slider("Scale Factor", 0.1, 5.0, 1.0, step=0.1)
    pattern_type = st.selectbox("Pattern Type", ["Sunflower", "Pine Cone", "Romanesco Broccoli"])

    # Phyllotaxis parameters based on pattern type
    if pattern_type == "Sunflower":
        divergence_angle = angle_offset * np.pi / 180  # Convert to radians
        c = scale_factor
    elif pattern_type == "Pine Cone":
        divergence_angle = (180 - angle_offset) * np.pi / 180
        c = scale_factor * phi_val
    else:  # Romanesco Broccoli
        divergence_angle = angle_offset * np.pi / 180
        c = scale_factor * phi_val ** 2

    # Generate phyllotaxis pattern
    angles = []
    radii = []

    for i in range(n_seeds):
        angle = i * divergence_angle
        radius = c * np.sqrt(i)

        angles.append(angle)
        radii.append(radius)

    # Convert to Cartesian coordinates
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Main phyllotaxis pattern
    scatter = ax1.scatter(x, y, c=np.arange(n_seeds), cmap='viridis', s=20, alpha=0.8)
    ax1.set_aspect('equal')
    ax1.set_title(f'{pattern_type} Phyllotaxis Pattern')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label='Seed Number')

    # Polar plot
    ax2.scatter(angles, radii, c=np.arange(n_seeds), cmap='plasma', s=15, alpha=0.7)
    ax2.set_title('Polar Representation')
    ax2.set_xlabel('Angle (radians)')
    ax2.set_ylabel('Radius')
    ax2.grid(True, alpha=0.3)

    # Fibonacci spiral overlay
    theta_spiral = np.linspace(0, 4*np.pi, 1000)
    r_spiral = c * np.sqrt(theta_spiral / divergence_angle)

    ax1.plot(r_spiral * np.cos(theta_spiral), r_spiral * np.sin(theta_spiral),
             'r-', linewidth=2, alpha=0.5, label='Fibonacci Spiral')
    ax1.legend()

    # Angle distribution histogram
    angles_deg = np.array(angles) * 180 / np.pi
    ax3.hist(angles_deg % 360, bins=36, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.set_title('Angle Distribution')
    ax3.set_xlabel('Angle (degrees)')
    ax3.set_ylabel('Count')
    ax3.grid(True, alpha=0.3)

    # Growth pattern analysis
    growth_ratios = []
    for i in range(2, len(radii)):
        ratio = radii[i] / radii[i-1]
        growth_ratios.append(ratio)

    ax4.plot(range(2, len(radii)), growth_ratios, 'g-o', markersize=3, alpha=0.7)
    ax4.axhline(y=np.sqrt(phi_val), color='r', linestyle='--',
                label=f'√φ = {np.sqrt(phi_val):.6f}')
    ax4.set_title('Radial Growth Ratios')
    ax4.set_xlabel('Point Index')
    ax4.set_ylabel('Growth Ratio')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Pattern statistics
    st.subheader("Pattern Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Points", f"{n_seeds}")
    with col2:
        st.metric("Max Radius", f"{np.max(radii):.2f}")
    with col3:
        st.metric("Golden Ratio φ", f"{phi_val:.6f}")
    with col4:
        st.metric("Divergence Angle", f"{angle_offset}°")

    # Fibonacci sequence in phyllotaxis
    st.subheader("Fibonacci Connection")
    fib_sequence = [1, 1]
    for i in range(2, 15):
        fib_sequence.append(fib_sequence[i-1] + fib_sequence[i-2])

    st.write("Fibonacci sequence:", fib_sequence[:10])
    st.write("Fibonacci ratios converging to φ:", [f"{fib_sequence[i+1]/fib_sequence[i]:.6f}" for i in range(5)])

    st.markdown(f"""
    **Phyllotaxis Explanation:**
    - **{pattern_type} Pattern:** Generated using {angle_offset}° divergence angle
    - **Golden Ratio Scaling:** φ = {phi_val:.6f} used for optimal packing efficiency
    - **Fibonacci Spiral:** The red curve shows the logarithmic spiral that optimally fills space
    - **Angle Distribution:** Shows how points are distributed around the pattern
    - **Growth Ratios:** Demonstrate the recursive scaling by √φ for each layer

    These patterns are found in nature (sunflowers, pine cones, broccoli) and represent
    the most efficient way to pack seeds or florets in a growing structure.
    """)
