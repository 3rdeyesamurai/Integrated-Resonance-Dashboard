import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import sqrt

def digital_root(n):
    """Calculate digital root of a number (n mod 9, with 9 as singularity)"""
    if n == 0:
        return 9
    return 1 + (n - 1) % 9

def vortex_doubling_sequence():
    """Generate RVM doubling circuit sequence: 1→2→4→8→7→5→1"""
    sequence = [1]
    current = 1
    for _ in range(6):
        current = (current * 2) % 9
        if current == 0:
            current = 9
        sequence.append(current)
    return sequence

def coordinated_rotation_field_emulator():
    st.header("🔄 RVM-Enhanced Coordinated-Rotation Field Emulator")
    st.markdown("""
    Simulate coordinated rotation fields with RVM field emulation.
    Features trefoil topology and polarization knots with vortex dynamics.
    """)

    # RVM Coordinated Rotation Integration
    with st.expander("🔢 RVM Coordinated Rotation Foundations"):
        st.markdown("""
        **Trefoil Topology**: |m1 - σ m2| = 3 creates RVM 3-lobed structure
        **Polarization Knots**: Field polarization following RVM vortex patterns
        **Coordinated Rotations**: Angular momentum states with doubling circuit
        **Phase Synchronization**: RVM flow in field interactions
        **Toroidal Field Dynamics**: Harmonic substrates for rotation fields
        """)

        phi = (1 + sqrt(5)) / 2
        cm = 3 / float(phi.evalf())
        doubling_seq = vortex_doubling_sequence()
        st.write(f"**Golden Ratio φ** = {float(phi.evalf()):.6f}")
        st.write(f"**Corrective Constant Cm** = 3/φ = {cm:.6f}")
        st.write(f"**RVM Doubling Sequence**: {doubling_seq}")

    # Calculate golden ratio
    phi = (1 + sqrt(5)) / 2
    phi_val = float(phi.evalf())

    # Input parameters
    A1 = st.slider("Amplitude A1", 0.1, 2.0, 1.0, step=0.1)
    A2 = st.slider("Amplitude A2", 0.1, 2.0, 0.8, step=0.1)
    m1 = st.slider("Mode m1", 1, 5, 2)
    m2 = st.slider("Mode m2", 1, 5, 3)
    sigma = st.slider("Rotation Parameter σ", -2.0, 2.0, 1.0, step=0.1)
    omega = st.slider("Frequency ω", 0.1, 2.0, 1.0, step=0.1)
    nu = st.slider("Frequency ν", 0.1, 2.0, 1.2, step=0.1)
    phi0 = st.slider("Phase Offset φ₀ (degrees)", 0, 360, 90, step=15)
    time = st.slider("Time t", 0.0, 10.0, 0.0, step=0.1)

    # Convert phase to radians
    phi0_rad = np.radians(phi0)

    # Check for trefoil condition
    mode_difference = abs(m1 - sigma * m2)
    is_trefoil = abs(mode_difference - 3) < 0.1

    # Create spatial grid
    theta = np.linspace(0, 4*np.pi, 200)  # Angular coordinate
    r = np.linspace(0.1, 2.0, 50)  # Radial coordinate
    Theta, R = np.meshgrid(theta, r)

    # Calculate electric field components
    # E(θ, t) = A1 * exp(i*(m1*θ - ω*t)) + A2 * exp(i*(σ*m2*θ - ν*t + φ0))

    # Real and imaginary parts
    E1_real = A1 * np.cos(m1 * Theta - omega * time)
    E1_imag = A1 * np.sin(m1 * Theta - omega * time)

    E2_real = A2 * np.cos(sigma * m2 * Theta - nu * time + phi0_rad)
    E2_imag = A2 * np.sin(sigma * m2 * Theta - nu * time + phi0_rad)

    # Total field
    E_real = E1_real + E2_real
    E_imag = E1_imag + E2_imag

    # Calculate field magnitude and phase
    E_magnitude = np.sqrt(E_real**2 + E_imag**2)
    E_phase = np.arctan2(E_imag, E_real)

    # Polarization analysis
    # Calculate polarization ellipse parameters
    polarization_angle = 0.5 * np.arctan2(2 * (E1_real * E2_real + E1_imag * E2_imag),
                                         (E1_real**2 + E1_imag**2 - E2_real**2 - E2_imag**2))

    # Create 3D coordinates for visualization
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    Z = E_magnitude * np.sin(E_phase)  # Height based on field structure

    # Create visualization
    fig = plt.figure(figsize=(16, 12))

    # 3D field visualization
    ax1 = fig.add_subplot(221, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, facecolors=plt.cm.viridis(E_magnitude/np.max(E_magnitude)),
                           rstride=2, cstride=2, alpha=0.8)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Field Structure')
    ax1.set_title('3D Field Topology')
    plt.colorbar(surf, ax=ax1, shrink=0.5, label='Field Magnitude')

    # 2D field pattern
    ax2 = fig.add_subplot(222)
    im = ax2.pcolormesh(Theta, R, E_magnitude, shading='gouraud', cmap='plasma')
    ax2.set_xlabel('θ (radians)')
    ax2.set_ylabel('r')
    ax2.set_title('Field Magnitude Pattern')
    plt.colorbar(im, ax=ax2, label='Magnitude')

    # Polarization pattern
    ax3 = fig.add_subplot(223)
    pol_im = ax3.pcolormesh(Theta, R, polarization_angle, shading='gouraud', cmap='twilight')
    ax3.set_xlabel('θ (radians)')
    ax3.set_ylabel('r')
    ax3.set_title('Polarization Angle')
    plt.colorbar(pol_im, ax=ax3, label='Polarization Angle (rad)')

    # Time evolution of field at fixed radius
    ax4 = fig.add_subplot(224)
    r_fixed = 1.0
    theta_fixed = np.linspace(0, 4*np.pi, 100)

    # Calculate field evolution over time
    time_range = np.linspace(0, 2*np.pi/omega, 50)
    field_evolution = np.zeros((len(time_range), len(theta_fixed)))

    for i, t in enumerate(time_range):
        E1_t = A1 * np.cos(m1 * theta_fixed - omega * t)
        E2_t = A2 * np.cos(sigma * m2 * theta_fixed - nu * t + phi0_rad)
        field_evolution[i, :] = E1_t + E2_t

    im_evol = ax4.imshow(field_evolution, extent=[0, 4*np.pi, 0, 2*np.pi/omega],
                        aspect='auto', cmap='RdYlBu_r', origin='lower')
    ax4.set_xlabel('θ (radians)')
    ax4.set_ylabel('Time')
    ax4.set_title('Field Evolution')
    plt.colorbar(im_evol, ax=ax4, label='Field Strength')

    plt.tight_layout()
    st.pyplot(fig)

    # RVM Digital Root Analysis
    st.subheader("🔢 RVM Digital Root Analysis of Rotation Parameters")
    rotation_params = {
        "Amplitude A1": A1,
        "Amplitude A2": A2,
        "Mode m1": m1,
        "Mode m2": m2,
        "Rotation Parameter σ": sigma,
        "Frequency ω": omega,
        "Frequency ν": nu,
        "Phase Offset φ₀": phi0,
        "Mode Difference": mode_difference
    }

    param_digital_roots = {param: digital_root(int(val * 100) if val < 1 else int(val))
                          for param, val in rotation_params.items()}

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Rotation Parameters:**")
        for param, val in rotation_params.items():
            st.write(f"{param}: {val:.6f}")

    with col2:
        st.write("**Digital Roots:**")
        for param, dr in param_digital_roots.items():
            st.write(f"{param}: {dr}")

    # RVM Vortex Field Topology
    st.subheader("🌀 RVM Vortex Field Topology")
    fig_rvm, ax_rvm = plt.subplots(figsize=(10, 8))

    # Create RVM 9-point circle for field topology mapping
    rvm_points = [1, 2, 4, 8, 7, 5, 3, 6, 9]
    angles = np.linspace(0, 2*np.pi, 9, endpoint=False)

    # Map field topology to RVM vortex circle
    for i, (point, angle) in enumerate(zip(rvm_points, angles)):
        x_point = 3 * np.cos(angle)
        y_point = 3 * np.sin(angle)

        # Color based on 3-6-9 triad
        if point in [3, 6, 9]:
            color = 'red'
        else:
            color = 'blue'

        ax_rvm.scatter(x_point, y_point, s=200, c=color, alpha=0.8, edgecolors='black')
        ax_rvm.text(x_point, y_point, str(point), ha='center', va='center', fontsize=12, fontweight='bold')

    # Draw circle
    circle = plt.Circle((0, 0), 3, fill=False, color='gray', linestyle='--', alpha=0.5)
    ax_rvm.add_artist(circle)

    # Overlay field topology scaled to vortex
    scale_vortex = 2.5 / np.max(E_magnitude) if np.max(E_magnitude) > 0 else 1
    x_vortex = []
    y_vortex = []
    colors_vortex = []

    for i in range(0, len(Theta), 10):  # Sample every 10th point
        for j in range(0, len(Theta[i]), 10):
            x_vortex.append(scale_vortex * E_magnitude[i,j] * np.cos(Theta[i,j]))
            y_vortex.append(scale_vortex * E_magnitude[i,j] * np.sin(Theta[i,j]))
            colors_vortex.append(E_magnitude[i,j])

    scatter_vortex = ax_rvm.scatter(x_vortex, y_vortex, c=colors_vortex, cmap='viridis', s=5, alpha=0.6, label='Field Topology')

    ax_rvm.set_xlim(-4, 4)
    ax_rvm.set_ylim(-4, 4)
    ax_rvm.set_aspect('equal')
    ax_rvm.set_title('RVM Vortex Field Topology')
    ax_rvm.grid(True, alpha=0.3)
    plt.colorbar(scatter_vortex, ax=ax_rvm, label='Field Magnitude')

    st.pyplot(fig_rvm)

    # Analysis metrics
    st.subheader("Field Analysis")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mode Difference |m1 - σ·m2|", f"{mode_difference:.3f}")
    with col2:
        st.metric("Trefoil Condition", "✅ Met" if is_trefoil else "❌ Not Met")
    with col3:
        st.metric("Max Field Magnitude", f"{np.max(E_magnitude):.3f}")
    with col4:
        st.metric("Golden Ratio φ", f"{phi_val:.6f}")

    # Detailed analysis
    st.subheader("Coordinated Rotation Analysis")

    # Calculate field properties
    avg_magnitude = np.mean(E_magnitude)
    field_variance = np.var(E_magnitude)
    coherence_factor = np.abs(np.mean(np.exp(1j * E_phase)))

    st.write(f"**Average Field Magnitude:** {avg_magnitude:.3f}")
    st.write(f"**Field Variance:** {field_variance:.3f}")
    st.write(f"**Phase Coherence:** {coherence_factor:.3f}")

    # Knot analysis
    if is_trefoil:
        st.success("🎯 **Trefoil Topology Detected!**")
        st.write("The field configuration creates a trefoil knot structure in the polarization field.")
    else:
        st.info("ℹ️ **Non-Trefoil Configuration**")
        st.write(f"Mode difference = {mode_difference:.3f} (trefoil requires ≈ 3.0)")

    # Polarization analysis
    st.subheader("Polarization Characteristics")

    # Calculate ellipticity
    ellipticity = np.sqrt(1 - (np.min(E_magnitude) / np.max(E_magnitude))**2)
    st.write(f"**Field Ellipticity:** {ellipticity:.3f}")

    # Calculate rotation direction
    rotation_direction = "Clockwise" if sigma > 0 else "Counter-clockwise"
    st.write(f"**Rotation Direction:** {rotation_direction}")

    # Frequency analysis
    frequency_ratio = nu / omega if omega != 0 else float('inf')
    st.write(f"**Frequency Ratio ν/ω:** {frequency_ratio:.3f}")

    # Energy analysis
    total_energy = np.sum(E_magnitude**2)
    energy_ratio = A2**2 / A1**2 if A1 != 0 else float('inf')
    st.write(f"**Total Field Energy:** {total_energy:.1f}")
    st.write(f"**Amplitude Ratio A2/A1:** {energy_ratio:.3f}")

    st.markdown(f"""
    **Coordinated Rotation Field Theory:**
    - **Complex Field Superposition**: E(θ,t) combines two rotating modes with different frequencies
    - **Trefoil Condition**: |m₁ - σ·m₂| = 3 creates characteristic three-lobed topology
    - **Polarization Knots**: Field polarization traces out knotted structures in space-time
    - **Phase Synchronization**: Relative phase φ₀ affects interference patterns

    **Physical Interpretations:**
    - **Electromagnetic Fields**: Polarization states in complex light fields
    - **Quantum Systems**: Angular momentum superposition in atomic physics
    - **Fluid Dynamics**: Vortex interactions in rotating fluids
    - **Plasma Physics**: Magnetic field configurations in fusion devices

    **Mathematical Properties:**
    - **Mode Coupling**: σ parameter controls the coupling between different angular modes
    - **Frequency Locking**: ω and ν determine the temporal evolution patterns
    - **Topological Invariants**: Knot structures preserve certain properties under continuous deformation
    - **Symmetry Breaking**: Phase offset φ₀ breaks rotational symmetry

    **Applications:**
    - **Optical Communications**: Complex modulation schemes for data transmission
    - **Quantum Computing**: Qubit states in angular momentum representation
    - **Astrophysics**: Magnetic field topologies in cosmic structures
    - **Biophysics**: Protein folding and molecular chirality
    """)

    # Interactive exploration
    st.subheader("Interactive Exploration")
    st.markdown("Adjust the mode parameters to explore different field topologies and discover when trefoil knots emerge in the polarization field.")
