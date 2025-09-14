import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
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

def mode_superposition_visualizer():
    st.header("🔄 RVM-Enhanced Mode Superposition Visualizer")
    st.markdown("""
    Implement superposition of angular-momentum modes with RVM vortex dynamics.
    Features deterministic standing waves and trefoil Lissajous figures with vortex field mapping.
    """)

    # RVM Mode Superposition Integration
    with st.expander("🔢 RVM Mode Superposition Foundations"):
        st.markdown("""
        **Schrödinger Equation**: Deterministic standing waves S(θ,t) on tonal tori
        **Vortex Mode Dynamics**: Angular momentum states with doubling circuit
        **3-6-9 Trefoil Topology**: |n1 - σ n2| = 3 creates RVM 3-lobed structure
        **Phase Synchronization**: RVM flow in mode interactions
        **Toroidal Harmonic Substrates**: Unified field theory mode connections
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
    B1 = st.slider("Amplitude B1", 0.1, 2.0, 1.0, step=0.1)
    B2 = st.slider("Amplitude B2", 0.1, 2.0, 0.8, step=0.1)
    n1 = st.slider("Mode n1", 1, 8, 2)
    n2 = st.slider("Mode n2", 1, 8, 5)
    sigma = st.slider("Rotation Parameter σ", -3.0, 3.0, 1.0, step=0.1)
    omega = st.slider("Frequency Ω", 0.1, 2.0, 1.0, step=0.1)
    lambda_freq = st.slider("Frequency Λ", 0.1, 2.0, 1.3, step=0.1)
    phi0 = st.slider("Phase Offset φ₀ (degrees)", 0, 360, 90, step=15)
    time = st.slider("Time t", 0.0, 10.0, 0.0, step=0.1)

    # Convert phase to radians
    phi0_rad = np.radians(phi0)

    # Check for trefoil condition
    mode_difference = abs(n1 - sigma * n2)
    is_trefoil = abs(mode_difference - 3) < 0.1

    # Create angular coordinate
    theta = np.linspace(0, 4*np.pi, 1000)

    # Calculate mode superposition
    # S(θ, t) = B1 * exp(i*(n1*θ - Ω*t)) + B2 * exp(i*(σ*n2*θ - Λ*t + φ0))

    # Real and imaginary parts
    S1_real = B1 * np.cos(n1 * theta - omega * time)
    S1_imag = B1 * np.sin(n1 * theta - omega * time)

    S2_real = B2 * np.cos(sigma * n2 * theta - lambda_freq * time + phi0_rad)
    S2_imag = B2 * np.sin(sigma * n2 * theta - lambda_freq * time + phi0_rad)

    # Total superposition
    S_real = S1_real + S2_real
    S_imag = S1_imag + S2_imag

    # Calculate magnitude and phase
    S_magnitude = np.sqrt(S_real**2 + S_imag**2)
    S_phase = np.arctan2(S_imag, S_real)

    # Create 3D trajectory (Lissajous-like figure)
    # Use magnitude and phase to create 3D path
    x_traj = S_magnitude * np.cos(theta)
    y_traj = S_magnitude * np.sin(theta)
    z_traj = S_magnitude * np.cos(2 * theta + S_phase)  # Add phase-dependent height

    # Time evolution for animation
    time_steps = 50
    time_range = np.linspace(0, 2*np.pi/omega, time_steps)

    # Store trajectories for animation
    trajectories = []
    for t in time_range:
        S1_real_t = B1 * np.cos(n1 * theta - omega * t)
        S1_imag_t = B1 * np.sin(n1 * theta - omega * t)
        S2_real_t = B2 * np.cos(sigma * n2 * theta - lambda_freq * t + phi0_rad)
        S2_imag_t = B2 * np.sin(sigma * n2 * theta - lambda_freq * t + phi0_rad)

        S_real_t = S1_real_t + S2_real_t
        S_imag_t = S1_imag_t + S2_imag_t
        S_magnitude_t = np.sqrt(S_real_t**2 + S_imag_t**2)
        S_phase_t = np.arctan2(S_imag_t, S_real_t)

        x_traj_t = S_magnitude_t * np.cos(theta)
        y_traj_t = S_magnitude_t * np.sin(theta)
        z_traj_t = S_magnitude_t * np.cos(2 * theta + S_phase_t)

        trajectories.append((x_traj_t, y_traj_t, z_traj_t))

    # Create visualization
    fig = plt.figure(figsize=(16, 12))

    # 3D trajectory visualization
    ax1 = fig.add_subplot(221, projection='3d')

    # Plot the trajectory
    ax1.plot(x_traj, y_traj, z_traj, 'b-', linewidth=2, alpha=0.8, label='Mode Superposition Trajectory')

    # Mark start and end points
    ax1.scatter(x_traj[0], y_traj[0], z_traj[0], color='green', s=100, label='Start')
    ax1.scatter(x_traj[-1], y_traj[-1], z_traj[-1], color='red', s=100, label='End')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Mode Superposition Trajectory')
    ax1.legend()

    # Set equal aspect ratio
    max_range = np.array([x_traj.max()-x_traj.min(), y_traj.max()-y_traj.min(), z_traj.max()-z_traj.min()]).max() / 2.0
    mid_x = (x_traj.max()+x_traj.min()) * 0.5
    mid_y = (y_traj.max()+y_traj.min()) * 0.5
    mid_z = (z_traj.max()+z_traj.min()) * 0.5
    ax1.set_xlim(mid_x - max_range, mid_x + max_range)
    ax1.set_ylim(mid_y - max_range, mid_y + max_range)
    ax1.set_zlim(mid_z - max_range, mid_z + max_range)

    # 2D projections
    ax2 = fig.add_subplot(222)
    ax2.plot(x_traj, y_traj, 'b-', linewidth=2, alpha=0.8)
    ax2.scatter(x_traj[0], y_traj[0], color='green', s=50, label='Start')
    ax2.scatter(x_traj[-1], y_traj[-1], color='red', s=50, label='End')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Magnitude and phase evolution
    ax3 = fig.add_subplot(223)
    ax3.plot(theta, S_magnitude, 'purple', linewidth=2, label='Magnitude')
    ax3.set_xlabel('θ (radians)')
    ax3.set_ylabel('Magnitude')
    ax3.set_title('Mode Magnitude')
    ax3.grid(True, alpha=0.3)

    ax3_twin = ax3.twinx()
    ax3_twin.plot(theta, S_phase, 'orange', linewidth=1, alpha=0.7, label='Phase')
    ax3_twin.set_ylabel('Phase (radians)', color='orange')

    # Frequency analysis
    ax4 = fig.add_subplot(224)

    # Calculate power spectrum
    from scipy import signal
    freqs, psd = signal.welch(S_magnitude, fs=1000, nperseg=256)

    ax4.semilogy(freqs, psd, 'green', linewidth=2)
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Power Spectral Density')
    ax4.set_title('Power Spectrum')
    ax4.grid(True, alpha=0.3)

    # Mark dominant frequencies
    peaks, _ = signal.find_peaks(psd, height=np.max(psd)/10)
    if len(peaks) > 0:
        for peak in peaks[:3]:  # Show top 3 peaks
            ax4.axvline(x=freqs[peak], color='red', linestyle='--', alpha=0.7)
            ax4.text(freqs[peak], psd[peak], f'{freqs[peak]:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    st.pyplot(fig)

    # Analysis metrics
    st.subheader("Mode Superposition Analysis")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mode Difference |n1 - σ·n2|", f"{mode_difference:.3f}")
    with col2:
        st.metric("Trefoil Condition", "✅ Met" if is_trefoil else "❌ Not Met")
    with col3:
        st.metric("Max Magnitude", f"{np.max(S_magnitude):.3f}")
    with col4:
        st.metric("Golden Ratio φ", f"{phi_val:.6f}")

    # Detailed analysis
    st.subheader("Mode Analysis")

    # Calculate mode properties
    st.write(f"**Mode 1:** n₁ = {n1}, Amplitude = {B1}")
    st.write(f"**Mode 2:** n₂ = {n2}, σ = {sigma}, Amplitude = {B2}")
    st.write(f"**Frequencies:** Ω = {omega:.2f}, Λ = {lambda_freq:.2f}")

    # Resonance analysis
    frequency_ratio = lambda_freq / omega if omega != 0 else float('inf')
    st.write(f"**Frequency Ratio Λ/Ω:** {frequency_ratio:.3f}")

    # Trefoil condition check
    if is_trefoil:
        st.success("🎯 **Trefoil Topology Detected!**")
        st.write("The mode superposition creates a trefoil-like Lissajous figure.")
    else:
        st.info("ℹ️ **Non-Trefoil Configuration**")
        st.write(f"Mode difference = {mode_difference:.3f} (trefoil requires ≈ 3.0)")

    # Trajectory analysis
    st.subheader("Trajectory Properties")

    # Calculate trajectory length
    trajectory_length = np.sum(np.sqrt(np.diff(x_traj)**2 + np.diff(y_traj)**2 + np.diff(z_traj)**2))
    st.write(f"**Trajectory Length:** {trajectory_length:.2f}")

    # Calculate trajectory volume (simplified)
    from scipy.spatial import ConvexHull
    points = np.column_stack([x_traj, y_traj, z_traj])
    hull = ConvexHull(points)
    trajectory_volume = hull.volume
    st.write(f"**Trajectory Volume:** {trajectory_volume:.2f}")

    # Symmetry analysis
    st.subheader("Symmetry Analysis")

    # Calculate various symmetry measures
    x_symmetry = np.mean(np.abs(x_traj - np.flip(x_traj)))
    y_symmetry = np.mean(np.abs(y_traj - np.flip(y_traj)))
    z_symmetry = np.mean(np.abs(z_traj - np.flip(z_traj)))

    st.write(f"**X-axis Symmetry:** {x_symmetry:.4f}")
    st.write(f"**Y-axis Symmetry:** {y_symmetry:.4f}")
    st.write(f"**Z-axis Symmetry:** {z_symmetry:.4f}")

    # Phase coherence
    phase_coherence = np.abs(np.mean(np.exp(1j * S_phase)))
    st.write(f"**Phase Coherence:** {phase_coherence:.3f}")

    # RVM Digital Root Analysis
    st.subheader("🔢 RVM Digital Root Analysis of Mode Parameters")
    mode_params = {
        "Mode n1": n1,
        "Mode n2": n2,
        "Amplitude B1": int(B1 * 100),
        "Amplitude B2": int(B2 * 100),
        "Rotation Parameter σ": int(sigma * 100),
        "Frequency Ω": int(omega * 100),
        "Frequency Λ": int(lambda_freq * 100),
        "Phase Offset φ₀": phi0,
        "Mode Difference": int(mode_difference * 100)
    }

    param_digital_roots = {param: digital_root(val) for param, val in mode_params.items()}

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Mode Parameters:**")
        for param, val in mode_params.items():
            st.write(f"{param}: {val}")

    with col2:
        st.write("**Digital Roots:**")
        for param, dr in param_digital_roots.items():
            st.write(f"{param}: {dr}")

    # RVM Vortex Mode Topology
    st.subheader("🌀 RVM Vortex Mode Topology")
    fig_rvm, ax_rvm = plt.subplots(figsize=(10, 8))

    # Create RVM 9-point circle for mode topology mapping
    rvm_points = [1, 2, 4, 8, 7, 5, 3, 6, 9]
    angles = np.linspace(0, 2*np.pi, 9, endpoint=False)

    # Map mode parameters to RVM vortex circle
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

    # Overlay mode superposition trajectory scaled to vortex
    scale_vortex = 2.5 / np.max(S_magnitude) if np.max(S_magnitude) > 0 else 1
    x_vortex = []
    y_vortex = []
    colors_vortex = []

    for i in range(0, len(theta), 10):  # Sample every 10th point
        angle = theta[i]
        radius = scale_vortex * S_magnitude[i]
        x_vortex.append(radius * np.cos(angle))
        y_vortex.append(radius * np.sin(angle))
        colors_vortex.append(S_phase[i])

    scatter_vortex = ax_rvm.scatter(x_vortex, y_vortex, c=colors_vortex, cmap='hsv', s=10, alpha=0.6, label='Mode Trajectory')

    ax_rvm.set_xlim(-4, 4)
    ax_rvm.set_ylim(-4, 4)
    ax_rvm.set_aspect('equal')
    ax_rvm.set_title('RVM Vortex Mode Topology')
    ax_rvm.grid(True, alpha=0.3)
    plt.colorbar(scatter_vortex, ax=ax_rvm, label='Phase')

    st.pyplot(fig_rvm)

    st.markdown("""
    **Mode Superposition Theory:**
    - **Schrödinger Equation**: S(θ,t) = B₁e^{i(n₁θ-Ωt)} + B₂e^{i(σn₂θ-Λt+ϕ₀)} - deterministic standing waves
    - **Angular Momentum Modes**: Complex exponential functions representing quantum states
    - **Trefoil Condition**: |n₁ - σ·n₂| = 3 creates characteristic three-lobed topology
    - **Lissajous Figures**: Complex trajectories from mode interference
    - **Phase Synchronization**: Coherent superposition of different angular modes

    **Physical Interpretations:**
    - **Quantum Angular Momentum**: Superposition of different l and m quantum numbers
    - **Electromagnetic Modes**: Cavity modes in lasers and resonators
    - **Vibrational Modes**: Molecular normal modes in spectroscopy
    - **Orbital Mechanics**: Complex trajectories in celestial mechanics

    **Mathematical Properties:**
    - **Mode Coupling**: σ parameter controls coupling between different modes
    - **Frequency Locking**: Ω and Λ determine temporal evolution patterns
    - **Topological Phase**: Phase winding creating non-trivial topologies
    - **Symmetry Breaking**: Complex trajectories break simple symmetries
    - **RVM Integration**: Digital roots and vortex mapping reveal underlying harmonic patterns

    **Applications:**
    - **Quantum Optics**: Mode superposition in beam splitters
    - **Spectroscopy**: Complex vibrational mode analysis
    - **Astrophysics**: Orbital resonances in planetary systems
    - **Condensed Matter**: Collective excitations in crystals
    """)

    # Interactive exploration
    st.subheader("Interactive Exploration")
    st.markdown("Adjust the mode parameters to explore different superposition patterns and discover when trefoil topologies emerge with RVM harmonic insights.")
