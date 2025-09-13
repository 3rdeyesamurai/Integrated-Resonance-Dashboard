import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sqrt, pi

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

def corrective_mirror_constant_calculator():
    st.header("🔮 RVM-Enhanced Corrective Mirror Constant Calculator")
    st.markdown("""
    Compute C_m = 3/φ ≈ 1.854 with RVM digital roots integration.
    Features phase restoration using RVM doubling circuit and 3-6-9 control axis.
    """)

    # RVM Corrective Mirror Integration
    with st.expander("🔢 RVM Corrective Mirror Foundations"):
        st.markdown("""
        **Corrective Constant**: Cm = 3/φ ≈ 1.854 (phase restoration)
        **RVM Digital Roots**: Applied to iteration parameters
        **Doubling Circuit Feedback**: 1→2→4→8→7→5→1 sequence in corrections
        **3-6-9 Control**: Triad values as equilibrium points
        **Vortex Phase Dynamics**: Spiral convergence to golden ratio equilibrium
        """)

        phi = (1 + sqrt(5)) / 2
        cm = 3 / float(phi.evalf())
        doubling_seq = vortex_doubling_sequence()
        st.write(f"**Golden Ratio φ** = {float(phi.evalf()):.6f}")
        st.write(f"**Corrective Constant Cm** = 3/φ = {cm:.6f}")
        st.write(f"**RVM Doubling Sequence**: {doubling_seq}")

    # Calculate golden ratio and corrective mirror constant
    phi = (1 + sqrt(5)) / 2
    phi_val = float(phi.evalf())
    C_m = 3 / phi_val

    st.metric("Corrective Mirror Constant C_m", f"{C_m:.6f}")

    # Input parameters
    initial_phase = st.slider("Initial Phase Offset (degrees)", 0, 180, 45, step=5)
    n_iterations = st.slider("Number of Iterations", 5, 50, 20)
    feedback_strength = st.slider("Feedback Strength", 0.1, 2.0, 0.8, step=0.1)
    equilibrium_threshold = st.slider("Equilibrium Threshold", 0.001, 0.1, 0.01, step=0.001)

    # Convert initial phase to radians
    initial_phase_rad = initial_phase * np.pi / 180

    # Simulation of phase correction using corrective mirror constant
    phases = [initial_phase_rad]
    corrections = []
    equilibrium_errors = []

    current_phase = initial_phase_rad

    for i in range(n_iterations):
        # Calculate phase error (deviation from equilibrium)
        phase_error = current_phase % (2 * np.pi)

        # Apply corrective mirror feedback
        correction = -feedback_strength * C_m * np.sin(phase_error)
        corrections.append(correction)

        # Update phase
        current_phase += correction
        phases.append(current_phase)

        # Calculate equilibrium error
        equilibrium_error = abs(np.sin(current_phase))
        equilibrium_errors.append(equilibrium_error)

        # Check for convergence
        if equilibrium_error < equilibrium_threshold:
            st.success(f"Equilibrium reached at iteration {i+1}")
            break

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Phase evolution
    iterations = range(len(phases))
    ax1.plot(iterations, np.array(phases) * 180 / np.pi, 'b-o', linewidth=2, markersize=4)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.7, label='Equilibrium (0°)')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Phase (degrees)')
    ax1.set_title('Phase Evolution with Corrective Feedback')
    ax1.grid(True)
    ax1.legend()

    # Correction magnitude
    ax2.plot(range(len(corrections)), np.abs(corrections), 'g-s', linewidth=2, markersize=4)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Correction Magnitude')
    ax2.set_title('Feedback Correction Magnitude')
    ax2.grid(True)

    # Equilibrium error convergence
    ax3.semilogy(range(len(equilibrium_errors)), equilibrium_errors, 'r-^', linewidth=2, markersize=4)
    ax3.axhline(y=equilibrium_threshold, color='orange', linestyle='--', label=f'Threshold: {equilibrium_threshold}')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Equilibrium Error (log scale)')
    ax3.set_title('Convergence to Equilibrium')
    ax3.grid(True)
    ax3.legend()

    # Phase portrait (phase vs phase velocity)
    if len(phases) > 1:
        phase_velocities = np.diff(phases)
        ax4.plot(np.array(phases[:-1]) * 180 / np.pi, phase_velocities, 'purple', alpha=0.7)
        ax4.scatter(np.array(phases[:-1]) * 180 / np.pi, phase_velocities, c=range(len(phase_velocities)), cmap='viridis', s=30)
        ax4.set_xlabel('Phase (degrees)')
        ax4.set_ylabel('Phase Velocity')
        ax4.set_title('Phase Portrait')
        ax4.grid(True)
        plt.colorbar(ax4.collections[0], ax=ax4, label='Iteration')

    plt.tight_layout()
    st.pyplot(fig)

    # RVM Vortex Phase Dynamics
    st.subheader("🌀 RVM Vortex Phase Dynamics")
    fig_rvm, ax_rvm = plt.subplots(figsize=(10, 8))

    # Create vortex pattern for phase evolution
    rvm_points = [1, 2, 4, 8, 7, 5, 3, 6, 9]
    angles = np.linspace(0, 2*np.pi, 9, endpoint=False)

    # Plot RVM 9-point circle
    for i, (point, angle) in enumerate(zip(rvm_points, angles)):
        x_point = 3 * np.cos(angle)
        y_point = 3 * np.sin(angle)

        # Color code based on 3-6-9 triad
        if point in [3, 6, 9]:
            color = 'red'
        else:
            color = 'blue'

        ax_rvm.scatter(x_point, y_point, s=200, c=color, alpha=0.8, edgecolors='black')
        ax_rvm.text(x_point, y_point, str(point), ha='center', va='center', fontsize=12, fontweight='bold')

    # Draw circle
    circle = plt.Circle((0, 0), 3, fill=False, color='gray', linestyle='--', alpha=0.5)
    ax_rvm.add_artist(circle)

    # Plot phase evolution as spiral trajectory
    phase_angles = np.array(phases) % (2 * np.pi)
    spiral_radii = 1 + 2 * np.array(equilibrium_errors)  # Radius decreases with convergence

    x_trajectory = spiral_radii * np.cos(phase_angles)
    y_trajectory = spiral_radii * np.sin(phase_angles)

    ax_rvm.plot(x_trajectory, y_trajectory, 'purple', linewidth=3, alpha=0.8, label='Phase Trajectory')
    ax_rvm.scatter(x_trajectory[0], y_trajectory[0], s=100, c='green', marker='*', label='Start')
    ax_rvm.scatter(x_trajectory[-1], y_trajectory[-1], s=100, c='red', marker='X', label='End')

    ax_rvm.set_xlim(-4, 4)
    ax_rvm.set_ylim(-4, 4)
    ax_rvm.set_aspect('equal')
    ax_rvm.set_title('RVM Vortex Phase Dynamics')
    ax_rvm.grid(True, alpha=0.3)
    ax_rvm.legend()

    # Add legend for RVM points
    red_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='3-6-9 Control Axis')
    blue_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Doubling Circuit')
    ax_rvm.legend(handles=[red_dot, blue_dot, plt.Line2D([0], [0], color='purple', linewidth=2, label='Phase Trajectory')])

    st.pyplot(fig_rvm)

    # RVM Digital Root Analysis
    st.subheader("🔢 RVM Digital Root Analysis of Simulation Parameters")
    sim_params = {
        "Initial Phase": initial_phase,
        "Iterations": n_iterations,
        "Feedback Strength": feedback_strength,
        "Equilibrium Threshold": equilibrium_threshold,
        "Final Phase": phases[-1] * 180 / np.pi,
        "Final Error": equilibrium_errors[-1]
    }

    param_digital_roots = {param: digital_root(int(val * 100) if val < 1 else int(val))
                          for param, val in sim_params.items()}

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Simulation Parameters:**")
        for param, val in sim_params.items():
            st.write(f"{param}: {val:.6f}")

    with col2:
        st.write("**Digital Roots:**")
        for param, dr in param_digital_roots.items():
            st.write(f"{param}: {dr}")

    # Analysis metrics
    st.subheader("Simulation Results")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Initial Phase", f"{initial_phase}°")
    with col2:
        st.metric("Final Phase", f"{phases[-1] * 180 / np.pi:.2f}°")
    with col3:
        st.metric("Total Iterations", len(phases)-1)
    with col4:
        st.metric("Final Error", f"{equilibrium_errors[-1]:.6f}")

    # Convergence analysis
    if len(equilibrium_errors) > 1:
        convergence_rate = (equilibrium_errors[0] - equilibrium_errors[-1]) / len(equilibrium_errors)
        st.metric("Average Convergence Rate", f"{convergence_rate:.6f}")

    # Theoretical analysis
    st.subheader("Theoretical Analysis")

    # Calculate theoretical convergence properties
    theoretical_period = 2 * np.pi / (feedback_strength * C_m)
    st.write(f"**Theoretical Period:** {theoretical_period:.2f} iterations")
    st.write(f"**Corrective Mirror Constant:** C_m = 3/φ = {C_m:.6f}")
    st.write(f"**Feedback Strength:** {feedback_strength}")
    st.write(f"**Effective Gain:** {feedback_strength * C_m:.4f}")

    # Stability analysis
    stability_criterion = feedback_strength * C_m
    if stability_criterion < 1:
        stability = "Stable"
        stability_color = "🟢"
    elif stability_criterion < 2:
        stability = "Marginally Stable"
        stability_color = "🟡"
    else:
        stability = "Unstable"
        stability_color = "🔴"

    st.subheader(f"Stability Analysis: {stability_color} {stability}")

    # Mathematical explanation
    st.markdown(f"""
    **Corrective Mirror Constant Theory:**
    - **C_m = 3/φ ≈ {C_m:.6f}** provides optimal feedback gain for harmonic systems
    - **Phase Correction:** Δφ = -k * C_m * sin(φ) where k is feedback strength
    - **Equilibrium Condition:** System reaches φ = 0 (or multiples of 2π)
    - **Golden Ratio Scaling:** φ ensures natural resonance frequencies

    **Applications:**
    - **Phase-locked loops** in electronics
    - **Neural synchronization** in brain models
    - **Quantum state stabilization** in field theory
    - **Musical instrument tuning** systems
    - **Biological rhythm regulation**

    **Convergence Properties:**
    - Exponential convergence when |k * C_m| < 1 (stable)
    - Oscillatory behavior when |k * C_m| > 1 (unstable)
    - Optimal performance at k * C_m ≈ 0.618 (golden ratio)
    """)

    # Interactive demonstration
    st.subheader("Interactive Demonstration")
    st.markdown("Adjust the parameters above to see how the corrective mirror constant affects phase stabilization and convergence behavior.")
