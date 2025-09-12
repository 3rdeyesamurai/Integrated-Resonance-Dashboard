import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import sqrt

def tonal_torus_trajectory_simulator():
    st.header("Tonal Torus Trajectory Simulator")
    st.markdown("Use NumPy and Matplotlib to model pitch-class motion on T^2 manifold, applying scalar step operators T_k (θ → θ + kΔ) for seconds, thirds, and fourths, visualizing three-lobed paths.")

    # Calculate golden ratio
    phi = (1 + sqrt(5)) / 2
    phi_val = float(phi.evalf())

    # Input parameters
    n_steps = st.slider("Number of Steps", 10, 100, 50)
    step_size = st.slider("Step Size Δ (semitones)", 1, 12, 7, step=1)  # Perfect fifth by default
    interval_type = st.selectbox("Interval Type", ["Seconds", "Thirds", "Fourths", "Fifths", "Custom"])
    modulation_rate = st.slider("Modulation Rate", 0.1, 2.0, 1.0, step=0.1)
    torus_major = st.slider("Torus Major Radius", 2.0, 6.0, 4.0, step=0.1)
    torus_minor = st.slider("Torus Minor Radius", 0.5, 2.0, 1.0, step=0.1)

    # Define interval mappings
    interval_map = {
        "Seconds": [1, 2],  # Minor and major seconds
        "Thirds": [3, 4],   # Minor and major thirds
        "Fourths": [5],     # Perfect fourth
        "Fifths": [7],      # Perfect fifth
        "Custom": [step_size]
    }

    selected_intervals = interval_map[interval_type]

    # Initialize pitch classes on torus (T^2 manifold)
    # Represent as angles on the torus
    theta = np.zeros(n_steps)  # Longitudinal angle (pitch class)
    phi = np.zeros(n_steps)    # Latitudinal angle (octave/register)

    # Starting point
    theta[0] = 0.0
    phi[0] = 0.0

    # Generate trajectory using scalar step operators T_k
    for i in range(1, n_steps):
        # Choose random interval from selected type
        k = np.random.choice(selected_intervals)

        # Apply scalar step operator T_k: θ → θ + kΔ
        delta_theta = k * step_size * np.pi / 6  # Convert semitones to radians (12 semitones = 2π)

        # Add modulation
        modulation = modulation_rate * np.sin(2 * np.pi * i / n_steps) * np.pi / 12

        # Update angles
        theta[i] = (theta[i-1] + delta_theta + modulation) % (2 * np.pi)
        phi[i] = (phi[i-1] + delta_theta * 0.1) % (2 * np.pi)  # Slower octave changes

    # Convert to 3D Cartesian coordinates on torus
    X = (torus_major + torus_minor * np.cos(phi)) * np.cos(theta)
    Y = (torus_major + torus_minor * np.cos(phi)) * np.sin(theta)
    Z = torus_minor * np.sin(phi)

    # Create torus surface for reference
    u_torus = np.linspace(0, 2*np.pi, 50)
    v_torus = np.linspace(0, 2*np.pi, 50)
    U_torus, V_torus = np.meshgrid(u_torus, v_torus)

    X_torus = (torus_major + torus_minor * np.cos(V_torus)) * np.cos(U_torus)
    Y_torus = (torus_major + torus_minor * np.cos(V_torus)) * np.sin(U_torus)
    Z_torus = torus_minor * np.sin(V_torus)

    # Calculate trajectory properties
    # Distance traveled
    distances = np.sqrt(np.diff(X)**2 + np.diff(Y)**2 + np.diff(Z)**2)
    total_distance = np.sum(distances)

    # Pitch class distribution
    pitch_classes = (theta * 12 / (2 * np.pi)) % 12  # Convert to semitones

    # Interval distribution
    intervals_used = []
    for i in range(1, len(theta)):
        interval = int(np.round((theta[i] - theta[i-1]) * 12 / (2 * np.pi))) % 12
        intervals_used.append(interval)

    # Create visualization
    fig = plt.figure(figsize=(16, 12))

    # 3D trajectory on torus
    ax1 = fig.add_subplot(221, projection='3d')

    # Plot torus surface (semi-transparent)
    ax1.plot_surface(X_torus, Y_torus, Z_torus, alpha=0.1, color='gray')

    # Plot trajectory
    scatter = ax1.scatter(X, Y, Z, c=range(n_steps), cmap='viridis', s=30, alpha=0.8)
    ax1.plot(X, Y, Z, 'b-', alpha=0.6, linewidth=2)

    # Mark start and end
    ax1.scatter(X[0], Y[0], Z[0], color='green', s=100, label='Start')
    ax1.scatter(X[-1], Y[-1], Z[-1], color='red', s=100, label='End')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Tonal Trajectory on T² Manifold')
    ax1.legend()
    plt.colorbar(scatter, ax=ax1, shrink=0.5, label='Step Number')

    # 2D projection
    ax2 = fig.add_subplot(222)
    ax2.plot(X, Y, 'b-', alpha=0.8, linewidth=2)
    ax2.scatter(X, Y, c=range(n_steps), cmap='viridis', s=20, alpha=0.7)
    ax2.scatter(X[0], Y[0], color='green', s=50, label='Start')
    ax2.scatter(X[-1], Y[-1], color='red', s=50, label='End')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Pitch class evolution
    ax3 = fig.add_subplot(223)
    ax3.plot(range(n_steps), pitch_classes, 'purple', linewidth=2, marker='o', markersize=3)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Pitch Class (semitones)')
    ax3.set_title('Pitch Class Evolution')
    ax3.grid(True, alpha=0.3)

    # Add pitch class labels
    pitch_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    ax3.set_yticks(range(12))
    ax3.set_yticklabels(pitch_names)

    # Interval distribution
    ax4 = fig.add_subplot(224)
    if intervals_used:
        unique_intervals, counts = np.unique(intervals_used, return_counts=True)
        bars = ax4.bar(unique_intervals, counts, color='skyblue', alpha=0.7)

        # Add interval labels
        interval_names = ['Unison', 'Minor 2nd', 'Major 2nd', 'Minor 3rd', 'Major 3rd', 'Perfect 4th',
                         'Tritone', 'Perfect 5th', 'Minor 6th', 'Major 6th', 'Minor 7th', 'Major 7th']
        ax4.set_xticks(range(12))
        ax4.set_xticklabels([interval_names[i] for i in range(12)], rotation=45, ha='right')

        ax4.set_xlabel('Interval')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Interval Distribution')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Analysis metrics
    st.subheader("Tonal Trajectory Analysis")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Steps", n_steps)
    with col2:
        st.metric("Trajectory Length", f"{total_distance:.2f}")
    with col3:
        st.metric("Unique Pitch Classes", f"{len(np.unique(np.round(pitch_classes)))}")
    with col4:
        st.metric("Golden Ratio φ", f"{phi_val:.6f}")

    # Detailed analysis
    st.subheader("Trajectory Properties")

    # Calculate various metrics
    avg_step_size = np.mean(distances) if len(distances) > 0 else 0
    std_step_size = np.std(distances) if len(distances) > 0 else 0

    st.write(f"**Average Step Size:** {avg_step_size:.3f}")
    st.write(f"**Step Size Std Dev:** {std_step_size:.3f}")

    # Pitch class coverage
    coverage = len(np.unique(np.round(pitch_classes))) / 12 * 100
    st.write(f"**Pitch Class Coverage:** {coverage:.1f}%")

    # Most common intervals
    if intervals_used:
        most_common = np.argmax(np.bincount(intervals_used))
        interval_names = ['Unison', 'Minor 2nd', 'Major 2nd', 'Minor 3rd', 'Major 3rd', 'Perfect 4th',
                         'Tritone', 'Perfect 5th', 'Minor 6th', 'Major 6th', 'Minor 7th', 'Major 7th']
        st.write(f"**Most Common Interval:** {interval_names[most_common]}")

    # Modulation analysis
    st.subheader("Modulation Analysis")

    # Calculate modulation strength
    modulation_strength = modulation_rate * np.max(np.abs(np.sin(2 * np.pi * np.arange(n_steps) / n_steps)))
    st.write(f"**Modulation Strength:** {modulation_strength:.3f}")

    # Check for periodicity
    # Simple autocorrelation
    if len(pitch_classes) > 10:
        autocorr = np.correlate(pitch_classes - np.mean(pitch_classes),
                               pitch_classes - np.mean(pitch_classes), mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        peaks, _ = plt.find_peaks(autocorr, height=np.max(autocorr)*0.5)
        if len(peaks) > 0:
            periodicity = peaks[0]
            st.write(f"**Detected Periodicity:** {periodicity} steps")
        else:
            st.write("**Periodicity:** No strong periodic pattern detected")

    # Torus topology analysis
    st.subheader("Torus Topology")

    # Calculate winding numbers
    theta_windings = int(np.round((theta[-1] - theta[0]) / (2 * np.pi)))
    phi_windings = int(np.round((phi[-1] - phi[0]) / (2 * np.pi)))

    st.write(f"**Longitudinal Windings:** {theta_windings}")
    st.write(f"**Latitudinal Windings:** {phi_windings}")

    # Check if trajectory closes on itself
    start_to_end_distance = np.sqrt((X[-1] - X[0])**2 + (Y[-1] - Y[0])**2 + (Z[-1] - Z[0])**2)
    closes_on_itself = start_to_end_distance < torus_minor * 0.1

    if closes_on_itself:
        st.success("✅ **Closed Trajectory** - Returns to starting point")
    else:
        st.info("🔄 **Open Trajectory** - Does not return to starting point")

    st.markdown(f"""
    **Tonal Torus Trajectory Theory:**
    - **T² Manifold**: Two-dimensional torus representing pitch space
    - **Scalar Step Operators**: T_k transformations mapping θ → θ + kΔ
    - **Interval-Based Motion**: Musical intervals as geodesic paths on torus
    - **Three-Lobed Paths**: Complex trajectories from interval combinations

    **Musical Interpretations:**
    - **Pitch Class Space**: 12 pitch classes arranged on circle
    - **Octave Equivalence**: Identification of pitches separated by octaves
    - **Interval Sequences**: Musical motion through pitch space
    - **Modulation**: Smooth transitions between tonal centers

    **Mathematical Properties:**
    - **Topological Invariants**: Winding numbers characterize trajectory type
    - **Geodesic Paths**: Shortest paths corresponding to consonant intervals
    - **Group Structure**: Pitch class group acting on torus
    - **Ergodicity**: Trajectory coverage of available pitch space

    **Applications:**
    - **Music Theory**: Analysis of tonal motion and harmonic progressions
    - **Composition**: Algorithmic generation of musical sequences
    - **Music Psychology**: Perception of tonal hierarchies and expectations
    - **Computational Musicology**: Modeling of musical style and structure
    """)

    # Interactive exploration
    st.subheader("Interactive Exploration")
    st.markdown("Adjust the interval types and step parameters to explore different musical trajectories on the tonal torus.")
