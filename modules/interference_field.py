import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

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

def interference_field_mapper():
    st.header("🌊 RVM-Enhanced Interference Field Mapper")
    st.markdown("""
    Simulate interference fields with RVM cyclic patterns and mod-9 periodicity.
    Features RVM doubling circuit amplitudes and vortex interference dynamics.
    """)

    # RVM Interference Field Integration
    with st.expander("🔢 RVM Interference Field Foundations"):
        st.markdown("""
        **Cyclic Patterns**: Interference patterns following RVM mod-9 periodicity
        **Doubling Circuit Amplitudes**: A_n based on 1→2→4→8→7→5→1 sequence
        **Vortex Interference**: Nodes/antinodes mapped to RVM 9-point circle
        **Phase Shifts**: Tied to 3-6-9 control axis (φ_n = π * n / 3)
        **Resonance Index**: R = 2|B1B2|/(|B1|² + |B2|²) peaks at B2/B1 = 3/6/9
        """)

        doubling_seq = vortex_doubling_sequence()
        st.write(f"**RVM Doubling Sequence**: {doubling_seq}")
        st.write("**3-6-9 Control Axis**: Energy flow control mechanism")

    # Input parameters
    n_sources = st.slider("Number of Sources", 2, 5, 3)
    A_base = st.slider("Base Amplitude", 0.1, 2.0, 1.0, step=0.1)
    wavelength = st.slider("Wavelength", 0.1, 2.0, 1.0, step=0.1)
    frequency = st.slider("Frequency (Hz)", 0.1, 2.0, 1.0, step=0.1)
    length = st.slider("Field Length", 5.0, 20.0, 10.0, step=1.0)
    time = st.slider("Time", 0.0, 10.0, 0.0, step=0.1)

    # Calculate wave parameters
    k = 2 * np.pi / wavelength
    omega = 2 * np.pi * frequency

    # Spatial grid
    x = np.linspace(0, length, 200)

    # RVM Doubling Circuit Amplitudes
    doubling_seq = vortex_doubling_sequence()
    # Extend sequence to match number of sources
    extended_doubling = doubling_seq * (n_sources // len(doubling_seq)) + doubling_seq[:n_sources % len(doubling_seq)]
    rvm_amplitudes = np.array(extended_doubling[:n_sources]) / 9.0  # Normalize to 0-1 range

    # Generate interference field with RVM amplitudes
    I = np.zeros_like(x)
    I_rvm = np.zeros_like(x)  # RVM-modulated interference

    for n in range(1, n_sources + 1):
        # Standard amplitudes
        A_n = A_base / n
        # RVM amplitudes from doubling circuit
        A_rvm = A_base * rvm_amplitudes[n-1]

        k_n = k * n
        omega_n = omega * n
        phi_n = np.pi * n / 3  # RVM phase shift tied to 3-6-9

        I += A_n * np.cos(k_n * x - omega_n * time + phi_n)
        I_rvm += A_rvm * np.cos(k_n * x - omega_n * time + phi_n)

    # Create heatmap data (intensity over space and time)
    t_range = np.linspace(0, 2/frequency, 50)
    X, T = np.meshgrid(x, t_range)
    Intensity = np.zeros_like(X)

    for i, t_val in enumerate(t_range):
        for n in range(1, n_sources + 1):
            A_n = A_base / n
            k_n = k * n
            omega_n = omega * n
            phi_n = np.pi * n / n_sources
            Intensity[i, :] += A_n * np.cos(k_n * x - omega_n * t_val + phi_n)

    # Plot interference pattern at current time
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Wave pattern
    ax1.plot(x, I, 'b-', linewidth=2)
    ax1.set_xlabel('Position (x)')
    ax1.set_ylabel('Intensity')
    ax1.set_title(f'Interference Pattern at t = {time:.1f}s')
    ax1.grid(True)

    # Mark nodes and antinodes
    nodes = []
    antinodes = []
    for i in range(len(x)-1):
        if I[i] * I[i+1] < 0:  # Zero crossing (node)
            nodes.append(x[i])
        elif abs(I[i]) > 0.8 * np.max(np.abs(I)):  # High amplitude (antinode)
            antinodes.append(x[i])

    for node in nodes[:10]:  # Limit markers
        ax1.axvline(x=node, color='red', linestyle='--', alpha=0.7)
    for antinode in antinodes[:10]:
        ax1.axvline(x=antinode, color='green', linestyle='--', alpha=0.7)

    # Heatmap
    im = ax2.imshow(Intensity, extent=[0, length, 0, 2/frequency],
                    aspect='auto', cmap='RdYlBu_r', origin='lower')
    ax2.set_xlabel('Position (x)')
    ax2.set_ylabel('Time (s)')
    ax2.set_title('Interference Field Heatmap')
    plt.colorbar(im, ax=ax2, label='Intensity')

    st.pyplot(fig1)

    # RVM vs Standard Interference Comparison
    st.subheader("🌀 RVM vs Standard Interference Patterns")
    fig_rvm, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Standard interference
    ax1.plot(x, I, 'b-', linewidth=2, label='Standard')
    ax1.set_title('Standard Interference Pattern')
    ax1.set_xlabel('Position (x)')
    ax1.set_ylabel('Intensity')
    ax1.grid(True)

    # RVM-modulated interference
    ax2.plot(x, I_rvm, 'r-', linewidth=2, label='RVM-Modulated')
    ax2.set_title('RVM Doubling Circuit Modulated Interference')
    ax2.set_xlabel('Position (x)')
    ax2.set_ylabel('Intensity')
    ax2.grid(True)

    # Mark RVM nodes/antinodes
    rvm_nodes = []
    rvm_antinodes = []
    for i in range(len(x)-1):
        if I_rvm[i] * I_rvm[i+1] < 0:  # Zero crossing (node)
            rvm_nodes.append(x[i])
        elif abs(I_rvm[i]) > 0.8 * np.max(np.abs(I_rvm)):  # High amplitude (antinode)
            rvm_antinodes.append(x[i])

    for node in rvm_nodes[:10]:
        ax2.axvline(x=node, color='purple', linestyle='--', alpha=0.7)
    for antinode in rvm_antinodes[:10]:
        ax2.axvline(x=antinode, color='orange', linestyle='--', alpha=0.7)

    st.pyplot(fig_rvm)

    # Vortex Interference Pattern
    st.subheader("🌀 RVM Vortex Interference Mapping")
    fig_vortex, ax_vortex = plt.subplots(figsize=(8, 8))

    # Create RVM 9-point circle for interference mapping
    rvm_points = [1, 2, 4, 8, 7, 5, 3, 6, 9]
    angles = np.linspace(0, 2*np.pi, 9, endpoint=False)

    # Map interference nodes/antinodes to vortex circle
    for i, (point, angle) in enumerate(zip(rvm_points, angles)):
        x_point = 3 * np.cos(angle)
        y_point = 3 * np.sin(angle)

        # Color based on 3-6-9 triad
        if point in [3, 6, 9]:
            color = 'red'
        else:
            color = 'blue'

        ax_vortex.scatter(x_point, y_point, s=200, c=color, alpha=0.8, edgecolors='black')
        ax_vortex.text(x_point, y_point, str(point), ha='center', va='center', fontsize=12, fontweight='bold')

    # Draw circle
    circle = plt.Circle((0, 0), 3, fill=False, color='gray', linestyle='--', alpha=0.5)
    ax_vortex.add_artist(circle)

    # Plot interference pattern as radial projection
    theta_int = np.linspace(0, 2*np.pi, len(x))
    r_int = 1 + I_rvm / np.max(np.abs(I_rvm))  # Normalize to 0-2 range
    x_int = r_int * np.cos(theta_int)
    y_int = r_int * np.sin(theta_int)

    ax_vortex.plot(x_int, y_int, 'purple', linewidth=2, alpha=0.7, label='Interference Pattern')

    ax_vortex.set_xlim(-4, 4)
    ax_vortex.set_ylim(-4, 4)
    ax_vortex.set_aspect('equal')
    ax_vortex.set_title('RVM Vortex Interference Pattern')
    ax_vortex.grid(True, alpha=0.3)
    ax_vortex.legend()

    st.pyplot(fig_vortex)

    # RVM Digital Root Analysis
    st.subheader("🔢 RVM Digital Root Analysis of Interference Parameters")
    int_params = {
        "Number of Sources": n_sources,
        "Wavelength": wavelength,
        "Frequency": frequency,
        "Field Length": length,
        "Wave Number (k)": k,
        "Angular Frequency (ω)": omega
    }

    param_digital_roots = {param: digital_root(int(val * 100) if val < 1 else int(val))
                          for param, val in int_params.items()}

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Interference Parameters:**")
        for param, val in int_params.items():
            st.write(f"{param}: {val:.3f}")

    with col2:
        st.write("**Digital Roots:**")
        for param, dr in param_digital_roots.items():
            st.write(f"{param}: {dr}")

    # Statistics
    st.subheader("Field Statistics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Max Intensity", f"{np.max(I):.3f}")
    with col2:
        st.metric("Min Intensity", f"{np.min(I):.3f}")
    with col3:
        st.metric("RMS Intensity", f"{np.sqrt(np.mean(I**2)):.3f}")

    st.markdown("""
    **Interference Field Analysis:**
    - **Nodes (Red lines):** Points of destructive interference, representing "gravity wells"
    - **Antinodes (Green lines):** Points of constructive interference, representing "tension zones"
    - The heatmap shows how the interference pattern evolves over time
    """)
