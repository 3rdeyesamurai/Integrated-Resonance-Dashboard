import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def interference_field_mapper():
    st.header("Interference Field Mapper")
    st.markdown("Simulate interference fields I(x, t) = Σ A_n cos(k_n x - ω_n t + φ_n) and generate heatmaps of nodes and antinodes.")

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

    # Generate interference field
    I = np.zeros_like(x)
    for n in range(1, n_sources + 1):
        A_n = A_base / n  # Decreasing amplitude
        k_n = k * n
        omega_n = omega * n
        phi_n = np.pi * n / n_sources  # Phase offset
        I += A_n * np.cos(k_n * x - omega_n * time + phi_n)

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
