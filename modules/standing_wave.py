import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

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

def standing_wave_simulator():
    st.header("🌊 RVM-Enhanced Standing Wave Simulator")
    st.markdown("""
    Simulate standing waves with RVM nodal/antinodal mapping.
    Nodes map to RVM doubling circuit (1-2-4-8-7-5), antinodes align with 3-6-9 control axis.
    """)

    # RVM Standing Wave Integration
    with st.expander("🔢 RVM Standing Wave Foundations"):
        st.markdown("""
        **Nodal Mapping**: Points of zero displacement map to RVM doubling circuit (1-2-4-8-7-5)
        **Antinodal Mapping**: Points of maximum displacement align with 3-6-9 control axis
        **Resonant Memory Zones**: Nodes as energy storage points
        **Tension Zones**: Antinodes as energy oscillation points
        **Vortex Dynamics**: Standing wave patterns following RVM flow
        """)

        doubling_seq = vortex_doubling_sequence()
        st.write(f"**RVM Doubling Sequence**: {doubling_seq}")
        st.write("**3-6-9 Control Axis**: Energy flow control mechanism")

    # Input parameters
    A = st.slider("Amplitude", 0.1, 2.0, 1.0, step=0.1)
    wavelength = st.slider("Wavelength", 0.1, 5.0, 1.0, step=0.1)
    frequency = st.slider("Frequency (Hz)", 0.1, 5.0, 1.0, step=0.1)
    length = st.slider("String Length", 1.0, 10.0, 5.0, step=0.5)

    # Calculate wave parameters
    k = 2 * np.pi / wavelength
    omega = 2 * np.pi * frequency

    # Spatial and temporal arrays
    x = np.linspace(0, length, 1000)
    t = np.linspace(0, 2/frequency, 100)  # Two periods

    # Standing wave function
    def psi(x, t):
        return 2 * A * np.cos(k * x) * np.sin(omega * t)

    # Create animation
    fig, ax = plt.subplots(figsize=(10, 6))
    line, = ax.plot(x, psi(x, 0))
    ax.set_xlim(0, length)
    ax.set_ylim(-2*A, 2*A)
    ax.set_xlabel("Position (x)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Standing Wave Animation")

    # Mark nodal points (where cos(kx) = 0)
    nodal_x = np.arange(0, length + wavelength/4, wavelength/2)
    for nx in nodal_x:
        if nx <= length:
            ax.axvline(x=nx, color='red', linestyle='--', alpha=0.7, label='Nodal Points (Resonant Memory Zones)' if nx == nodal_x[0] else "")

    # Mark antinodal points (where cos(kx) = ±1)
    antinode_x = np.arange(wavelength/4, length, wavelength/2)
    for ax_pos in antinode_x:
        if ax_pos <= length:
            ax.axvline(x=ax_pos, color='green', linestyle='--', alpha=0.7, label='Antinodal Points (Tension Zones)' if ax_pos == antinode_x[0] else "")

    ax.legend()

    def animate(frame):
        y = psi(x, t[frame])
        line.set_ydata(y)
        return line,

    ani = FuncAnimation(fig, animate, frames=len(t), interval=50, blit=True)

    # Display animation
    st.pyplot(fig)

    # RVM Nodal/Antinodal Mapping
    st.subheader("🌀 RVM Nodal/Antinodal Vortex Mapping")
    fig_rvm, ax_rvm = plt.subplots(figsize=(12, 6))

    # Plot standing wave
    ax_rvm.plot(x, psi(x, 0), 'b-', linewidth=2, label='Standing Wave')

    # RVM nodal points mapping (doubling circuit: 1,2,4,8,7,5)
    doubling_seq = vortex_doubling_sequence()
    nodal_colors = ['red', 'blue', 'blue', 'blue', 'blue', 'blue']  # Red for start/end, blue for others

    for i, (nx, rvm_val, color) in enumerate(zip(nodal_x, doubling_seq, nodal_colors)):
        if nx <= length:
            ax_rvm.axvline(x=nx, color=color, linestyle='--', linewidth=2, alpha=0.8)
            ax_rvm.scatter(nx, 0, s=100, c=color, edgecolors='black', zorder=5)
            ax_rvm.text(nx, -2.2*A, f'RVM:{rvm_val}', ha='center', va='top', fontsize=10, fontweight='bold')

    # RVM antinodal points mapping (3-6-9 triad)
    triad_values = [3, 6, 9]
    triad_colors = ['red', 'red', 'red']  # All red for 3-6-9 triad

    for i, (ax_pos, triad_val, color) in enumerate(zip(antinode_x, triad_values * (len(antinode_x)//3 + 1), triad_colors)):
        if ax_pos <= length and i < len(triad_values):
            ax_rvm.axvline(x=ax_pos, color=color, linestyle='-', linewidth=2, alpha=0.8)
            ax_rvm.scatter(ax_pos, 2*A if i % 2 == 0 else -2*A, s=100, c=color, marker='^', edgecolors='black', zorder=5)
            ax_rvm.text(ax_pos, 2.2*A if i % 2 == 0 else -2.2*A, f'3-6-9:{triad_val}', ha='center', va='bottom' if i % 2 == 0 else 'top', fontsize=10, fontweight='bold')

    ax_rvm.set_xlim(0, length)
    ax_rvm.set_ylim(-2.5*A, 2.5*A)
    ax_rvm.set_xlabel("Position (x)")
    ax_rvm.set_ylabel("Amplitude")
    ax_rvm.set_title("RVM Vortex Mapping on Standing Wave")
    ax_rvm.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='3-6-9 Control Axis (Antinodes)'),
        Patch(facecolor='blue', edgecolor='black', label='Doubling Circuit (Nodes)'),
        plt.Line2D([0], [0], color='blue', linewidth=2, label='Standing Wave')
    ]
    ax_rvm.legend(handles=legend_elements, loc='upper right')

    st.pyplot(fig_rvm)

    # Static plot at t=0
    st.subheader("Static View (t=0)")
    fig_static, ax_static = plt.subplots(figsize=(10, 4))
    ax_static.plot(x, psi(x, 0))
    ax_static.set_xlim(0, length)
    ax_static.set_ylim(-2*A, 2*A)
    ax_static.set_xlabel("Position (x)")
    ax_static.set_ylabel("Amplitude")
    ax_static.set_title("Standing Wave at t=0")
    st.pyplot(fig_static)

    # RVM Wave Parameter Analysis
    st.subheader("🔢 RVM Digital Root Analysis of Wave Parameters")
    wave_params = {
        "Amplitude": A,
        "Wavelength": wavelength,
        "Frequency": frequency,
        "Length": length,
        "Wave Number (k)": k,
        "Angular Frequency (ω)": omega
    }

    param_digital_roots = {param: digital_root(int(val * 100) if val < 1 else int(val))
                          for param, val in wave_params.items()}

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Wave Parameters:**")
        for param, val in wave_params.items():
            st.write(f"{param}: {val:.3f}")

    with col2:
        st.write("**Digital Roots:**")
        for param, dr in param_digital_roots.items():
            st.write(f"{param}: {dr}")

    # Explanation
    st.markdown("""
    **Nodal Points (Red lines):** Points of zero displacement, representing "resonant memory zones" where energy is stored.
    **Antinodal Points (Green lines):** Points of maximum displacement, representing "tension zones" where energy oscillates.
    """)
