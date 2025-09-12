import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

def standing_wave_simulator():
    st.header("Standing Wave Simulator")
    st.markdown("Simulate standing waves from superposed traveling waves, highlighting nodal and antinodal points as resonant memory zones.")

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

    # Explanation
    st.markdown("""
    **Nodal Points (Red lines):** Points of zero displacement, representing "resonant memory zones" where energy is stored.
    **Antinodal Points (Green lines):** Points of maximum displacement, representing "tension zones" where energy oscillates.
    """)
