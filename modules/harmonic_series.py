import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def harmonic_series_generator():
    st.header("Harmonic Series Generator")
    st.markdown("Generate and visualize the harmonic series for a given fundamental frequency.")

    # Input parameters
    f0 = st.slider("Fundamental Frequency (Hz)", 20, 1000, 440, step=10)
    n_harmonics = st.slider("Number of Harmonics", 1, 20, 10)
    duration = st.slider("Duration (seconds)", 0.1, 5.0, 1.0, step=0.1)

    # Generate harmonic frequencies
    harmonics = np.arange(1, n_harmonics + 1) * f0
    st.write(f"Harmonic Frequencies: {harmonics}")

    # Generate time array
    t = np.linspace(0, duration, int(44100 * duration))

    # Generate waveforms
    waveforms = []
    for n in range(1, n_harmonics + 1):
        wave = np.sin(2 * np.pi * n * f0 * t)
        waveforms.append(wave)

    # Plot individual harmonics
    fig, axes = plt.subplots(n_harmonics, 1, figsize=(10, 2*n_harmonics))
    for i, (wave, freq) in enumerate(zip(waveforms, harmonics)):
        axes[i].plot(t[:1000], wave[:1000])  # Plot first 1000 samples
        axes[i].set_title(f"Harmonic {i+1}: {freq:.1f} Hz")
        axes[i].set_xlabel("Time (s)")
        axes[i].set_ylabel("Amplitude")
    plt.tight_layout()
    st.pyplot(fig)

    # Plot sum of harmonics
    total_wave = np.sum(waveforms, axis=0)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(t[:1000], total_wave[:1000])
    ax2.set_title("Sum of Harmonics")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    st.pyplot(fig2)

    # Highlight integer ratios
    st.subheader("Integer Ratio Interactions")
    ratios = []
    for i in range(len(harmonics)):
        for j in range(i+1, len(harmonics)):
            ratio = harmonics[j] / harmonics[i]
            if ratio == int(ratio):
                ratios.append(f"{i+1}:{j+1} = {int(ratio)}:1")
    if ratios:
        st.write("Perfect integer ratios found:")
        for ratio in ratios:
            st.write(ratio)
    else:
        st.write("No perfect integer ratios in the selected harmonics.")
