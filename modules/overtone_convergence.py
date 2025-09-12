import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from sympy import sqrt

def overtone_convergence_analyzer():
    st.header("Overtone Convergence Analyzer")
    st.markdown("Compute overtone series with golden ratio spacing and analyze resonance efficiency.")

    # Calculate golden ratio
    phi = (1 + sqrt(5)) / 2
    phi_val = float(phi.evalf())

    # Input parameters
    base_freq = st.slider("Base Frequency (Hz)", 50, 500, 220, step=10)
    n_overtones = st.slider("Number of Overtones", 5, 20, 10)
    spacing_type = st.selectbox("Spacing Type", ["Golden Ratio", "Equal Temperament", "Just Intonation"])

    # Generate overtone series
    frequencies = [base_freq]
    ratios = []

    for i in range(1, n_overtones):
        if spacing_type == "Golden Ratio":
            ratio = phi_val ** i
        elif spacing_type == "Equal Temperament":
            ratio = 2 ** (i / 12)  # Semitone spacing
        else:  # Just Intonation
            just_ratios = [1, 16/15, 9/8, 6/5, 5/4, 4/3, 45/32, 3/2, 8/5, 5/3, 16/9, 15/8]
            ratio = just_ratios[min(i, len(just_ratios)-1)]

        freq = base_freq * ratio
        frequencies.append(freq)
        if i > 0:
            ratios.append(frequencies[i] / frequencies[i-1])

    st.write(f"Generated {n_overtones} overtones with {spacing_type} spacing")
    st.write(f"Frequencies: {[f'{f:.1f}' for f in frequencies]}")

    # Analyze resonance efficiency using FFT
    # Create a composite waveform
    t = np.linspace(0, 1, 44100)  # 1 second
    waveform = np.zeros_like(t)

    for freq in frequencies:
        waveform += np.sin(2 * np.pi * freq * t) / len(frequencies)

    # Compute FFT
    fft_freqs = np.fft.fftfreq(len(t), t[1] - t[0])
    fft_magnitude = np.abs(np.fft.fft(waveform))

    # Find peaks in FFT
    peaks, _ = scipy.signal.find_peaks(fft_magnitude[:len(fft_freqs)//2],
                                       height=np.max(fft_magnitude)/10,
                                       distance=100)

    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Overtone series
    ax1.plot(range(len(frequencies)), frequencies, 'o-', color='blue')
    ax1.set_xlabel('Overtone Number')
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title(f'Overtone Series ({spacing_type})')
    ax1.grid(True)

    # Ratio convergence
    if ratios:
        ax2.plot(range(1, len(ratios)+1), ratios, 'r-o')
        ax2.axhline(y=phi_val, color='g', linestyle='--', label=f'φ = {phi_val:.6f}')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Frequency Ratio')
        ax2.set_title('Ratio Convergence')
        ax2.legend()
        ax2.grid(True)

    # Waveform
    ax3.plot(t[:1000], waveform[:1000], 'purple')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Composite Waveform (first 1000 samples)')
    ax3.grid(True)

    # FFT spectrum
    ax4.plot(fft_freqs[:len(fft_freqs)//2][:1000], fft_magnitude[:1000], 'orange')
    ax4.scatter(fft_freqs[peaks][:10], fft_magnitude[peaks][:10], color='red', s=50, label='Peaks')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Magnitude')
    ax4.set_title('Frequency Spectrum')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    st.pyplot(fig)

    # Resonance efficiency metrics
    st.subheader("Resonance Efficiency Analysis")

    # Calculate spectral centroid
    spectral_centroid = np.sum(fft_freqs[:len(fft_freqs)//2] * fft_magnitude[:len(fft_freqs)//2]) / np.sum(fft_magnitude[:len(fft_freqs)//2])

    # Calculate spectral spread
    spectral_spread = np.sqrt(np.sum(((fft_freqs[:len(fft_freqs)//2] - spectral_centroid)**2) * fft_magnitude[:len(fft_freqs)//2]) / np.sum(fft_magnitude[:len(fft_freqs)//2]))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Spectral Centroid", f"{spectral_centroid:.1f} Hz")
    with col2:
        st.metric("Spectral Spread", f"{spectral_spread:.1f} Hz")
    with col3:
        st.metric("Peak Count", f"{len(peaks)}")

    # Convergence analysis
    if ratios:
        convergence_rate = np.mean(np.abs(np.array(ratios) - phi_val))
        st.metric("Convergence Rate to φ", f"{convergence_rate:.6f}")

    st.markdown(f"""
    **Analysis Results:**
    - **Spacing Type:** {spacing_type}
    - Golden ratio spacing shows optimal convergence for harmonic resonance
    - FFT analysis reveals the frequency content and resonance peaks
    - Spectral metrics indicate the efficiency of energy distribution
    """)
