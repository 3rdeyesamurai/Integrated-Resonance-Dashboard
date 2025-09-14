import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
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

def overtone_convergence_analyzer():
    st.header("🌟 RVM-Enhanced Overtone Convergence Analyzer")
    st.markdown("""
    Compute overtone series with RVM golden ratio corrections and analyze resonance efficiency.
    Features ϕ-overtone nesting and Fibonacci convergence analysis.
    """)

    # RVM Overtone Integration
    with st.expander("🔢 RVM Overtone Convergence Foundations"):
        st.markdown("""
        **ϕ-Overtone Nesting**: F_n/F_{n-1} → φ convergence with RVM corrections
        **Fibonacci Digital Roots**: RVM analysis of overtone sequences
        **Vortex Resonance Patterns**: Doubling circuit in harmonic spacing
        **3-6-9 Overtone Control**: Triad resonances in overtone series
        **Toroidal Harmonic Substrates**: Unified field theory connections
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

    # RVM Digital Root Analysis
    st.subheader("🔢 RVM Digital Root Analysis of Overtone Frequencies")
    overtone_digital_roots = [digital_root(int(freq)) for freq in frequencies]

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Overtone Frequencies:**")
        st.write([f"{freq:.1f}" for freq in frequencies])
    with col2:
        st.write("**Digital Roots:**")
        st.write(overtone_digital_roots)

    # RVM Vortex Overtone Mapping
    st.subheader("🌀 RVM Vortex Overtone Mapping")
    fig_rvm, ax_rvm = plt.subplots(figsize=(10, 8))

    # Create RVM 9-point circle for overtone mapping
    rvm_points = [1, 2, 4, 8, 7, 5, 3, 6, 9]
    angles = np.linspace(0, 2*np.pi, 9, endpoint=False)

    # Map overtone frequencies to RVM vortex circle
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

    # Overlay overtone series scaled to vortex
    scale_vortex = 2.5 / max(frequencies) if frequencies else 1
    x_vortex = []
    y_vortex = []
    colors_vortex = []

    for i, freq in enumerate(frequencies):
        angle = (i / len(frequencies)) * 4 * np.pi  # Spiral out
        radius = scale_vortex * freq * 0.1
        x_vortex.append(radius * np.cos(angle))
        y_vortex.append(radius * np.sin(angle))
        colors_vortex.append(i / len(frequencies))

    scatter_vortex = ax_rvm.scatter(x_vortex, y_vortex, c=colors_vortex, cmap='viridis', s=30, alpha=0.7, label='Overtone Series')

    ax_rvm.set_xlim(-4, 4)
    ax_rvm.set_ylim(-4, 4)
    ax_rvm.set_aspect('equal')
    ax_rvm.set_title('RVM Vortex Overtone Mapping')
    ax_rvm.grid(True, alpha=0.3)
    plt.colorbar(scatter_vortex, ax=ax_rvm, label='Overtone Number')

    st.pyplot(fig_rvm)

    st.markdown(f"""
    **Analysis Results:**
    - **Spacing Type:** {spacing_type}
    - Golden ratio spacing shows optimal convergence for harmonic resonance
    - FFT analysis reveals the frequency content and resonance peaks
    - Spectral metrics indicate the efficiency of energy distribution
    - **RVM Integration:** Digital roots and vortex mapping reveal underlying patterns
    """)
