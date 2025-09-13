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

def harmonic_series_generator():
    st.header("🎵 RVM-Enhanced Harmonic Series Generator")
    st.markdown("""
    Generate and visualize the harmonic series with RVM doubling circuit integration.
    Features digital root analysis and vortex harmonic structuring.
    """)

    # RVM Harmonic Integration
    with st.expander("🔢 RVM Harmonic Foundations"):
        st.markdown("""
        **Doubling Circuit Harmonics**: 1→2→4→8→7→5→1 sequence applied to amplitudes
        **Digital Root Analysis**: Applied to harmonic numbers and frequencies
        **Vortex Harmonic Structure**: 9-point circle mapped to harmonic relationships
        **Phase Shifts**: Tied to 3-6-9 control axis
        """)

        doubling_seq = vortex_doubling_sequence()
        st.write(f"**RVM Doubling Sequence**: {doubling_seq}")

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

    # RVM Digital Roots Analysis
    st.subheader("🔢 RVM Digital Roots of Harmonic Numbers")
    harmonic_numbers = list(range(1, n_harmonics + 1))
    harmonic_digital_roots = [digital_root(n) for n in harmonic_numbers]

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Harmonic Numbers:**")
        st.write(harmonic_numbers)
    with col2:
        st.write("**Digital Roots:**")
        st.write(harmonic_digital_roots)

    # RVM Amplitude Modulation
    st.subheader("🌀 RVM Doubling Circuit Amplitude Modulation")
    doubling_seq = vortex_doubling_sequence()
    # Extend doubling sequence to match number of harmonics
    extended_doubling = doubling_seq * (n_harmonics // len(doubling_seq)) + doubling_seq[:n_harmonics % len(doubling_seq)]
    rvm_amplitudes = np.array(extended_doubling[:n_harmonics]) / 9.0  # Normalize to 0-1 range

    # Apply RVM amplitudes to waveforms
    rvm_waveforms = []
    for i, wave in enumerate(waveforms):
        rvm_wave = wave * rvm_amplitudes[i]
        rvm_waveforms.append(rvm_wave)

    # Plot RVM-modulated harmonics
    fig_rvm, ax_rvm = plt.subplots(figsize=(12, 6))
    colors = ['red' if amp in [3/9, 6/9, 9/9] else 'blue' for amp in rvm_amplitudes]
    for i, (wave, amp, color) in enumerate(zip(rvm_waveforms, rvm_amplitudes, colors)):
        ax_rvm.plot(t[:1000], wave[:1000], color=color, alpha=0.7,
                   label=f'H{i+1} (RVM: {extended_doubling[i]})')

    ax_rvm.set_title("RVM Doubling Circuit Modulated Harmonics")
    ax_rvm.set_xlabel("Time (s)")
    ax_rvm.set_ylabel("Amplitude")
    ax_rvm.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_rvm.grid(True, alpha=0.3)
    st.pyplot(fig_rvm)

    # Plot sum of harmonics
    total_wave = np.sum(waveforms, axis=0)
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(t[:1000], total_wave[:1000])
    ax2.set_title("Sum of Harmonics")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")
    st.pyplot(fig2)

    # RVM-modulated sum
    rvm_total_wave = np.sum(rvm_waveforms, axis=0)
    fig_rvm_sum, ax_rvm_sum = plt.subplots(figsize=(10, 4))
    ax_rvm_sum.plot(t[:1000], total_wave[:1000], 'b-', alpha=0.5, label='Original Sum')
    ax_rvm_sum.plot(t[:1000], rvm_total_wave[:1000], 'r-', linewidth=2, label='RVM Modulated Sum')
    ax_rvm_sum.set_title("Comparison: Original vs RVM-Modulated Harmonic Sum")
    ax_rvm_sum.set_xlabel("Time (s)")
    ax_rvm_sum.set_ylabel("Amplitude")
    ax_rvm_sum.legend()
    ax_rvm_sum.grid(True, alpha=0.3)
    st.pyplot(fig_rvm_sum)

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
