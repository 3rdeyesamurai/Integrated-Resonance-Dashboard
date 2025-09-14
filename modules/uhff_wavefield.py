import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sympy import sqrt, Rational

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

def uhff_wavefield_summator():
    st.header("🌊 RVM-Enhanced UHFF Wavefield Summator")
    st.markdown("""
    Build multi-oscillator summation with RVM ϕ-overtone nesting.
    Features Fourier analysis with golden ratio convergence and vortex field mapping.
    """)

    # RVM Wavefield Integration
    with st.expander("🔢 RVM Wavefield Summation Foundations"):
        st.markdown("""
        **ϕ-Overtone Nesting**: U(x,t) = Σ A_j sin(k_j x - ω_j t + φ_j) with golden ratio scaling
        **Fourier Convergence**: F_n/F_{n-1} → φ with RVM corrections
        **Vortex Field Patterns**: Standing wave lattices mapped to RVM 9-point circle
        **3-6-9 Resonance Nodes**: Triad control in standing wave formation
        **Toroidal Harmonic Substrates**: Unified field theory wave interactions
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
    n_oscillators = st.slider("Number of Oscillators", 2, 8, 5)
    base_amplitude = st.slider("Base Amplitude", 0.1, 2.0, 1.0, step=0.1)
    base_wavelength = st.slider("Base Wavelength", 0.1, 2.0, 1.0, step=0.1)
    base_frequency = st.slider("Base Frequency", 0.1, 2.0, 1.0, step=0.1)
    time = st.slider("Time t", 0.0, 10.0, 0.0, step=0.1)
    spatial_range = st.slider("Spatial Range", 5.0, 20.0, 10.0, step=1.0)

    # Generate oscillator parameters
    oscillators = []

    for j in range(1, n_oscillators + 1):
        # Use Fibonacci ratios for frequencies and wavelengths
        fib_ratios = [1, 1, 2, 3, 5, 8, 13, 21]
        ratio = fib_ratios[min(j, len(fib_ratios)-1)]

        # Amplitude with golden ratio scaling
        A_j = base_amplitude * (phi_val ** (-j/2))

        # Wavelength and frequency with rational relationships
        k_j = 2 * np.pi / (base_wavelength * ratio)
        omega_j = 2 * np.pi * (base_frequency * ratio)

        # Phase offset
        phi_j = np.pi * j / n_oscillators  # Evenly distributed phases

        oscillators.append({
            'A': A_j,
            'k': k_j,
            'omega': omega_j,
            'phi': phi_j,
            'ratio': ratio
        })

    # Create spatial grid
    x = np.linspace(0, spatial_range, 1000)

    # Calculate total wavefield
    U_total = np.zeros_like(x)

    # Store individual wave components
    wave_components = []

    for osc in oscillators:
        U_j = osc['A'] * np.sin(osc['k'] * x - osc['omega'] * time + osc['phi'])
        U_total += U_j
        wave_components.append(U_j)

    # Detect standing wave patterns (nodes and antinodes)
    # Find local maxima (antinodes) and minima (nodes)
    peaks, _ = find_peaks(U_total, height=0.1, distance=20)
    troughs, _ = find_peaks(-U_total, height=0.1, distance=20)

    # Calculate "mass" density (related to standing wave energy)
    mass_density = U_total ** 2  # Energy density proportional to amplitude squared

    # Detect rational frequency ratios
    rational_ratios = []
    for i in range(len(oscillators)):
        for j in range(i+1, len(oscillators)):
            freq_ratio = oscillators[j]['omega'] / oscillators[i]['omega']
            # Check if ratio is close to a simple rational number
            for p in range(1, 5):
                for q in range(1, 5):
                    if abs(freq_ratio - p/q) < 0.05:
                        rational_ratios.append(f"{p}:{q} (≈{freq_ratio:.3f})")

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Total wavefield
    ax1.plot(x, U_total, 'b-', linewidth=2, label='Total Wavefield')
    ax1.scatter(x[peaks], U_total[peaks], color='red', s=50, label='Antinodes', zorder=5)
    ax1.scatter(x[troughs], U_total[troughs], color='green', s=50, label='Nodes', zorder=5)
    ax1.set_xlabel('Position x')
    ax1.set_ylabel('Amplitude U(x,t)')
    ax1.set_title('UHFF Wavefield Summation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Individual components
    colors = plt.cm.tab10(np.linspace(0, 1, n_oscillators))
    for i, (component, osc) in enumerate(zip(wave_components, oscillators)):
        ax2.plot(x, component, color=colors[i], alpha=0.7,
                label=f'Osc {i+1}: f={osc["omega"]/(2*np.pi):.2f}Hz')
    ax2.plot(x, U_total, 'k-', linewidth=2, label='Total')
    ax2.set_xlabel('Position x')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Individual Wave Components')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Mass density (standing wave energy)
    ax3.plot(x, mass_density, 'purple', linewidth=2)
    ax3.fill_between(x, mass_density, alpha=0.3, color='purple')
    ax3.scatter(x[peaks], mass_density[peaks], color='red', s=30, label='High Energy')
    ax3.scatter(x[troughs], mass_density[troughs], color='blue', s=30, label='Low Energy')
    ax3.set_xlabel('Position x')
    ax3.set_ylabel('Mass Density (Energy)')
    ax3.set_title('Standing Wave Mass Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Frequency spectrum analysis
    ax4.bar(range(len(oscillators)), [osc['omega']/(2*np.pi) for osc in oscillators],
            color='skyblue', alpha=0.7)
    ax4.set_xlabel('Oscillator Index')
    ax4.set_ylabel('Frequency (Hz)')
    ax4.set_title('Frequency Distribution')
    ax4.grid(True, alpha=0.3)

    # Add frequency ratio labels
    for i, osc in enumerate(oscillators):
        ax4.text(i, osc['omega']/(2*np.pi), f'{osc["ratio"]}:1',
                ha='center', va='bottom')

    plt.tight_layout()
    st.pyplot(fig)

    # Analysis metrics
    st.subheader("Wavefield Analysis")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Oscillators", n_oscillators)
    with col2:
        st.metric("Standing Nodes", len(troughs))
    with col3:
        st.metric("Standing Antinodes", len(peaks))
    with col4:
        st.metric("Golden Ratio φ", f"{phi_val:.6f}")

    # Detailed analysis
    st.subheader("Rational Ratio Analysis")

    if rational_ratios:
        st.write("**Detected Rational Frequency Ratios:**")
        for ratio in rational_ratios[:5]:  # Show first 5
            st.write(f"- {ratio}")
    else:
        st.write("No simple rational ratios detected in current configuration.")

    # Standing wave analysis
    st.subheader("Standing Wave Characteristics")

    # Calculate standing wave ratio
    if len(peaks) > 0 and len(troughs) > 0:
        standing_wave_ratio = len(peaks) / len(troughs)
        st.write(f"**Standing Wave Ratio:** {standing_wave_ratio:.2f}")
    else:
        st.write("**Standing Wave Ratio:** Undefined (insufficient nodes/antinodes)")

    # Energy analysis
    total_energy = np.sum(mass_density)
    avg_energy = np.mean(mass_density)
    energy_variance = np.var(mass_density)

    st.write(f"**Total Energy:** {total_energy:.2f}")
    st.write(f"**Average Energy Density:** {avg_energy:.4f}")
    st.write(f"**Energy Variance:** {energy_variance:.4f}")

    # Phase coherence
    phase_coherence = np.abs(np.mean(np.exp(1j * np.angle(U_total + 1j*np.zeros_like(U_total)))))
    st.write(f"**Phase Coherence:** {phase_coherence:.3f}")

    # Oscillator details
    st.subheader("Oscillator Configuration")

    osc_data = []
    for i, osc in enumerate(oscillators):
        osc_data.append({
            "Oscillator": f"#{i+1}",
            "Amplitude": f"{osc['A']:.3f}",
            "Frequency (Hz)": f"{osc['omega']/(2*np.pi):.2f}",
            "Wavelength": f"{2*np.pi/osc['k']:.2f}",
            "Phase (rad)": f"{osc['phi']:.2f}",
            "Ratio": f"{osc['ratio']}:1"
        })

    st.table(osc_data)

    # RVM Digital Root Analysis
    st.subheader("🔢 RVM Digital Root Analysis of Wavefield Parameters")
    wavefield_params = {
        "Number of Oscillators": n_oscillators,
        "Base Amplitude": int(base_amplitude * 100),
        "Base Frequency": int(base_frequency * 100),
        "Spatial Range": int(spatial_range * 10),
        "Standing Nodes": len(troughs),
        "Standing Antinodes": len(peaks),
        "Total Energy": int(total_energy * 10)
    }

    param_digital_roots = {param: digital_root(val) for param, val in wavefield_params.items()}

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Wavefield Parameters:**")
        for param, val in wavefield_params.items():
            st.write(f"{param}: {val}")

    with col2:
        st.write("**Digital Roots:**")
        for param, dr in param_digital_roots.items():
            st.write(f"{param}: {dr}")

    # RVM Vortex Wavefield Mapping
    st.subheader("🌀 RVM Vortex Wavefield Mapping")
    fig_rvm, ax_rvm = plt.subplots(figsize=(10, 8))

    # Create RVM 9-point circle for wavefield mapping
    rvm_points = [1, 2, 4, 8, 7, 5, 3, 6, 9]
    angles = np.linspace(0, 2*np.pi, 9, endpoint=False)

    # Map wavefield parameters to RVM vortex circle
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

    # Overlay standing wave pattern scaled to vortex
    scale_vortex = 2.5 / np.max(U_total) if np.max(U_total) > 0 else 1
    x_vortex = []
    y_vortex = []
    colors_vortex = []

    for i in range(0, len(x), 10):  # Sample every 10th point
        angle = (x[i] / spatial_range) * 4 * np.pi  # Map position to angle
        radius = scale_vortex * abs(U_total[i])
        x_vortex.append(radius * np.cos(angle))
        y_vortex.append(radius * np.sin(angle))
        colors_vortex.append(U_total[i])

    scatter_vortex = ax_rvm.scatter(x_vortex, y_vortex, c=colors_vortex, cmap='RdYlBu', s=10, alpha=0.6, label='Wavefield Pattern')

    ax_rvm.set_xlim(-4, 4)
    ax_rvm.set_ylim(-4, 4)
    ax_rvm.set_aspect('equal')
    ax_rvm.set_title('RVM Vortex Wavefield Mapping')
    ax_rvm.grid(True, alpha=0.3)
    plt.colorbar(scatter_vortex, ax=ax_rvm, label='Wave Amplitude')

    st.pyplot(fig_rvm)

    st.markdown(f"""
    **UHFF Wavefield Summation Theory:**
    - **Multi-Oscillator Superposition**: U(x,t) = Σ Aⱼ sin(kⱼx - ωⱼt + φⱼ)
    - **ϕ-Overtone Nesting**: Golden ratio scaling with F_n/F_{n-1} → φ convergence
    - **Rational Frequency Ratios**: Simple integer relationships create coherent standing waves
    - **Mass as Standing Lattices**: Energy density U² represents "mass" distribution
    - **Phase Alignment**: Coherent phases create stable interference patterns

    **Physical Interpretations:**
    - **Quantum Systems**: Electron wave functions in multi-level atoms
    - **Acoustic Fields**: Sound wave interference in musical instruments
    - **Electromagnetic Waves**: Light field superpositions in interferometers
    - **Matter Waves**: Particle wave functions in crystalline structures

    **Key Phenomena:**
    - **Standing Wave Formation**: Constructive/destructive interference creates nodes and antinodes
    - **Energy Localization**: "Mass" concentrates at antinodes, creating lattice structures
    - **Frequency Locking**: Rational ratios lead to stable, periodic patterns
    - **Phase Synchronization**: Coherent phases maintain stable interference patterns
    - **RVM Integration**: Digital roots and vortex mapping reveal underlying harmonic patterns

    **Applications:**
    - **Crystallography**: Atomic lattice formation through wave interference
    - **Quantum Chemistry**: Molecular orbital formation from atomic wave functions
    - **Optics**: Holographic interference patterns with golden ratio scaling
    - **Seismology**: Earthquake wave interference in geological structures
    """)

    # Interactive exploration
    st.subheader("Interactive Exploration")
    st.markdown("Adjust the number of oscillators and their parameters to explore how rational frequency ratios and golden ratio scaling create stable standing wave lattices and energy concentrations with RVM harmonic patterns.")
