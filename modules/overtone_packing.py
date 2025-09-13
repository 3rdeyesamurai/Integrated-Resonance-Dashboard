import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpMaximize, LpStatus
from sympy import sqrt

def overtone_packing_efficiency_optimizer():
    st.header("Overtone Packing Efficiency Optimizer")
    st.markdown("Use PuLP for linear programming to optimize harmonic packing with φ ratios, minimizing phase conflicts in musical intervals or atomic orbitals.")

    # Calculate golden ratio
    phi = (1 + sqrt(5)) / 2
    phi_val = float(phi.evalf())

    # Input parameters
    n_overtones = st.slider("Number of Overtones", 5, 15, 8)
    optimization_type = st.selectbox("Optimization Type", ["Minimize Phase Conflicts", "Maximize Resonance", "Balance Efficiency"])
    base_frequency = st.slider("Base Frequency (Hz)", 100, 1000, 440, step=10)
    packing_constraint = st.slider("Packing Constraint", 0.1, 2.0, 0.8, step=0.1)

    # Generate overtone candidates
    candidates = []
    for i in range(1, n_overtones + 1):
        # Include harmonic, phi-scaled, and Fibonacci-based frequencies
        harmonic_freq = base_frequency * i
        phi_freq = base_frequency * (phi_val ** i)
        fib_ratios = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]
        fib_freq = base_frequency * fib_ratios[min(i, len(fib_ratios)-1)]

        candidates.extend([harmonic_freq, phi_freq, fib_freq])

    # Remove duplicates and sort
    candidates = sorted(list(set(candidates)))

    # Linear programming optimization
    if optimization_type == "Minimize Phase Conflicts":
        # Minimize phase differences between selected frequencies
        prob = LpProblem("Overtone_Packing_Min_Conflicts", LpMinimize)

        # Binary variables for frequency selection
        freq_vars = LpVariable.dicts("freq", range(len(candidates)), 0, 1, cat='Binary')

        # Constraint: select exactly n_overtones frequencies
        prob += lpSum(freq_vars[i] for i in range(len(candidates))) == n_overtones

        # Objective: minimize sum of phase differences
        phase_conflicts = []
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                freq_i = candidates[i]
                freq_j = candidates[j]
                # Phase difference metric
                phase_diff = abs(np.sin(2 * np.pi * freq_i) - np.sin(2 * np.pi * freq_j))
                # Use binary variable multiplication properly
                phase_conflicts.append(phase_diff * (freq_vars[i] + freq_vars[j] - 1))

        prob += lpSum(phase_conflicts)

    elif optimization_type == "Maximize Resonance":
        # Maximize resonance efficiency
        prob = LpProblem("Overtone_Packing_Max_Resonance", LpMaximize)

        freq_vars = LpVariable.dicts("freq", range(len(candidates)), 0, 1, cat='Binary')
        prob += lpSum(freq_vars[i] for i in range(len(candidates))) == n_overtones

        # Objective: maximize golden ratio relationships
        resonance_scores = []
        for i in range(len(candidates)):
            freq = candidates[i]
            # Score based on proximity to golden ratio harmonics
            score = 0
            for harmonic in range(1, n_overtones + 1):
                golden_freq = base_frequency * (phi_val ** harmonic)
                proximity = 1 / (1 + abs(freq - golden_freq) / base_frequency)
                score += proximity
            resonance_scores.append(score * freq_vars[i])

        prob += lpSum(resonance_scores)

    else:  # Balance Efficiency
        prob = LpProblem("Overtone_Packing_Balance", LpMaximize)

        freq_vars = LpVariable.dicts("freq", range(len(candidates)), 0, 1, cat='Binary')
        prob += lpSum(freq_vars[i] for i in range(len(candidates))) == n_overtones

        # Constraint on frequency spacing
        for i in range(len(candidates)):
            for j in range(i+1, len(candidates)):
                freq_ratio = candidates[j] / candidates[i]
                # Penalize ratios too close to 1 (overlapping) or too far
                spacing_penalty = abs(np.log(freq_ratio) - np.log(phi_val))
                prob += freq_vars[i] + freq_vars[j] <= 1 + spacing_penalty * packing_constraint

        # Objective: balance spacing and resonance
        balance_scores = []
        for i in range(len(candidates)):
            freq = candidates[i]
            spacing_score = 0
            resonance_score = 0

            # Spacing score
            for j in range(len(candidates)):
                if i != j:
                    ratio = abs(np.log(freq / candidates[j]))
                    spacing_score += 1 / (1 + ratio)

            # Resonance score
            for harmonic in range(1, n_overtones + 1):
                golden_freq = base_frequency * (phi_val ** harmonic)
                resonance_score += 1 / (1 + abs(freq - golden_freq) / base_frequency)

            balance_scores.append((spacing_score + resonance_score) * freq_vars[i])

        prob += lpSum(balance_scores)

    # Solve the optimization problem
    status = prob.solve()

    # Extract selected frequencies
    selected_freqs = []
    for i in range(len(candidates)):
        if freq_vars[i].value() == 1:
            selected_freqs.append(candidates[i])

    selected_freqs = sorted(selected_freqs)

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Frequency spectrum
    ax1.scatter(candidates, [1] * len(candidates), alpha=0.3, color='gray', label='Available')
    ax1.scatter(selected_freqs, [1] * len(selected_freqs), color='red', s=100, label='Selected')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_title('Frequency Selection')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Ratio analysis
    if len(selected_freqs) > 1:
        ratios = []
        for i in range(len(selected_freqs)):
            for j in range(i+1, len(selected_freqs)):
                ratio = selected_freqs[j] / selected_freqs[i]
                ratios.append(ratio)

        ax2.hist(ratios, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(phi_val, color='red', linestyle='--', label=f'φ = {phi_val:.3f}')
        ax2.set_xlabel('Frequency Ratio')
        ax2.set_ylabel('Count')
        ax2.set_title('Frequency Ratios Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Resonance efficiency
    resonance_efficiency = []
    for freq in selected_freqs:
        efficiency = 0
        for harmonic in range(1, n_overtones + 1):
            golden_freq = base_frequency * (phi_val ** harmonic)
            efficiency += 1 / (1 + abs(freq - golden_freq) / base_frequency)
        resonance_efficiency.append(efficiency)

    ax3.bar(range(len(selected_freqs)), resonance_efficiency, color='green', alpha=0.7)
    ax3.set_xlabel('Selected Frequency Index')
    ax3.set_ylabel('Resonance Efficiency')
    ax3.set_title('Resonance Efficiency per Frequency')
    ax3.grid(True, alpha=0.3)

    # Packing density
    freq_diffs = np.diff(selected_freqs)
    ax4.plot(range(len(freq_diffs)), freq_diffs, 'o-', color='purple')
    ax4.set_xlabel('Interval Index')
    ax4.set_ylabel('Frequency Difference (Hz)')
    ax4.set_title('Frequency Spacing')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Results summary
    st.subheader("Optimization Results")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Selected Frequencies", len(selected_freqs))
    with col2:
        st.metric("Optimization Status", LpStatus[status])
    with col3:
        st.metric("Objective Value", f"{prob.objective.value():.4f}")
    with col4:
        st.metric("Golden Ratio φ", f"{phi_val:.6f}")

    # Display selected frequencies
    st.subheader("Selected Frequencies")
    freq_table = []
    for i, freq in enumerate(selected_freqs):
        ratio_to_base = freq / base_frequency
        nearest_harmonic = round(ratio_to_base)
        deviation = abs(ratio_to_base - nearest_harmonic)
        freq_table.append({
            "Frequency (Hz)": f"{freq:.1f}",
            "Ratio to Base": f"{ratio_to_base:.3f}",
            "Nearest Harmonic": nearest_harmonic,
            "Deviation": f"{deviation:.3f}"
        })

    st.table(freq_table)

    # Analysis
    st.subheader("Packing Analysis")

    if len(selected_freqs) > 1:
        avg_spacing = np.mean(np.diff(selected_freqs))
        std_spacing = np.std(np.diff(selected_freqs))
        packing_efficiency = 1 / (1 + std_spacing / avg_spacing)

        st.write(f"**Average Spacing:** {avg_spacing:.1f} Hz")
        st.write(f"**Spacing Standard Deviation:** {std_spacing:.1f} Hz")
        st.write(f"**Packing Efficiency:** {packing_efficiency:.3f}")

        # Golden ratio compliance
        golden_compliance = 0
        for i in range(len(selected_freqs)):
            for j in range(i+1, len(selected_freqs)):
                ratio = selected_freqs[j] / selected_freqs[i]
                golden_compliance += 1 / (1 + abs(np.log(ratio) - np.log(phi_val)))

        st.write(f"**Golden Ratio Compliance:** {golden_compliance:.2f}")

    st.markdown(f"""
    **Optimization Strategy:**
    - **{optimization_type}**: {LpStatus[status]}
    - **Packing Constraint**: {packing_constraint}
    - **Golden Ratio Integration**: φ = {phi_val:.6f} used for optimal spacing

    **Applications:**
    - **Musical Composition**: Optimal chord voicings with minimal dissonance
    - **Atomic Physics**: Electron orbital arrangements with minimal energy conflicts
    - **Signal Processing**: Frequency channel allocation with minimal interference
    - **Quantum Systems**: Energy level optimization in harmonic oscillators
    """)
