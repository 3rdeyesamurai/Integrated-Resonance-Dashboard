import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, sqrt, fibonacci

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

def golden_ratio_harmonic_structurer():
    st.header("🌟 RVM-Enhanced Golden Ratio Harmonic Structurer")
    st.markdown("""
    Compute Fibonacci sequences converging to the golden ratio φ with RVM digital roots integration.
    Features vortex patterns, doubling circuit harmonics, and corrective constant Cm = 3/φ.
    """)

    # RVM Golden Ratio Integration
    with st.expander("🔢 RVM Golden Ratio Foundations"):
        st.markdown("""
        **Fibonacci Convergence**: Fn/Fn−1 → φ as n → ∞
        **Corrective Constant**: Cm = 3/φ ≈ 1.854 (phase restoration)
        **RVM Digital Roots**: Applied to Fibonacci numbers
        **Vortex Harmonics**: Doubling circuit applied to golden ratio scaling
        """)

        phi = (1 + sqrt(5)) / 2
        cm = 3 / float(phi.evalf())
        st.write(f"**Golden Ratio φ** = {float(phi.evalf()):.6f}")
        st.write(f"**Corrective Constant Cm** = 3/φ = {cm:.6f}")

    # Calculate golden ratio
    phi = (1 + sqrt(5)) / 2
    st.write(f"Golden Ratio φ = {float(phi.evalf()):.6f}")

    # Input parameters
    n_terms = st.slider("Number of Fibonacci Terms", 5, 20, 10)
    base_freq = st.slider("Base Frequency (Hz)", 50, 500, 220, step=10)

    # Generate Fibonacci sequence
    fib_seq = [fibonacci(i) for i in range(1, n_terms + 1)]
    st.write(f"Fibonacci Sequence: {fib_seq}")

    # Compute ratios converging to phi
    ratios = []
    for i in range(1, len(fib_seq)):
        ratio = fib_seq[i] / fib_seq[i-1]
        ratios.append(float(ratio))
    st.write(f"Converging Ratios: {[f'{r:.6f}' for r in ratios]}")

    # Apply to harmonic overtones
    overtones = [base_freq * (phi ** i) for i in range(n_terms)]
    st.write(f"Golden Ratio Structured Overtones: {[f'{f:.2f}' for f in overtones]}")

    # Visualize convergence
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Fibonacci sequence
    ax1.plot(range(1, n_terms + 1), fib_seq, 'o-', label='Fibonacci Numbers')
    ax1.set_xlabel('Term')
    ax1.set_ylabel('Value')
    ax1.set_title('Fibonacci Sequence')
    ax1.grid(True)

    # Convergence to phi
    ax2.plot(range(2, n_terms + 1), ratios, 'r-o', label='F(n)/F(n-1)')
    ax2.axhline(y=float(phi.evalf()), color='g', linestyle='--', label=f'φ = {float(phi.evalf()):.6f}')
    ax2.set_xlabel('Term')
    ax2.set_ylabel('Ratio')
    ax2.set_title('Convergence to Golden Ratio')
    ax2.legend()
    ax2.grid(True)

    st.pyplot(fig)

    # RVM Digital Roots Analysis
    st.subheader("🔢 RVM Digital Roots of Fibonacci Numbers")
    fib_digital_roots = [digital_root(fib) for fib in fib_seq]

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Fibonacci Numbers:**")
        st.write(fib_seq)
    with col2:
        st.write("**Digital Roots:**")
        st.write(fib_digital_roots)

    # Visualize digital roots
    fig_dr, ax_dr = plt.subplots(figsize=(10, 5))
    ax_dr.plot(range(1, n_terms + 1), fib_seq, 'b-o', label='Fibonacci Numbers', alpha=0.7)
    ax_dr.set_xlabel('Term')
    ax_dr.set_ylabel('Value (log scale)', color='b')
    ax_dr.set_yscale('log')
    ax_dr.tick_params(axis='y', labelcolor='b')

    ax_dr2 = ax_dr.twinx()
    ax_dr2.plot(range(1, n_terms + 1), fib_digital_roots, 'r-s', label='Digital Roots', linewidth=2)
    ax_dr2.set_ylabel('Digital Root', color='r')
    ax_dr2.tick_params(axis='y', labelcolor='r')
    ax_dr2.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 9])

    ax_dr.set_title('Fibonacci Numbers and Their RVM Digital Roots')
    fig_dr.legend(loc='upper left')
    st.pyplot(fig_dr)

    # Vortex Pattern with Golden Ratio
    st.subheader("🌀 RVM Vortex Pattern with Golden Ratio Scaling")
    fig_vortex, ax_vortex = plt.subplots(figsize=(8, 8))

    # Create vortex pattern using golden ratio scaling
    rvm_points = [1, 2, 4, 8, 7, 5, 3, 6, 9]
    angles = np.linspace(0, 2*np.pi, 9, endpoint=False)

    # Apply golden ratio scaling to radius
    radii = [float(phi.evalf()) ** (i % 6) for i in range(9)]

    for i, (point, angle, radius_scale) in enumerate(zip(rvm_points, angles, radii)):
        x_point = radius_scale * np.cos(angle)
        y_point = radius_scale * np.sin(angle)

        # Color code based on 3-6-9 triad
        if point in [3, 6, 9]:
            color = 'red'
        else:
            color = 'blue'

        ax_vortex.scatter(x_point, y_point, s=100 * radius_scale, c=color, alpha=0.8, edgecolors='black')
        ax_vortex.text(x_point, y_point, str(point), ha='center', va='center', fontsize=10, fontweight='bold')

    # Draw spiral using golden ratio
    theta_spiral = np.linspace(0, 4*np.pi, 100)
    r_spiral = 0.1 * float(phi.evalf()) ** (theta_spiral / (2*np.pi))
    x_spiral = r_spiral * np.cos(theta_spiral)
    y_spiral = r_spiral * np.sin(theta_spiral)
    ax_vortex.plot(x_spiral, y_spiral, 'g-', alpha=0.5, linewidth=2, label='Golden Spiral')

    ax_vortex.set_aspect('equal')
    ax_vortex.set_title('RVM Vortex Pattern with Golden Ratio Scaling')
    ax_vortex.grid(True, alpha=0.3)
    ax_vortex.legend()
    st.pyplot(fig_vortex)

    # Visualize harmonic structure
    st.subheader("Harmonic Overtone Structure")
    fig2, ax = plt.subplots(figsize=(10, 6))
    positions = np.arange(len(overtones))
    ax.bar(positions, overtones, color='skyblue')
    ax.set_xlabel('Overtone Number')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title('Golden Ratio Structured Harmonic Overtones')
    ax.set_xticks(positions)
    ax.set_xticklabels([f'O{i+1}' for i in range(len(overtones))])
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # Efficiency in recursive growth
    st.subheader("Recursive Growth Efficiency")
    growth_rates = [ratios[i] / ratios[i-1] if i > 0 else 1 for i in range(len(ratios))]
    fig3, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(growth_rates) + 1), growth_rates, 'g-o')
    ax.axhline(y=1, color='r', linestyle='--', label='No growth')
    ax.set_xlabel('Step')
    ax.set_ylabel('Growth Rate')
    ax.set_title('Recursive Growth Efficiency')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig3)

    st.markdown("""
    **Golden Ratio Applications:**
    - Fibonacci sequence ratios converge to φ ≈ 1.618
    - Applied to harmonic overtones for optimal spacing
    - Demonstrates efficiency in recursive growth patterns
    """)
