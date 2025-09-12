import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, sqrt, fibonacci

def golden_ratio_harmonic_structurer():
    st.header("Golden Ratio Harmonic Structurer")
    st.markdown("Compute Fibonacci sequences converging to the golden ratio φ and apply it to structure harmonic overtones.")

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
