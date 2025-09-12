import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sympy import sqrt

def fibonacci_knot_generator():
    st.header("Fibonacci Knot Generator")
    st.markdown("Create a script to generate Fibonacci-derived torus knots (e.g., (2,3)), using parametric equations and interactive 3D visualization.")

    # Calculate golden ratio
    phi = (1 + sqrt(5)) / 2
    phi_val = float(phi.evalf())

    # Input parameters
    p_param = st.slider("p (longitudinal windings)", 1, 8, 2)
    q_param = st.slider("q (meridional windings)", 1, 8, 3)
    torus_major = st.slider("Torus Major Radius", 2.0, 6.0, 4.0, step=0.1)
    torus_minor = st.slider("Torus Minor Radius", 0.5, 2.0, 1.0, step=0.1)
    num_points = st.slider("Number of Points", 200, 1000, 500, step=50)
    rotation_angle = st.slider("Rotation Angle", 0, 360, 30, step=15)

    # Generate Fibonacci sequence for additional parameters
    fib_seq = [1, 1]
    for i in range(2, max(p_param, q_param) + 3):
        fib_seq.append(fib_seq[i-1] + fib_seq[i-2])

    # Use Fibonacci numbers for knot parameters if available
    if p_param <= len(fib_seq):
        p_fib = fib_seq[p_param-1]
    else:
        p_fib = p_param

    if q_param <= len(fib_seq):
        q_fib = fib_seq[q_param-1]
    else:
        q_fib = q_param

    # Parametric equations for torus knot
    t = np.linspace(0, 2*np.pi, num_points)

    # Standard torus knot
    x_standard = (torus_major + torus_minor * np.cos(q_param * t)) * np.cos(p_param * t)
    y_standard = (torus_major + torus_minor * np.cos(q_param * t)) * np.sin(p_param * t)
    z_standard = torus_minor * np.sin(q_param * t)

    # Fibonacci-derived knot (using Fibonacci numbers)
    x_fibonacci = (torus_major + torus_minor * np.cos(q_fib * t)) * np.cos(p_fib * t)
    y_fibonacci = (torus_major + torus_minor * np.cos(q_fib * t)) * np.sin(p_fib * t)
    z_fibonacci = torus_minor * np.sin(q_fib * t)

    # Golden ratio scaled knot
    x_golden = (torus_major + torus_minor * np.cos(q_param * phi_val * t)) * np.cos(p_param * t)
    y_golden = (torus_major + torus_minor * np.cos(q_param * phi_val * t)) * np.sin(p_param * t)
    z_golden = torus_minor * np.sin(q_param * phi_val * t)

    # Apply rotation
    rotation_rad = np.radians(rotation_angle)

    # Rotation matrix around z-axis
    cos_rot = np.cos(rotation_rad)
    sin_rot = np.sin(rotation_rad)

    # Rotate all knots
    def rotate_points(x, y, z, cos_rot, sin_rot):
        x_rot = x * cos_rot - y * sin_rot
        y_rot = x * sin_rot + y * cos_rot
        return x_rot, y_rot, z

    x_standard_rot, y_standard_rot, z_standard_rot = rotate_points(x_standard, y_standard, z_standard, cos_rot, sin_rot)
    x_fibonacci_rot, y_fibonacci_rot, z_fibonacci_rot = rotate_points(x_fibonacci, y_fibonacci, z_fibonacci, cos_rot, sin_rot)
    x_golden_rot, y_golden_rot, z_golden_rot = rotate_points(x_golden, y_golden, z_golden, cos_rot, sin_rot)

    # Create torus surface for reference
    u_torus = np.linspace(0, 2*np.pi, 50)
    v_torus = np.linspace(0, 2*np.pi, 50)
    U_torus, V_torus = np.meshgrid(u_torus, v_torus)

    X_torus = (torus_major + torus_minor * np.cos(V_torus)) * np.cos(U_torus)
    Y_torus = (torus_major + torus_minor * np.cos(V_torus)) * np.sin(U_torus)
    Z_torus = torus_minor * np.sin(V_torus)

    # Create visualization
    fig = plt.figure(figsize=(16, 12))

    # 3D knot visualization
    ax1 = fig.add_subplot(221, projection='3d')

    # Plot torus surface (semi-transparent)
    ax1.plot_surface(X_torus, Y_torus, Z_torus, alpha=0.1, color='gray')

    # Plot knots
    ax1.plot(x_standard_rot, y_standard_rot, z_standard_rot, 'b-', linewidth=3, label=f'Standard ({p_param},{q_param})')
    ax1.plot(x_fibonacci_rot, y_fibonacci_rot, z_fibonacci_rot, 'r-', linewidth=3, label=f'Fibonacci ({p_fib},{q_fib})')
    ax1.plot(x_golden_rot, y_golden_rot, z_golden_rot, 'g-', linewidth=3, label=f'Golden (φ×{q_param})')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Fibonacci-Derived Torus Knots')
    ax1.legend()
    ax1.set_xlim([-torus_major-1, torus_major+1])
    ax1.set_ylim([-torus_major-1, torus_major+1])
    ax1.set_zlim([-torus_minor-1, torus_minor-1])

    # 2D projections
    ax2 = fig.add_subplot(222)
    ax2.plot(x_standard_rot, y_standard_rot, 'b-', linewidth=2, label=f'Standard ({p_param},{q_param})')
    ax2.plot(x_fibonacci_rot, y_fibonacci_rot, 'r-', linewidth=2, label=f'Fibonacci ({p_fib},{q_fib})')
    ax2.plot(x_golden_rot, y_golden_rot, 'g-', linewidth=2, label=f'Golden (φ×{q_param})')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    # Knot length comparison
    ax3 = fig.add_subplot(223)

    # Calculate knot lengths
    length_standard = np.sum(np.sqrt(np.diff(x_standard)**2 + np.diff(y_standard)**2 + np.diff(z_standard)**2))
    length_fibonacci = np.sum(np.sqrt(np.diff(x_fibonacci)**2 + np.diff(y_fibonacci)**2 + np.diff(z_fibonacci)**2))
    length_golden = np.sum(np.sqrt(np.diff(x_golden)**2 + np.diff(y_golden)**2 + np.diff(z_golden)**2))

    knots = ['Standard', 'Fibonacci', 'Golden']
    lengths = [length_standard, length_fibonacci, length_golden]
    colors = ['blue', 'red', 'green']

    bars = ax3.bar(knots, lengths, color=colors, alpha=0.7)
    ax3.set_ylabel('Knot Length')
    ax3.set_title('Knot Length Comparison')
    ax3.grid(True, alpha=0.3)

    # Add value labels
    for bar, length in zip(bars, lengths):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height, f'{length:.1f}', ha='center', va='bottom')

    # Curvature analysis
    ax4 = fig.add_subplot(224)

    # Calculate curvature (simplified)
    curvature_standard = np.abs(np.gradient(np.gradient(x_standard)) + np.gradient(np.gradient(y_standard)))
    curvature_fibonacci = np.abs(np.gradient(np.gradient(x_fibonacci)) + np.gradient(np.gradient(y_fibonacci)))
    curvature_golden = np.abs(np.gradient(np.gradient(x_golden)) + np.gradient(np.gradient(y_golden)))

    ax4.plot(t[:-2], curvature_standard[:-2], 'b-', alpha=0.7, label='Standard')
    ax4.plot(t[:-2], curvature_fibonacci[:-2], 'r-', alpha=0.7, label='Fibonacci')
    ax4.plot(t[:-2], curvature_golden[:-2], 'g-', alpha=0.7, label='Golden')
    ax4.set_xlabel('Parameter t')
    ax4.set_ylabel('Curvature')
    ax4.set_title('Knot Curvature')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Analysis metrics
    st.subheader("Knot Analysis")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Standard Knot", f"({p_param},{q_param})")
    with col2:
        st.metric("Fibonacci Knot", f"({p_fib},{q_fib})")
    with col3:
        st.metric("Rotation", f"{rotation_angle}°")
    with col4:
        st.metric("Golden Ratio φ", f"{phi_val:.6f}")

    # Detailed analysis
    st.subheader("Knot Properties")

    # Calculate winding numbers
    st.write(f"**Longitudinal Windings (p):** Standard={p_param}, Fibonacci={p_fib}")
    st.write(f"**Meridional Windings (q):** Standard={q_param}, Fibonacci={q_fib}, Golden={q_param}×φ")

    # Knot type classification
    def classify_knot(p, q):
        gcd = np.gcd(p, q)
        if gcd == 1:
            return "Prime Knot"
        else:
            return f"({p//gcd},{q//gcd}) × {gcd}"

    st.write(f"**Standard Knot Type:** {classify_knot(p_param, q_param)}")
    st.write(f"**Fibonacci Knot Type:** {classify_knot(p_fib, q_fib)}")

    # Symmetry analysis
    st.subheader("Symmetry Analysis")

    # Calculate symmetry measures
    def calculate_symmetry(x, y, z):
        # Check rotational symmetry
        symmetry_score = 0
        for angle in [np.pi/2, np.pi, 3*np.pi/2]:
            x_rot = x * np.cos(angle) - y * np.sin(angle)
            y_rot = x * np.sin(angle) + y * np.cos(angle)
            symmetry_score += np.mean(np.sqrt((x - x_rot)**2 + (y - y_rot)**2 + (z - z)**2))
        return symmetry_score / 3

    sym_standard = calculate_symmetry(x_standard, y_standard, z_standard)
    sym_fibonacci = calculate_symmetry(x_fibonacci, y_fibonacci, z_fibonacci)
    sym_golden = calculate_symmetry(x_golden, y_golden, z_golden)

    st.write(f"**Standard Knot Symmetry:** {sym_standard:.3f}")
    st.write(f"**Fibonacci Knot Symmetry:** {sym_fibonacci:.3f}")
    st.write(f"**Golden Knot Symmetry:** {sym_golden:.3f}")

    # Fibonacci sequence display
    st.subheader("Fibonacci Sequence")
    st.write(f"**Generated Sequence:** {fib_seq}")
    st.write(f"**Convergence to φ:** {[fib_seq[i+1]/fib_seq[i] for i in range(min(5, len(fib_seq)-1))]}")

    st.markdown(f"""
    **Fibonacci Knot Theory:**
    - **Torus Knots**: Closed curves on toroidal surfaces with (p,q) winding numbers
    - **Fibonacci Scaling**: Using Fibonacci numbers Fₙ for winding parameters
    - **Golden Ratio Integration**: φ-scaling provides optimal knot properties
    - **Topological Invariants**: Conserved properties under continuous deformations

    **Mathematical Properties:**
    - **Winding Numbers**: p = longitudinal windings, q = meridional windings
    - **Knot Classification**: Based on greatest common divisor gcd(p,q)
    - **Symmetry Groups**: Rotational and reflection symmetries
    - **Length Minimization**: Optimal knot configurations

    **Physical Interpretations:**
    - **DNA Supercoiling**: Double helical structures with Fibonacci winding
    - **Magnetic Field Lines**: Plasma confinement in tokamak devices
    - **Vortex Filaments**: Quantum fluids and superconductors
    - **Protein Folding**: Complex molecular structures

    **Applications:**
    - **Molecular Biology**: DNA topology and chromatin structure
    - **Plasma Physics**: Magnetic field line chaos and reconnection
    - **Quantum Computing**: Topological quantum bits
    - **Fluid Dynamics**: Vortex knot dynamics in turbulence
    """)

    # Interactive exploration
    st.subheader("Interactive Exploration")
    st.markdown("Adjust the winding parameters and rotation to explore different knot topologies and their Fibonacci relationships.")
