# Integrated Resonance Dashboard

A comprehensive Streamlit web application for exploring the Unified Harmonic Field Framework through interactive mathematical and physical simulations.

## 🌙 Dark Theme Implementation

The dashboard features a complete dark theme with:
- **Dark Background**: Professional dark color scheme (#0e1117)
- **Custom Styling**: Enhanced UI elements with dark theme colors
- **Matplotlib Integration**: Dark-themed plots with proper contrast
- **Consistent Design**: Unified color palette across all components
- **Accessibility**: High contrast ratios for readability

### Features

#### Implemented Tools (18/19) ✅

1. **Dashboard Overview** 🎯
   - Comprehensive framework introduction
   - Tool categorization and integration examples
   - Interactive getting started guide

2. **Harmonic Series Generator** 🎼
   - Generate and visualize harmonic series (f_n = n * f_0)
   - Plot individual harmonics and their sum
   - Highlight integer ratio interactions

3. **Standing Wave Simulator** 🌊
   - Simulate standing waves from superposed traveling waves
   - Ψ(x, t) = 2A cos(kx) sin(ωt)
   - Mark nodal and antinodal points as "resonant memory zones"

4. **Golden Ratio Harmonic Structurer** 📐
   - Compute Fibonacci sequences converging to φ = (1 + √5)/2
   - Apply golden ratio to structure harmonic overtones
   - Visualize efficiency in recursive growth

5. **Torus Knot Visualizer** 🔗
   - Render parametric torus knots (x(t) = (R + r cos(qt)) cos(pt), etc.)
   - Support for (p,q) torus knots like (2,3) trefoil
   - 3D visualization with animation

6. **Interference Field Mapper** ⚡
   - Simulate interference fields I(x, t) = Σ A_n cos(k_n x - ω_n t + φ_n)
   - Generate heatmaps of nodes (gravity wells) and antinodes (tension zones)

7. **Dimensional Recursion Explorer** 🔄
   - Model dimensionality progression (1D string → 2D membrane → 3D torus)
   - Recursive functions with SymPy projections scaled by φ
   - 3D visualization of hierarchical structures

8. **Overtone Convergence Analyzer** 📊
   - Compute overtone series with golden ratio spacing (f_{n+1}/f_n ≈ φ)
   - Use SciPy for FFT analysis of resonance efficiency
   - Plot convergence graphs and spectral analysis

9. **Corrective Mirror Constant Calculator** 🔧
   - Compute C_m = 3/φ ≈ 1.854 for harmonic equilibrium
   - Apply feedback in phase restoration simulations
   - Plot convergence over iterations with stability analysis

10. **Scalar Resonance Index Computer** 📈
    - Calculate R = 2 |B1 B2| / (|B1|² + |B2|²) for tonal flow modes
    - Visualize resonance strength on color-coded torus
    - Phase difference and frequency ratio analysis

11. **Phyllotaxis Pattern Generator** 🌿
    - Implement Fibonacci growth (F_n = F_{n-1} + F_{n-2})
    - Generate phyllotaxis patterns (sunflower seeds, pine cones, etc.)
    - Scale by golden ratio φ with interactive visualizations

12. **Overtone Packing Efficiency Optimizer** 🎯
    - Use PuLP for linear programming to optimize harmonic packing with φ ratios
    - Minimize phase conflicts in musical intervals or atomic orbitals
    - Multiple optimization strategies (minimize conflicts, maximize resonance, balance efficiency)

13. **Triadic Phenomena Mapper** 🔺
    - Visualize triadic structures (spatial dimensions, force unifications, quantum states, consciousness fields)
    - Overlay Fibonacci knots on 3D grids
    - Trefoil lobe analysis and symmetry calculations

14. **Phase-Tuning Consciousness Model** 🧠
    - Agent-based simulation where consciousness modulates phases in harmonic field
    - NetworkX modeling of phase-coherent attractors and decoherence
    - Kuramoto synchronization with small-world network topology

15. **Coordinated-Rotation Field Emulator** 🌪️
    - Simulate E(θ, t) = A1 e^{i(m1 θ - ω t)} + A2 e^{i(σ m2 θ - ν t + φ0)}
    - ΔΦ leading to trefoil topology when |m1 - σ m2| = 3
    - Polarization knot visualization and field analysis

16. **UHFF Wavefield Summator** 🌊
    - Multi-oscillator summation simulator U(x, t) = Σ A_j sin(k_j x - ω_j t + φ_j)
    - Detect rational ratios and phase alignments for "mass" as standing lattices
    - Energy density analysis and standing wave detection

17. **Toroidal Field Spiral Drawer** 🌀
    - Draw golden spirals (φ-scaling) tracing toroidal fields
    - Simulate DNA helices, magnetic fields, electromagnetic waves, vortex fields
    - Vector field visualization and field strength analysis

18. **Fibonacci Knot Generator** 🪢
    - Generate Fibonacci-derived torus knots (e.g., (2,3))
    - Parametric equations with interactive 3D visualization
    - Knot classification, symmetry analysis, and curvature studies

19. **Mode Superposition Visualizer** ⚛️
    - Superposition of angular-momentum modes S(θ, t) = B1 e^{i(n1 θ - Ω t)} + B2 e^{i(σ n2 θ - Λ t + φ0)}
    - Animate trefoil-like Lissajous figures for |n1 - σ n2| = 3
    - 3D trajectory analysis and power spectrum computation

20. **Tonal Torus Trajectory Simulator** 🎵
    - Model pitch-class motion on T² manifold
    - Apply scalar step operators T_k (θ → θ + kΔ) for seconds, thirds, and fourths
    - Visualize three-lobed paths and analyze musical trajectories

### Placeholder Tools (Ready for Implementation)

- Dimensional Recursion Explorer
- Tonal Torus Trajectory Simulator
- Mode Superposition Visualizer
- Fibonacci Knot Generator
- Corrective Mirror Constant Calculator
- Toroidal Field Spiral Drawer
- Scalar Resonance Index Computer
- UHFF Wavefield Summator
- Coordinated-Rotation Field Emulator
- Phase-Tuning Consciousness Model
- Triadic Phenomena Mapper
- Overtone Packing Efficiency Optimizer

## Installation

1. Clone the repository:
```bash
git clone https://github.com/3rdeyesamurai/Integrated-Resonance-Dashboard.git
cd Integrated-Resonance-Dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser to `http://localhost:8501`

## Dependencies

- streamlit
- numpy
- matplotlib
- sympy
- scipy
- pygame
- networkx
- pulp

## Usage

Select any tool from the sidebar to explore different aspects of the Unified Harmonic Field Framework. Each tool provides interactive controls to modify parameters and visualize the results in real-time.

## Mathematical Framework

The dashboard implements various mathematical concepts from the Unified Harmonic Field Framework:

- **Harmonic Series**: Integer multiples of fundamental frequencies
- **Golden Ratio**: φ = (1 + √5)/2 and its applications in growth and resonance
- **Fibonacci Sequence**: Recursive growth patterns converging to φ
- **Torus Topology**: Parametric equations for knot theory and field structures
- **Wave Interference**: Superposition of multiple wave sources
- **Phyllotaxis**: Optimal packing patterns in nature using Fibonacci spirals

## Contributing

The project is structured with modular design. Each tool is implemented in its own module under the `modules/` directory. To add new features:

1. Create a new Python file in `modules/`
2. Implement the function with the same name as the file
3. Import and add to the main app.py
4. Update the sidebar selection options

## License

This project is open-source and available under standard licensing terms.
