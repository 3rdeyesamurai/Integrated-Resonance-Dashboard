import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import symbols, sqrt, cos, sin, pi
import scipy.signal
import networkx as nx
from pulp import LpProblem, LpVariable, lpSum, LpMinimize
import pygame  # Note: Pygame may not work in Streamlit cloud, use for local only

# Dark theme helper function for matplotlib
def apply_dark_theme_to_plot():
    """Apply dark theme styling to matplotlib plots"""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'axes.facecolor': '#1a1c23',
        'figure.facecolor': '#0e1117',
        'axes.edgecolor': '#30363d',
        'axes.labelcolor': '#c9d1d9',
        'xtick.color': '#c9d1d9',
        'ytick.color': '#c9d1d9',
        'text.color': '#c9d1d9',
        'grid.color': '#30363d',
        'grid.alpha': 0.3,
        'legend.facecolor': '#1a1c23',
        'legend.edgecolor': '#30363d',
        'legend.labelcolor': '#c9d1d9'
    })

# Import custom modules
from modules.harmonic_series import harmonic_series_generator
from modules.standing_wave import standing_wave_simulator
from modules.golden_ratio import golden_ratio_harmonic_structurer
from modules.torus_knot import torus_knot_visualizer
from modules.interference_field import interference_field_mapper
from modules.dimensional_recursion import dimensional_recursion_explorer
from modules.overtone_convergence import overtone_convergence_analyzer
from modules.tonal_torus import tonal_torus_trajectory_simulator
from modules.mode_superposition import mode_superposition_visualizer
from modules.fibonacci_knot import fibonacci_knot_generator
from modules.corrective_mirror import corrective_mirror_constant_calculator
from modules.toroidal_field import toroidal_field_spiral_drawer
from modules.scalar_resonance import scalar_resonance_index_computer
from modules.uhff_wavefield import uhff_wavefield_summator
from modules.coordinated_rotation import coordinated_rotation_field_emulator
from modules.phyllotaxis import phyllotaxis_pattern_generator
from modules.phase_tuning import phase_tuning_consciousness_model
from modules.triadic_phenomena import triadic_phenomena_mapper
from modules.overtone_packing import overtone_packing_efficiency_optimizer

# Dark theme configuration
st.set_page_config(
    page_title="Integrated Resonance Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for dark theme with improved readability and accessibility
st.markdown("""
<style>
    /* ===== DARK THEME FOUNDATION ===== */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* ===== SIDEBAR & NAVIGATION ===== */
    .css-1d391kg, .css-12oz5g7 {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* ===== TYPOGRAPHY & TEXT ===== */
    /* Headers with better contrast */
    .css-10trblm, .css-1v0mbdj {
        color: #00d4aa !important;
        font-weight: 600;
        letter-spacing: -0.025em;
    }

    /* Body text with improved readability */
    .css-1r6slb0 p, .css-1r6slb0 li, .css-1r6slb0 div {
        color: #e6edf3 !important;
        line-height: 1.6;
        font-size: 14px;
    }

    /* Subtle text for secondary information */
    .css-1r6slb0 .stText, .css-1r6slb0 .stMarkdown {
        color: #c9d1d9;
    }

    /* ===== INTERACTIVE ELEMENTS ===== */
    /* Metric cards with enhanced contrast */
    .css-1r6slb0 .css-1otj3f4 {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    }

    /* Metric labels and values */
    .css-1r6slb0 .css-1otj3f4 .css-1r0o7nj {
        color: #00d4aa !important;
        font-weight: 600;
    }

    .css-1r6slb0 .css-1otj3f4 .css-1xarl3l {
        color: #e6edf3 !important;
        font-weight: 500;
    }

    /* ===== FORM CONTROLS ===== */
    /* Slider container */
    .css-1cpxqw2 {
        background-color: #161b22;
        border-radius: 6px;
        padding: 8px;
    }

    /* Slider track and handle */
    .css-1cpxqw2 .stSlider .st-bs {
        background-color: #30363d;
    }

    .css-1cpxqw2 .stSlider .st-bq {
        background-color: #00d4aa;
    }

    /* Buttons with better contrast */
    .css-1cpxqw2 button, .stButton button {
        background-color: #00d4aa !important;
        color: #0e1117 !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }

    .css-1cpxqw2 button:hover, .stButton button:hover {
        background-color: #00b894 !important;
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0, 212, 170, 0.3);
    }

    /* Text inputs with dark theme */
    .css-1cpxqw2 input, .stTextInput input {
        background-color: #30363d !important;
        color: #e6edf3 !important;
        border: 2px solid #484f58 !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        font-size: 14px !important;
    }

    .css-1cpxqw2 input:focus, .stTextInput input:focus {
        border-color: #00d4aa !important;
        box-shadow: 0 0 0 2px rgba(0, 212, 170, 0.2) !important;
    }

    /* Select boxes */
    .css-1cpxqw2 select, .stSelectbox select {
        background-color: #30363d !important;
        color: #e6edf3 !important;
        border: 2px solid #484f58 !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        font-size: 14px !important;
    }

    .css-1cpxqw2 select:focus, .stSelectbox select:focus {
        border-color: #00d4aa !important;
        box-shadow: 0 0 0 2px rgba(0, 212, 170, 0.2) !important;
    }

    /* ===== DATA DISPLAY ===== */
    /* Tables with dark theme */
    .css-1r6slb0 table {
        background-color: #161b22 !important;
        color: #e6edf3 !important;
        border-collapse: collapse;
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }

    .css-1r6slb0 th {
        background-color: #21262d !important;
        color: #00d4aa !important;
        font-weight: 600 !important;
        padding: 12px 16px !important;
        border-bottom: 2px solid #30363d !important;
    }

    .css-1r6slb0 td {
        padding: 10px 16px !important;
        border-bottom: 1px solid #30363d !important;
        color: #c9d1d9 !important;
    }

    .css-1r6slb0 tr:hover {
        background-color: #21262d !important;
    }

    /* ===== CODE & TECHNICAL ===== */
    /* Code blocks with syntax highlighting */
    .css-1r6slb0 code {
        background-color: #30363d !important;
        color: #00d4aa !important;
        border-radius: 4px !important;
        padding: 2px 6px !important;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important;
        font-size: 13px !important;
    }

    /* Pre-formatted code blocks */
    .css-1r6slb0 pre {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        padding: 16px !important;
    }

    .css-1r6slb0 pre code {
        background-color: transparent !important;
        color: #e6edf3 !important;
        padding: 0 !important;
    }

    /* ===== EXPANDABLE ELEMENTS ===== */
    /* Expanders with better styling */
    .css-1r6slb0 .streamlit-expanderHeader {
        background-color: #161b22 !important;
        color: #00d4aa !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease;
    }

    .css-1r6slb0 .streamlit-expanderHeader:hover {
        background-color: #21262d !important;
        border-color: #00d4aa !important;
    }

    /* ===== ALERTS & MESSAGES ===== */
    /* Success messages */
    .css-1r6slb0 .element-container .stAlert[data-baseweb="notification"] {
        background-color: #161b22 !important;
        border: 1px solid #238636 !important;
        border-radius: 8px !important;
        color: #56d364 !important;
    }

    /* Error messages */
    .css-1r6slb0 .element-container .stAlert[data-baseweb="notification"][data-baseweb="notification-type-error"] {
        background-color: #161b22 !important;
        border: 1px solid #da3633 !important;
        color: #f85149 !important;
    }

    /* Warning messages */
    .css-1r6slb0 .element-container .stAlert[data-baseweb="notification"][data-baseweb="notification-type-warning"] {
        background-color: #161b22 !important;
        border: 1px solid #bb8009 !important;
        color: #d29922 !important;
    }

    /* Info messages */
    .css-1r6slb0 .element-container .stAlert[data-baseweb="notification"][data-baseweb="notification-type-info"] {
        background-color: #161b22 !important;
        border: 1px solid #79c0ff !important;
        color: #79c0ff !important;
    }

    /* ===== VISUALIZATIONS ===== */
    /* Plot containers */
    .css-1r6slb0 .plotly-graph-div, .css-1r6slb0 .matplotlib-figure {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }

    /* ===== SCROLLBARS ===== */
    /* Custom scrollbar styling */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }

    ::-webkit-scrollbar-track {
        background: #161b22;
        border-radius: 6px;
    }

    ::-webkit-scrollbar-thumb {
        background: #30363d;
        border-radius: 6px;
        border: 2px solid #161b22;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #484f58;
    }

    ::-webkit-scrollbar-corner {
        background: #161b22;
    }

    /* ===== FOCUS STATES ===== */
    /* Enhanced focus indicators for accessibility */
    .css-1cpxqw2 input:focus, .css-1cpxqw2 select:focus, .css-1cpxqw2 button:focus {
        outline: 2px solid #00d4aa !important;
        outline-offset: 2px !important;
    }

    /* ===== RESPONSIVE DESIGN ===== */
    /* Mobile-friendly adjustments */
    @media (max-width: 768px) {
        .css-1r6slb0 .css-1otj3f4 {
            margin-bottom: 12px;
        }

        .css-1r6slb0 table {
            font-size: 12px;
        }

        .css-1r6slb0 th, .css-1r6slb0 td {
            padding: 8px 12px !important;
        }
    }

    /* ===== SEMANTIC COLORS ===== */
    /* Success states */
    .css-1r6slb0 .stSuccess {
        background-color: #161b22 !important;
        border: 1px solid #238636 !important;
        color: #56d364 !important;
    }

    /* Error states */
    .css-1r6slb0 .stError {
        background-color: #161b22 !important;
        border: 1px solid #da3633 !important;
        color: #f85149 !important;
    }

    /* Warning states */
    .css-1r6slb0 .stWarning {
        background-color: #161b22 !important;
        border: 1px solid #bb8009 !important;
        color: #d29922 !important;
    }

    /* Info states */
    .css-1r6slb0 .stInfo {
        background-color: #161b22 !important;
        border: 1px solid #79c0ff !important;
        color: #79c0ff !important;
    }

    /* ===== ACCESSIBILITY ENHANCEMENTS ===== */
    /* High contrast mode support */
    @media (prefers-contrast: high) {
        .stApp {
            background-color: #000000;
        }

        .css-1r6slb0 p, .css-1r6slb0 li, .css-1r6slb0 div {
            color: #ffffff !important;
        }

        .css-1r6slb0 .css-1otj3f4 {
            border: 2px solid #ffffff !important;
        }
    }

    /* Reduced motion support */
    @media (prefers-reduced-motion: reduce) {
        .css-1cpxqw2 button, .stButton button {
            transition: none !important;
        }

        .css-1r6slb0 .streamlit-expanderHeader {
            transition: none !important;
        }
    }
</style>
""", unsafe_allow_html=True)

st.title("Integrated Resonance Dashboard")
st.markdown("Explore the Unified Harmonic Field Framework through interactive tools.")

# Sidebar for tool selection
tool = st.sidebar.selectbox(
    "Select Tool",
    [
        "Dashboard Overview",
        "Harmonic Series Generator",
        "Standing Wave Simulator",
        "Golden Ratio Harmonic Structurer",
        "Torus Knot Visualizer",
        "Interference Field Mapper",
        "Dimensional Recursion Explorer",
        "Overtone Convergence Analyzer",
        "Tonal Torus Trajectory Simulator",
        "Mode Superposition Visualizer",
        "Fibonacci Knot Generator",
        "Corrective Mirror Constant Calculator",
        "Toroidal Field Spiral Drawer",
        "Scalar Resonance Index Computer",
        "UHFF Wavefield Summator",
        "Coordinated-Rotation Field Emulator",
        "Phyllotaxis Pattern Generator",
        "Phase-Tuning Consciousness Model",
        "Triadic Phenomena Mapper",
        "Overtone Packing Efficiency Optimizer"
    ]
)

# Dashboard Overview function
def dashboard_overview():
    st.header("🎵 Unified Harmonic Field Framework Dashboard")
    st.markdown("""
    Welcome to the Integrated Resonance Dashboard - a comprehensive exploration platform for the Unified Harmonic Field Framework (UHFF).
    This dashboard integrates multiple mathematical and physical tools to investigate harmonic phenomena across different domains.
    """)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Implemented Tools", "10", "70% Complete")
    with col2:
        st.metric("Mathematical Domains", "5", "Wave, Field, Topology")
    with col3:
        st.metric("Visualization Types", "8", "2D/3D Plots, Animations")
    with col4:
        st.metric("Interactive Parameters", "50+", "Real-time Control")

    # Framework overview
    st.subheader("🔬 Framework Overview")
    st.markdown("""
    The Unified Harmonic Field Framework explores how fundamental mathematical constants and patterns
    manifest across physics, biology, and consciousness. Key principles include:
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Core Principles:**
        - **Golden Ratio (φ)**: Optimal scaling in natural systems
        - **Harmonic Series**: Integer relationships in frequency domains
        - **Torus Topology**: Fundamental geometry of field interactions
        - **Fibonacci Sequences**: Recursive growth patterns
        - **Phase Synchronization**: Coherent system behavior
        """)

    with col2:
        st.markdown("""
        **Applications:**
        - **Physics**: Wave interference, resonance phenomena
        - **Biology**: Phyllotaxis, neural synchronization
        - **Music**: Harmonic structures, tonal relationships
        - **Consciousness**: Phase coherence, field interactions
        - **Technology**: Signal processing, control systems
        """)

    # Tool categories
    st.subheader("🛠️ Tool Categories")

    categories = {
        "🎼 Harmonic Analysis": ["Harmonic Series Generator", "Overtone Convergence Analyzer", "Golden Ratio Harmonic Structurer"],
        "🌊 Wave Phenomena": ["Standing Wave Simulator", "Interference Field Mapper", "UHFF Wavefield Summator"],
        "🔗 Topological Structures": ["Torus Knot Visualizer", "Dimensional Recursion Explorer", "Fibonacci Knot Generator"],
        "📐 Mathematical Constants": ["Corrective Mirror Constant Calculator", "Scalar Resonance Index Computer"],
        "🌿 Natural Patterns": ["Phyllotaxis Pattern Generator", "Triadic Phenomena Mapper"],
        "⚡ Field Interactions": ["Coordinated-Rotation Field Emulator", "Toroidal Field Spiral Drawer"],
        "🧠 Consciousness Models": ["Phase-Tuning Consciousness Model", "Mode Superposition Visualizer"],
        "🎯 Optimization": ["Overtone Packing Efficiency Optimizer", "Tonal Torus Trajectory Simulator"]
    }

    for category, tools in categories.items():
        with st.expander(f"{category} ({len(tools)} tools)"):
            for tool in tools:
                implemented = "✅" if tool in [
                    "Harmonic Series Generator", "Standing Wave Simulator", "Golden Ratio Harmonic Structurer",
                    "Torus Knot Visualizer", "Interference Field Mapper", "Dimensional Recursion Explorer",
                    "Overtone Convergence Analyzer", "Corrective Mirror Constant Calculator",
                    "Scalar Resonance Index Computer", "Phyllotaxis Pattern Generator"
                ] else "⏳"
                st.write(f"{implemented} {tool}")

    # Integration demonstration
    st.subheader("🔄 Integration Example")
    st.markdown("See how different tools work together in the UHFF framework:")

    # Create a simple integration visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Golden ratio spiral
    phi = (1 + np.sqrt(5)) / 2
    theta = np.linspace(0, 4*np.pi, 100)
    r = 0.1 * phi ** (theta / (np.pi/2))
    ax1.plot(r * np.cos(theta), r * np.sin(theta), 'goldenrod', linewidth=3)
    ax1.set_title('Golden Ratio Spiral')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)

    # Harmonic series
    n = np.arange(1, 11)
    harmonics = n * 440  # A4 note
    ax2.bar(n, harmonics, color='skyblue', alpha=0.7)
    ax2.set_xlabel('Harmonic Number')
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_title('Harmonic Series')
    ax2.grid(True, alpha=0.3)

    # Torus knot (2,3) - 2D projection
    t = np.linspace(0, 2*np.pi, 100)
    p, q = 2, 3
    R, r = 3, 1
    x = (R + r * np.cos(q * t)) * np.cos(p * t)
    y = (R + r * np.cos(q * t)) * np.sin(p * t)
    ax3.plot(x, y, 'purple', linewidth=2)
    ax3.set_title('(2,3) Torus Knot (2D projection)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.grid(True, alpha=0.3)

    # Interference pattern
    x_int = np.linspace(0, 10, 100)
    I = 2 * np.cos(np.pi * x_int) * np.sin(2 * np.pi * x_int)
    ax4.plot(x_int, I, 'red', linewidth=2)
    ax4.fill_between(x_int, I, alpha=0.3, color='red')
    ax4.set_xlabel('Position')
    ax4.set_ylabel('Intensity')
    ax4.set_title('Wave Interference')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Getting started
    st.subheader("🚀 Getting Started")
    st.markdown("""
    1. **Explore Individual Tools**: Select any tool from the sidebar to dive deep into specific phenomena
    2. **Understand Relationships**: See how golden ratio, harmonics, and topology interconnect
    3. **Experiment Interactively**: Adjust parameters in real-time to observe system behavior
    4. **Connect Concepts**: Use multiple tools together to understand the unified framework

    **Key Insight**: The golden ratio φ ≈ 1.618 appears throughout nature and mathematics,
    providing optimal scaling for harmonic systems, from musical intervals to biological growth patterns.
    """)

    st.info("💡 **Tip**: Start with the 'Golden Ratio Harmonic Structurer' to understand the fundamental mathematical constant, then explore how it manifests in different physical systems.")

# Main content area
if tool == "Dashboard Overview":
    dashboard_overview()
elif tool == "Harmonic Series Generator":
    harmonic_series_generator()
elif tool == "Standing Wave Simulator":
    standing_wave_simulator()
elif tool == "Golden Ratio Harmonic Structurer":
    golden_ratio_harmonic_structurer()
elif tool == "Torus Knot Visualizer":
    torus_knot_visualizer()
elif tool == "Interference Field Mapper":
    interference_field_mapper()
elif tool == "Dimensional Recursion Explorer":
    dimensional_recursion_explorer()
elif tool == "Overtone Convergence Analyzer":
    overtone_convergence_analyzer()
elif tool == "Tonal Torus Trajectory Simulator":
    tonal_torus_trajectory_simulator()
elif tool == "Mode Superposition Visualizer":
    mode_superposition_visualizer()
elif tool == "Fibonacci Knot Generator":
    fibonacci_knot_generator()
elif tool == "Corrective Mirror Constant Calculator":
    corrective_mirror_constant_calculator()
elif tool == "Toroidal Field Spiral Drawer":
    toroidal_field_spiral_drawer()
elif tool == "Scalar Resonance Index Computer":
    scalar_resonance_index_computer()
elif tool == "UHFF Wavefield Summator":
    uhff_wavefield_summator()
elif tool == "Coordinated-Rotation Field Emulator":
    coordinated_rotation_field_emulator()
elif tool == "Phyllotaxis Pattern Generator":
    phyllotaxis_pattern_generator()
elif tool == "Phase-Tuning Consciousness Model":
    phase_tuning_consciousness_model()
elif tool == "Triadic Phenomena Mapper":
    triadic_phenomena_mapper()
elif tool == "Overtone Packing Efficiency Optimizer":
    overtone_packing_efficiency_optimizer()

st.sidebar.markdown("---")
st.sidebar.markdown("Built with Streamlit for exploring Unified Harmonic Field Framework predictions.")
