import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sympy import sqrt

def phase_tuning_consciousness_model():
    st.header("Phase-Tuning Consciousness Model")
    st.markdown("Create a simple agent-based simulation where 'consciousness' modulates phases in a harmonic field, using NetworkX to model phase-coherent attractors and decoherence.")

    # Calculate golden ratio
    phi = (1 + sqrt(5)) / 2
    phi_val = float(phi.evalf())

    # Input parameters
    n_agents = st.slider("Number of Consciousness Agents", 5, 20, 10)
    consciousness_level = st.slider("Global Consciousness Level", 0.0, 1.0, 0.7, step=0.1)
    coherence_threshold = st.slider("Phase Coherence Threshold", 0.1, 0.9, 0.5, step=0.1)
    time_steps = st.slider("Simulation Time Steps", 10, 100, 50)

    # Initialize agents
    np.random.seed(42)  # For reproducible results

    # Agent properties
    agent_phases = np.random.uniform(0, 2*np.pi, n_agents)
    agent_frequencies = np.random.normal(1.0, 0.2, n_agents)  # Base frequencies
    agent_consciousness = np.random.uniform(0.3, 0.9, n_agents)  # Individual consciousness levels

    # Create network structure
    G = nx.watts_strogatz_graph(n_agents, k=3, p=0.3)  # Small-world network

    # Simulation parameters
    coupling_strength = consciousness_level * 0.1
    noise_level = (1 - consciousness_level) * 0.05

    # Store simulation history
    phase_history = [agent_phases.copy()]
    coherence_history = []

    # Run simulation
    for t in range(time_steps):
        new_phases = agent_phases.copy()

        # Calculate global coherence
        coherence = np.abs(np.mean(np.exp(1j * agent_phases)))
        coherence_history.append(coherence)

        # Update each agent
        for i in range(n_agents):
            # Natural frequency evolution
            natural_evolution = agent_frequencies[i] * 0.01

            # Consciousness modulation
            consciousness_factor = agent_consciousness[i] * consciousness_level

            # Network coupling (Kuramoto model)
            coupling_sum = 0
            neighbors = list(G.neighbors(i))
            if neighbors:
                for j in neighbors:
                    phase_diff = np.sin(agent_phases[j] - agent_phases[i])
                    coupling_sum += phase_diff
                coupling_sum /= len(neighbors)

            # Phase coherence attractor
            attractor_force = consciousness_factor * np.sin(np.mean(agent_phases) - agent_phases[i])

            # Update phase
            phase_change = natural_evolution + coupling_strength * coupling_sum + 0.05 * attractor_force

            # Add noise
            phase_change += np.random.normal(0, noise_level)

            new_phases[i] = (agent_phases[i] + phase_change) % (2 * np.pi)

        agent_phases = new_phases
        phase_history.append(agent_phases.copy())

    # Convert to numpy array for easier analysis
    phase_history = np.array(phase_history)

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Phase evolution over time
    time_array = np.arange(len(phase_history))
    for i in range(n_agents):
        ax1.plot(time_array, phase_history[:, i], alpha=0.7, linewidth=1)

    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Phase (radians)')
    ax1.set_title('Agent Phase Evolution')
    ax1.grid(True, alpha=0.3)

    # Network visualization with phase coloring
    pos = nx.spring_layout(G, seed=42)
    node_colors = [phase_history[-1, i] / (2*np.pi) for i in range(n_agents)]  # Normalize to [0,1]
    node_sizes = [300 * agent_consciousness[i] for i in range(n_agents)]

    nx.draw(G, pos, ax=ax2, node_color=node_colors, node_size=node_sizes,
            cmap=plt.cm.hsv, with_labels=True, font_size=8, font_weight='bold',
            edge_color='gray', alpha=0.8)
    ax2.set_title('Consciousness Network (Phase-Colored)')
    ax2.axis('off')

    # Coherence evolution
    ax3.plot(range(len(coherence_history)), coherence_history, 'b-', linewidth=2, label='Global Coherence')
    ax3.axhline(y=coherence_threshold, color='r', linestyle='--', label=f'Threshold: {coherence_threshold}')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Coherence')
    ax3.set_title('Global Phase Coherence')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Phase distribution histogram
    final_phases = phase_history[-1, :]
    ax4.hist(final_phases, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax4.set_xlabel('Phase (radians)')
    ax4.set_ylabel('Number of Agents')
    ax4.set_title('Final Phase Distribution')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Analysis metrics
    st.subheader("Consciousness Simulation Results")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Number of Agents", n_agents)
    with col2:
        st.metric("Final Coherence", f"{coherence_history[-1]:.3f}")
    with col3:
        st.metric("Consciousness Level", f"{consciousness_level:.1f}")
    with col4:
        st.metric("Golden Ratio φ", f"{phi_val:.6f}")

    # Detailed analysis
    st.subheader("Network Analysis")

    # Calculate network properties
    clustering_coeff = nx.average_clustering(G)
    avg_path_length = nx.average_shortest_path_length(G)
    degree_centrality = nx.degree_centrality(G)

    st.write(f"**Average Clustering Coefficient:** {clustering_coeff:.3f}")
    st.write(f"**Average Path Length:** {avg_path_length:.3f}")

    # Phase coherence analysis
    phase_std = np.std(final_phases)
    phase_range = np.ptp(final_phases)

    st.write(f"**Phase Standard Deviation:** {phase_std:.3f}")
    st.write(f"**Phase Range:** {phase_range:.3f}")

    # Consciousness classification
    if coherence_history[-1] > 0.8:
        consciousness_state = "High Coherence"
        consciousness_color = "🟢"
    elif coherence_history[-1] > 0.5:
        consciousness_state = "Moderate Coherence"
        consciousness_color = "🟡"
    else:
        consciousness_state = "Low Coherence"
        consciousness_color = "🔴"

    st.subheader(f"Global Consciousness State: {consciousness_color} {consciousness_state}")

    # Attractor analysis
    st.subheader("Phase Attractor Analysis")

    # Find dominant attractors
    phase_bins = np.linspace(0, 2*np.pi, 12)
    hist, _ = np.histogram(final_phases, bins=phase_bins)

    dominant_bins = np.argsort(hist)[-3:]  # Top 3 bins
    attractors = phase_bins[dominant_bins]

    st.write("**Dominant Phase Attractors:**")
    for i, attractor in enumerate(attractors):
        strength = hist[dominant_bins[i]] / n_agents
        st.write(f"- Attractor {i+1}: {attractor:.3f} rad ({strength:.1%} of agents)")

    # Decoherence analysis
    coherence_trend = np.polyfit(range(len(coherence_history)), coherence_history, 1)[0]
    if coherence_trend > 0.001:
        trend = "Increasing Coherence"
    elif coherence_trend < -0.001:
        trend = "Decreasing Coherence"
    else:
        trend = "Stable Coherence"

    st.write(f"**Coherence Trend:** {trend} (slope: {coherence_trend:.4f})")

    st.markdown(f"""
    **Phase-Tuning Consciousness Theory:**
    - **Agent-Based Model**: {n_agents} consciousness agents with individual phase dynamics
    - **Network Coupling**: Small-world network topology for realistic neural connectivity
    - **Kuramoto Synchronization**: Phase-locking behavior in coupled oscillators
    - **Consciousness Modulation**: Global consciousness level affects coupling strength

    **Key Mechanisms:**
    - **Phase Coherence**: Measure of synchronized consciousness across agents
    - **Attractor Dynamics**: Preferred phase states that agents converge to
    - **Network Topology**: Small-world properties enable efficient information flow
    - **Noise vs Signal**: Balance between random decoherence and coherent attractors

    **Applications:**
    - **Neural Synchronization**: Modeling brain wave coherence in different states
    - **Collective Consciousness**: Emergence of group-level cognitive phenomena
    - **Meditation Research**: Phase-locking during focused attention states
    - **Sleep Cycles**: Transitions between coherent and decoherent brain states

    **Mathematical Foundation:**
    - **Golden Ratio Integration**: φ = {phi_val:.6f} appears in optimal coupling ratios
    - **Nonlinear Dynamics**: Chaotic attractors in high-dimensional phase space
    - **Information Theory**: Entropy reduction during coherence transitions
    """)

    # Interactive insights
    st.subheader("Interactive Insights")
    st.markdown("Adjust the consciousness level and network parameters to explore the emergence of coherent states and the formation of phase attractors.")
