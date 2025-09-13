import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

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

def torus_knot_visualizer():
    st.header("🌀 RVM-Enhanced Torus Knot Visualizer")
    st.markdown("""
    Render parametric torus knots with Rodin Vortex Math (RVM) integration.
    Features RVM 9-point circle, doubling circuit, and 3-6-9 control axis visualization.
    """)

    # RVM Foundations Display
    with st.expander("🔢 RVM Foundations"):
        st.markdown("""
        **Digital Root**: Numbers reduced to single digit 1-9 (9 as singularity)
        **Doubling Circuit**: 1→2→4→8→7→5→1 (6-step cycle)
        **3-6-9 Control Axis**: Energy flow control mechanism
        **Vortex Pattern**: 9-point circle representing energy flow
        """)

        # Show RVM 9-point circle
        rvm_points = [1, 2, 4, 8, 7, 5, 3, 6, 9]
        doubling_seq = vortex_doubling_sequence()

        col1, col2 = st.columns(2)
        with col1:
            st.write("**RVM 9-Point Circle:**")
            st.write(f"Points: {rvm_points}")
        with col2:
            st.write("**Doubling Sequence:**")
            st.write(f"1→2→4→8→7→5→{doubling_seq[-1]}")

    # Input parameters
    p = st.slider("p (longitudinal windings)", 1, 10, 2)
    q = st.slider("q (meridional windings)", 1, 10, 3)
    R = st.slider("Major Radius (R)", 1.0, 5.0, 3.0, step=0.1)
    r = st.slider("Minor Radius (r)", 0.1, 2.0, 1.0, step=0.1)
    num_points = st.slider("Number of Points", 100, 1000, 500, step=50)

    # Parametric equations for torus knot
    t = np.linspace(0, 2 * np.pi, num_points)

    x = (R + r * np.cos(q * t)) * np.cos(p * t)
    y = (R + r * np.cos(q * t)) * np.sin(p * t)
    z = r * np.sin(q * t)

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the knot
    ax.plot(x, y, z, 'b-', linewidth=2, label=f'({p},{q}) Torus Knot')

    # Plot the torus surface for reference
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, 2 * np.pi, 50)
    U, V = np.meshgrid(u, v)
    X = (R + r * np.cos(V)) * np.cos(U)
    Y = (R + r * np.cos(V)) * np.sin(U)
    Z = r * np.sin(V)
    ax.plot_surface(X, Y, Z, alpha=0.1, color='gray')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'({p},{q}) Torus Knot on Torus')
    ax.legend()

    # Set equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    st.pyplot(fig)

    # RVM 9-Point Circle Visualization
    st.subheader("🌀 RVM 9-Point Vortex Circle")
    fig_rvm, ax_rvm = plt.subplots(figsize=(8, 8))

    # Create 9-point circle
    rvm_points = [1, 2, 4, 8, 7, 5, 3, 6, 9]
    angles = np.linspace(0, 2*np.pi, 9, endpoint=False)
    radius = 3

    # Plot points
    for i, (point, angle) in enumerate(zip(rvm_points, angles)):
        x_point = radius * np.cos(angle)
        y_point = radius * np.sin(angle)

        # Color code: red for 3-6-9 triad, blue for doubling circuit
        if point in [3, 6, 9]:
            color = 'red'
            label = f'3-6-9 Control ({point})'
        else:
            color = 'blue'
            label = f'Doubling Circuit ({point})'

        ax_rvm.scatter(x_point, y_point, s=200, c=color, alpha=0.8, edgecolors='black', linewidth=2)
        ax_rvm.text(x_point, y_point, str(point), ha='center', va='center', fontsize=12, fontweight='bold')

    # Draw circle
    circle = plt.Circle((0, 0), radius, fill=False, color='gray', linestyle='--', alpha=0.5)
    ax_rvm.add_artist(circle)

    # Draw connections for doubling circuit
    doubling_seq = vortex_doubling_sequence()
    for i in range(len(doubling_seq)-1):
        start_point = doubling_seq[i]
        end_point = doubling_seq[i+1]

        start_idx = rvm_points.index(start_point)
        end_idx = rvm_points.index(end_point)

        start_angle = angles[start_idx]
        end_angle = angles[end_idx]

        start_x = radius * np.cos(start_angle)
        start_y = radius * np.sin(start_angle)
        end_x = radius * np.cos(end_angle)
        end_y = radius * np.sin(end_angle)

        ax_rvm.arrow(start_x, start_y, end_x-start_x, end_y-start_y,
                    head_width=0.3, head_length=0.3, fc='blue', ec='blue', alpha=0.7)

    ax_rvm.set_xlim(-radius-1, radius+1)
    ax_rvm.set_ylim(-radius-1, radius+1)
    ax_rvm.set_aspect('equal')
    ax_rvm.set_title('RVM 9-Point Vortex Circle with Doubling Circuit Flow')
    ax_rvm.grid(True, alpha=0.3)

    # Add legend
    red_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='3-6-9 Control Axis')
    blue_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Doubling Circuit')
    ax_rvm.legend(handles=[red_dot, blue_dot], loc='upper right')

    st.pyplot(fig_rvm)

    # Animation of winding
    st.subheader("Animation of Knot Formation")
    fig_anim, ax_anim = plt.subplots(figsize=(8, 6))
    ax_anim = fig_anim.add_subplot(111, projection='3d')

    # Plot torus
    ax_anim.plot_surface(X, Y, Z, alpha=0.1, color='gray')

    line, = ax_anim.plot([], [], [], 'b-', linewidth=2)

    def animate(frame):
        current_t = t[:frame+1]
        current_x = x[:frame+1]
        current_y = y[:frame+1]
        current_z = z[:frame+1]
        line.set_data(current_x, current_y)
        line.set_3d_properties(current_z)
        return line,

    ani = FuncAnimation(fig_anim, animate, frames=num_points, interval=50, blit=False)

    st.pyplot(fig_anim)

    # Knot properties
    st.subheader("Knot Properties")
    st.write(f"Type: ({p},{q}) Torus Knot")
    st.write(f"Major Radius: {R}")
    st.write(f"Minor Radius: {r}")
    st.write(f"Total windings around torus: {p}")
    st.write(f"Windings through torus holes: {q}")

    if p == 2 and q == 3:
        st.write("This is a trefoil knot!")

    # Calculate knot length approximation
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    segment_lengths = np.sqrt(dx**2 + dy**2 + dz**2)
    total_length = np.sum(segment_lengths)
    st.write(f"Approximate knot length: {total_length:.2f}")

    st.markdown("""
    **Torus Knot Explanation:**
    - p: Number of times the knot winds around the torus longitudinally
    - q: Number of times the knot winds around the torus meridionally
    - The (2,3) knot is the famous trefoil knot
    - These knots have applications in topology, physics, and biology
    """)
