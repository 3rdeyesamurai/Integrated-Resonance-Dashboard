import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

def torus_knot_visualizer():
    st.header("Torus Knot Visualizer")
    st.markdown("Render parametric torus knots and animate windings around the torus.")

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
