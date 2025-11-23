from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from SOS_interactions import (
    fully_connected_interactions,
    topological_neighbor_interactions,
)
from lattice import Lattice
from tqdm import tqdm
'''
AI Disclaimer:
ChatGPT was used to help the following parts of this code:
    -Visualization and Plotting Functions
    -generate_save_path() function
    -sweep_parameters() function

    
Everything else was wrritten by hand by me
'''


def _sample_clustered_headings(N, base_direction, initial_radius=1.0, rng=None):
    """Sample N headings clustered around a given base_direction.

    initial_radius in [0, 1]:
        - 1.0: headings are uniformly distributed on the sphere.
        - 0.0: all headings exactly aligned with base_direction.
        - values in between: headings confined to a spherical cap.
    """
    if rng is None:
        rng = np.random.default_rng()

    base = np.asarray(base_direction, dtype=float)
    base /= np.linalg.norm(base)

    headings = []
    for _ in range(N):
        if initial_radius >= 0.9999:
            vec = rng.normal(size=3)
            vec /= np.linalg.norm(vec)
        elif initial_radius <= 0.0001:
            vec = base.copy()
        else:
            v = rng.normal(size=3)
            v -= v.dot(base) * base
            if np.linalg.norm(v) < 1e-12:
                v = np.array([1.0, 0.0, 0.0])
            v /= np.linalg.norm(v)

            max_angle = initial_radius * np.pi
            angle = rng.uniform(0, max_angle)
            vec = np.cos(angle) * base + np.sin(angle) * v
            vec /= np.linalg.norm(vec)

        headings.append(vec)

    return np.array(headings)


# random bird alignment on the unit sphere (single swarm)
def create_sample(x=5, y=5, z=5, initial_radius=1.0, rng=None):
    """Create N random unit headings, optionally clustered by an initial radius.

    This uses a default base direction of [0, 0, 1].
    """
    N = x * y * z
    base = np.array([0.0, 0.0, 1.0])
    return _sample_clustered_headings(N, base, initial_radius=initial_radius, rng=rng)


def create_two_swarm_sample(
    x=5,
    y=5,
    z=5,
    initial_radius_a=0.3,
    initial_radius_b=0.3,
    direction_a=(0.0, 0.0, 1.0),
    direction_b=(0.0, 0.0, -1.0),
    frac_a=0.5,
    rng=None,
):
    """Create headings for two interacting swarms mixed on the lattice.

    Parameters
    ----------
    x, y, z : int
        Lattice dimensions; total birds N = x*y*z.
    initial_radius_a, initial_radius_b : float in [0, 1]
        Cluster radii for swarm A and B.
    direction_a, direction_b : 3-tuples
        Mean heading directions for swarm A and B.
    frac_a : float in [0, 1]
        Fraction of birds belonging to swarm A (rest are swarm B).

    Returns
    -------
    headings : (N, 3) array
        Mixed headings; assignment of A/B to lattice sites is random.
    labels : (N,) array of ints (0 or 1)
        0 for swarm A, 1 for swarm B, in the same order as headings.
    """
    if rng is None:
        rng = np.random.default_rng()

    N = x * y * z
    n_a = int(round(frac_a * N))
    n_b = N - n_a

    dir_a = np.asarray(direction_a, dtype=float)
    dir_b = np.asarray(direction_b, dtype=float)

    headings_a = _sample_clustered_headings(n_a, dir_a, initial_radius=initial_radius_a, rng=rng)
    headings_b = _sample_clustered_headings(n_b, dir_b, initial_radius=initial_radius_b, rng=rng)

    headings_all = np.vstack([headings_a, headings_b])
    labels_all = np.concatenate([np.zeros(n_a, dtype=int), np.ones(n_b, dtype=int)])

    perm = rng.permutation(N)
    headings_mixed = headings_all[perm]
    labels_mixed = labels_all[perm]

    return headings_mixed, labels_mixed


def sample_plot(lattice, headings):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot vectors (arrows) from the origin to each heading
    for i in range(lattice.x):
        for j in range(lattice.y):
            for k in range(lattice.z):
                pos = lattice.lattice[i, j, k]
                ax.quiver(
                    0, 0, 0,
                    pos[0], pos[1], pos[2],
                    length=1.0, normalize=True,
                    color='b', linewidth=0.5
                )


    # create unit sphere
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    u, v = np.meshgrid(u, v)
    x = np.sin(v) * np.cos(u)
    y = np.sin(v) * np.sin(u)
    z = np.cos(v)

    # plot semi-transparent sphere
    ax.plot_surface(x, y, z, color='cyan', alpha=0.2, linewidth=0)

    # set equal aspect and limits around origin
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    plt.show()



def construct_dynamics_func(interactions):
    def SOS_dynamics(headings):
        N = len(headings)
        heading_influence = np.zeros_like(headings)
        for i in range(N):
            for j in interactions[i]:
                heading_influence[i] += interactions[i][j] * (headings[j] - np.dot(headings[j], headings[i]) * headings[i])
        return heading_influence
    return SOS_dynamics




def SOS_sim(interactions, init_cons, steps=1000, t_span=(0.0, 1.0)):
    dynamics_func = construct_dynamics_func(interactions)
    def F(t, y):
        headings = y.reshape((init_cons.shape[0], 3))
        dheadings = dynamics_func(headings)
        return dheadings.flatten()
    t_eval = np.linspace(t_span[0], t_span[1], steps)
    y0 = init_cons.flatten()
    sol = solve_ivp(F, t_span, y0, t_eval=t_eval)
    return sol


def mean_pairwise_distance(headings):
    N = headings.shape[0]
    # Compute all pairwise difference vectors using broadcasting: (N, N, 3)
    diffs = headings[:, None, :] - headings[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    # Take upper triangle i < j to avoid double counting and zeros on diagonal
    i_upper, j_upper = np.triu_indices(N, k=1)
    if len(i_upper) == 0:
        return 0.0
    return dists[i_upper, j_upper].mean()


def plot_convergence(sol, step_skip=1, conditions=None, save_path=None, show=True):
    """Plot convergence of the mean pairwise distance over time.

    Parameters
    ----------
    sol : OdeSolution
        Result from SOS_sim (scipy.integrate.solve_ivp).
    step_skip : int, optional
        Use every `step_skip`-th time step when computing the statistic.
    conditions : dict or None, optional
        Optional dictionary of simulation conditions to annotate on the plot.
    """

    # sol.y has shape (3N, T); transpose to (T, 3N)
    y = sol.y.T
    T, dim = y.shape
    N = dim // 3

    # Optionally subsample time steps
    idx = np.arange(0, T, step_skip)
    y = y[idx]
    t_vals = sol.t[idx]

    # Reshape to (T_sub, N, 3)
    headings_t = y.reshape(len(t_vals), N, 3)

    # Compute mean pairwise distance at each sampled time
    mpd = np.array([mean_pairwise_distance(H) for H in headings_t])

    return t_vals, mpd


def plot_convergence_figure(t_vals, mpd, conditions=None, save_path=None, show=True):
    """Create a convergence plot from precomputed mean pairwise distances.

    This helper lets us reuse the same styling when sweeping parameters,
    without recomputing distances multiple times.
    """

    # Plot
    plt.figure()
    plt.plot(t_vals, mpd, marker='o', markersize=3, linewidth=1)
    plt.xlabel('Time')
    plt.ylabel('Mean pairwise distance')
    plt.title('Convergence of mean pairwise distance')
    if conditions is not None:
        cond_str = ", ".join(f"{k}={v}" for k, v in conditions.items())
        plt.text(0.05, 0.95, cond_str,
                 transform=plt.gca().transAxes,
                 fontsize=8,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # draw horizontal line at y=0 and ensure it's within the y-limits
    plt.axhline(0.0, color='gray', linestyle='--', linewidth=0.8, zorder=0)
    ymin, ymax = plt.ylim()
    plt.ylim(min(ymin, 0.0), max(ymax, 0.0))
    if save_path is not None:
        plt.savefig(save_path)
    if(show): plt.show()
    plt.close()
    

def plot_solution(sol, step_skip=1, conditions=None, save_path=None):
    """Animate the evolution of headings on the unit sphere.

    Parameters
    ----------
    sol : OdeSolution
        Result from SOS_sim (scipy.integrate.solve_ivp).
    step_skip : int, optional
        Use every `step_skip`-th time step to speed up the animation.
    save_path : str or None, optional
        If provided, save the animation to this path (e.g. 'sos.mp4').
    """

    # sol.y has shape (3N, T); transpose to (T, 3N)
    y = sol.y.T
    T, dim = y.shape
    N = dim // 3

    # Optionally subsample time steps for speed
    idx = np.arange(0, T, step_skip)
    y = y[idx]
    t_vals = sol.t[idx]
    T_sub = len(t_vals)

    # Headings over time: (T_sub, N, 3)
    headings_t = y.reshape(T_sub, N, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create unit sphere surface (static background)
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    u, v = np.meshgrid(u, v)
    xs = np.sin(v) * np.cos(u)
    ys = np.sin(v) * np.sin(u)
    zs = np.cos(v)
    ax.plot_surface(xs, ys, zs, color='cyan', alpha=0.2, linewidth=0)

    # Initial headings
    H0 = headings_t[0]
    n = N
    origins = np.zeros(n)

    # Initial quiver for arrows
    quiv = ax.quiver(
        origins, origins, origins,
        H0[:, 0], H0[:, 1], H0[:, 2],
        length=1.0, normalize=False,
        color='tab:blue', linewidth=0.5,
    )

    # Scatter endpoints
    scat = ax.scatter(H0[:, 0], H0[:, 1], H0[:, 2], color='k', s=10)

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    # Optional text box showing conditions in the 3D plot
    if conditions is not None:
        cond_str = ", ".join(f"{k}={v}" for k, v in conditions.items())
        ax.text2D(0.05, 0.95, cond_str,
                  transform=ax.transAxes,
                  fontsize=8,
                  verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.set_title(f"t = {t_vals[0]:.2f}")

    def update(frame):
        nonlocal quiv
        H = headings_t[frame]

        # Remove old quiver and draw a new one
        quiv.remove()
        quiv = ax.quiver(
            origins, origins, origins,
            H[:, 0], H[:, 1], H[:, 2],
            length=1.0, normalize=False,
            color='tab:blue', linewidth=0.5,
        )

        # Update scatter points
        scat._offsets3d = (H[:, 0], H[:, 1], H[:, 2])

        ax.set_title(f"t = {t_vals[frame]:.2f}")
        return quiv, scat

    anim = FuncAnimation(
        fig, update, frames=T_sub, interval=50, blit=False
    )

    if save_path is not None:
        anim.save(save_path, fps=20)
    else:
        plt.show()

def generate_save_path(conditions, extension="gif"):
    parts = [f"{k}-{v}" for k, v in conditions.items()]
    filename = "results\\sos_" + "_".join(parts) + f".{extension}"
    return filename

def sweep_parameters(param_grid, save_anim= False):
    """Sweep over a grid of parameters and run simulations.

    Parameters
    ----------
    param_grid : list[dict]
        List of dictionaries, each specifying a set of conditions
        (e.g. {"x":5, "y":5, "z":5, "initial_radius":0.5, "order":1, "weight":1.0}).
    save_anim : bool, optional
        If True, save animations as well as convergence plots.
    """

    all_curves = []  # to build combined overlay figure at the end

    for params in tqdm(param_grid):
        # Basic lattice/initial condition parameters with defaults
        x = params.get("x", 5)
        y = params.get("y", 5)
        z = params.get("z", 5)
        init_radius = params.get("initial_radius", 0.5)
        order = params.get("order", 1)
        weight = params.get("weight", 1.0)
        steps = params.get("steps", 200)

        conditions = {
            "x": x,
            "y": y,
            "z": z,
            "initial_radius": init_radius,
            "order": order,
            "weight": weight,
        }

        # Build lattice and initial headings
        lat = Lattice(x, y, z, spacing=1.0)
        headings = create_sample(x=x, y=y, z=z, initial_radius=init_radius)
        lat.populate_lattice(headings)

        # Interactions (currently using topological neighbours; adjust if needed)
        interactions = topological_neighbor_interactions(lat, order=order, weight=weight)

        # Run simulation
        sol = SOS_sim(interactions, init_cons=headings, steps=steps)

        # Compute convergence curve (time + mean pairwise distance)
        t_vals, mpd = plot_convergence(sol, step_skip=1, conditions=None, save_path=None, show=False)

        # Store for overlay plot
        all_curves.append((t_vals, mpd, conditions))

        # Individual file paths
        anim_path = generate_save_path(conditions, extension="gif") if save_anim else None
        conv_path = generate_save_path(conditions, extension="png")

        # Individual convergence plot
        plot_convergence_figure(t_vals, mpd, conditions=conditions, save_path=conv_path, show=False)
        if save_anim:
            plot_solution(sol, step_skip=5, conditions=conditions, save_path=anim_path)

    # Create combined overlay figure for all conditions
    if all_curves:
        plt.figure()
        for t_vals, mpd, cond in all_curves:
            label = ", ".join(f"{k}={v}" for k, v in cond.items())
            plt.plot(t_vals, mpd, linewidth=1, label=label)

        plt.xlabel('Time')
        plt.ylabel('Mean pairwise distance')
        plt.title('Convergence comparison across conditions')
        plt.grid(True, alpha=0.3)
        plt.axhline(0.0, color='gray', linestyle='--', linewidth=0.8, zorder=0)
        plt.tight_layout()
        plt.legend(fontsize=6)

        # Save a generic combined figure
        plt.savefig('results\\sos_convergence_overlay.png')
        plt.close()

def create_param_grid(n_range, initial_radius_values, order_values, weight_values):
    grid = []
    for n in n_range:
        for init_radius in initial_radius_values:
            for order in order_values:
                for weight in weight_values:
                    grid.append({
                        "x": n,
                        "y": n,
                        "z": n,
                        "initial_radius": init_radius,
                        "order": order,
                        "weight": weight
                    })
    return grid


def t1():
    x = 5
    y = 5
    z = 5
    init_radius = 0.5

    conditions = {
        'x': x,
        'y': y,
        'z': z,
        'initial_radius': init_radius
    }

    anim_save_path = generate_save_path(conditions)
    conv_save_path = generate_save_path(conditions, extension="png")
   

    lat = Lattice(x, y, z, spacing=1.0)
    headings = create_sample(x=x, y=y, z=z, initial_radius=init_radius)
    lat.populate_lattice(headings)
    interactions = topological_neighbor_interactions(lat, order=1, weight=1.0)
    sol = SOS_sim(interactions, init_cons=headings, steps=200, t_span=(0.0, 5.0))
    plot_convergence(sol,conditions=conditions, save_path=conv_save_path)
    plot_solution(sol, step_skip=5, conditions=conditions, save_path=anim_save_path)

def t2():
    n_range = [5,7,9]
    initial_radius_values = [0.3,0.5,1]
    order_values = [1,2]
    weight_values = [1, round(1/6, 2)]


    grid = create_param_grid(n_range, initial_radius_values, order_values, weight_values)
    grid_b = [
        #order 1 weights
        {"x":5, "y":5, "z":5, "initial_radius":0.3, "order":1, "weight":1},
        {"x":5, "y":5, "z":5, "initial_radius":0.5, "order":1, "weight":1},
        {"x":5, "y":5, "z":5, "initial_radius":1.0, "order":1, "weight":1},

        {"x":7, "y":7, "z":7, "initial_radius":0.3, "order":1, "weight":1},
        {"x":7, "y":7, "z":7, "initial_radius":0.5, "order":1, "weight":1},
        {"x":7, "y":7, "z":7, "initial_radius":1.0, "order":1, "weight":1},

        {"x":9, "y":9, "z":9, "initial_radius":0.3, "order":1, "weight":1},
        {"x":9, "y":9, "z":9, "initial_radius":0.5, "order":1, "weight":1},
        {"x":9, "y":9, "z":9, "initial_radius":1.0, "order":1, "weight":1},
        
        #order 2 weights
        {"x":5, "y":5, "z":5, "initial_radius":0.3, "order":2, "weight":round(1/6, 2)},
        {"x":5, "y":5, "z":5, "initial_radius":0.5, "order":2, "weight":round(1/6, 2)},
        {"x":5, "y":5, "z":5, "initial_radius":1.0, "order":2, "weight":round(1/6, 2)},

        {"x":7, "y":7, "z":7, "initial_radius":0.3, "order":2, "weight":round(1/6, 2)},
        {"x":7, "y":7, "z":7, "initial_radius":0.5, "order":2, "weight":round(1/6, 2)},
        {"x":7, "y":7, "z":7, "initial_radius":1.0, "order":2, "weight":round(1/6, 2)},

        {"x":9, "y":9, "z":9, "initial_radius":0.3, "order":2, "weight":round(1/6, 2)},
        {"x":9, "y":9, "z":9, "initial_radius":0.5, "order":2, "weight":round(1/6, 2)},
        {"x":9, "y":9, "z":9, "initial_radius":1.0, "order":2, "weight":round(1/6, 2)},
    ]
    sweep_parameters(grid_b, save_anim=False)

def two_swarm_test():
    x = y = z = 5
    steps = 200

    # Example: two opposite headings with small initial radii
    headings, labels = create_two_swarm_sample(
        x=x,
        y=y,
        z=z,
        initial_radius_a=0.2,
        initial_radius_b=0.2,
        direction_a=(0.0, 0.0, 1.0),
        direction_b=(0.0, 0.0, -1.0),
        frac_a=0.5,
    )

    lat = Lattice(x, y, z, spacing=1.0)
    lat.populate_lattice(headings)

    interactions = topological_neighbor_interactions(lat, order=1, weight=1.0)
    sol = SOS_sim(interactions, init_cons=headings, steps=steps, t_span=(0.0, 5.0))

    conditions = {
        "x": x,
        "y": y,
        "z": z,
        "two_swarms": True,
        "frac_a": 0.5,
        "init_rad_a": 0.2,
        "init_rad_b": 0.2,
    }

    t_vals, mpd = plot_convergence(sol, step_skip=1, conditions=None, save_path=None, show=True)
    conv_path = generate_save_path(conditions, extension="png")
    plot_convergence_figure(t_vals, mpd, conditions=conditions, save_path=conv_path, show=True)

    anim_path = generate_save_path(conditions, extension="gif")
    plot_solution(sol, step_skip=5, conditions=conditions, save_path=anim_path)


def visualize_initial_headings_3d(
    x=5,
    y=5,
    z=5,
    two_swarms=True,
):
    """Quick diagnostic plot: lattice points + heading vectors in 3D.

    If two_swarms is True, use the two-swarm initial condition; otherwise
    use a single fully random swarm.
    """

    if two_swarms:
        headings, labels = create_two_swarm_sample(
            x=x,
            y=y,
            z=z,
            initial_radius_a=0.2,
            initial_radius_b=0.2,
            direction_a=(0.0, 0.0, 1.0),
            direction_b=(0.0, 0.0, -1.0),
            frac_a=0.5,
        )
    else:
        headings = create_sample(x=x, y=y, z=z, initial_radius=1.0)
        labels = None

    # Build lattice and assign headings to each lattice site
    lat = Lattice(x, y, z, spacing=1.0)
    lat.populate_lattice(headings)

    # Extract positions (from indices) and headings (from stored data)
    positions_list = []
    heading_list = []

    for i in range(lat.x):
        for j in range(lat.y):
            for k in range(lat.z):
                pos = np.array([i, j, k], dtype=float) * lat.spacing
                h = np.array(lat.lattice[i, j, k], dtype=float)
                positions_list.append(pos)
                heading_list.append(h)

    positions = np.vstack(positions_list)   # (N, 3)
    H = np.vstack(heading_list)            # (N, 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot lattice points
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        color="k",
        s=20,
        alpha=0.6,
    )

    # Small arrows from each lattice point in heading direction
    ax.quiver(
        positions[:, 0], positions[:, 1], positions[:, 2],
        H[:, 0], H[:, 1], H[:, 2],
        length=0.3,
        normalize=True,
        color="tab:blue",
        linewidth=1,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(
        "Initial headings on lattice"
        + (" (two swarms)" if two_swarms else " (single swarm)")
    )

    # Make axes roughly equal for better visual
    max_range = (positions.max(axis=0) - positions.min(axis=0)).max()
    mid = positions.mean(axis=0)
    for axis, m in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], mid):
        axis(m - max_range / 2, m + max_range / 2)

    plt.show()


def overlay_two_swarm_vs_random(
    x=5,
    y=5,
    z=5,
    steps=200,
    t_span=(0.0, 5.0),
    initial_radius_two_swarm_a=0.2,
    initial_radius_two_swarm_b=0.2,
    frac_a=0.5,
):
    """Overlay convergence for two scenarios:

    1) Two interacting swarms (opposite headings, small initial radii).
    2) Single swarm with completely random headings (initial_radius = 1.0).

    Saves a single PNG in `results/` with both curves.
    """

    # --- Scenario 1: two interacting swarms ---
    headings_two, labels = create_two_swarm_sample(
        x=x,
        y=y,
        z=z,
        initial_radius_a=initial_radius_two_swarm_a,
        initial_radius_b=initial_radius_two_swarm_b,
        direction_a=(0.0, 0.0, 1.0),
        direction_b=(0.0, 0.0, -1.0),
        frac_a=frac_a,
    )

    lat_two = Lattice(x, y, z, spacing=1.0)
    lat_two.populate_lattice(headings_two)
    interactions_two = topological_neighbor_interactions(lat_two, order=1, weight=1.0)
    sol_two = SOS_sim(interactions_two, init_cons=headings_two, steps=steps, t_span=t_span)
    t_two, mpd_two = plot_convergence(sol_two, step_skip=1, conditions=None, save_path=None, show=False)

    # --- Scenario 2: single fully random swarm ---
    headings_rand = create_sample(x=x, y=y, z=z, initial_radius=1.0)
    lat_rand = Lattice(x, y, z, spacing=1.0)
    lat_rand.populate_lattice(headings_rand)
    interactions_rand = topological_neighbor_interactions(lat_rand, order=1, weight=1.0)
    sol_rand = SOS_sim(interactions_rand, init_cons=headings_rand, steps=steps, t_span=t_span)
    t_rand, mpd_rand = plot_convergence(sol_rand, step_skip=1, conditions=None, save_path=None, show=False)

    # --- Overlay figure ---
    plt.figure()
    plt.plot(t_two, mpd_two, label="two swarms (opposite, clustered)", linewidth=1.5)
    plt.plot(t_rand, mpd_rand, label="single swarm (fully random)", linewidth=1.5)
    plt.xlabel("Time")
    plt.ylabel("Mean pairwise distance")
    plt.title("Convergence: two-swarm vs random single swarm")
    plt.grid(True, alpha=0.3)
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, zorder=0)
    plt.tight_layout()
    plt.legend(fontsize=8)

    save_path = "results\\sos_two_swarm_vs_random_" + str(time()) + ".png"
    plt.savefig(save_path)
    plt.show()
    plt.close()

if __name__ == "__main__":
    start_time = time()
    t1()
    #two_swarm_test()
    #overlay_two_swarm_vs_random(x=9, y=9, z=9, steps=600, t_span=(0.0, 10.0))
    #visualize_initial_headings_3d(x=5, y=5, z=5, two_swarms=False)


    print()
    print("---------------------------------------------------------")
    print("DONE")
    print("RUN_TIME: {:.2f} seconds".format(time() - start_time))
    

