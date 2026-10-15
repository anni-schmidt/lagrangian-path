"""
Streamlines in time, and average particle tracks
=================================================

Two plots for the same velocity fields the Lagrangian tracker uses
(unstructured meshes, and ROMS/DOPPIO or HYCOM grids via the tracker's
readers, including a chosen depth):

1. <prefix>_streamlines.png  -- the flow as streamlines at several moments
   (panels), coloured by speed on one shared scale so panels compare
   directly. Optionally <prefix>_streamlines.gif, an animation through time.
   Streamlines are of the *instantaneous* velocity at each time, so they
   show how the flow pattern changes, not where particles go.

2. <prefix>_mean_tracks.png  -- "average" particle tracks. Small clusters
   of particles (N x N, a few km apart) are released over a coarse grid of
   locations and advected through the time-varying field; each cluster's
   mean path is drawn, coloured by elapsed time, over the time-mean speed.
   The right-hand panel shows each cluster's spread (RMS distance of its
   particles from their mean) against time: a mean track only represents
   its cluster while that spread stays small. A mean path stops if any
   particle of its cluster leaves the mesh.

Usage
-----
    python flow_plots.py --synthetic            # time-dependent double gyre demo

    python flow_plots.py --source fvcom_out.nc --bbox -71.0 -69.5 41.5 42.5 \\
        --time 2026-08-19T00:00 --duration-hours 72

    python flow_plots.py --grid-type roms --source <DOPPIO url or .nc> \\
        --bbox -74.5 -72.5 38.5 40.0 --time 2020-01-05T00:00 --depth 20 --animate

Python:
    import flow_plots as fp
    fp.plot_streamlines_in_time(field, bbox, times, "streams.png")
    tracks = fp.compute_mean_tracks(field, bbox, "2020-01-05T00:00", 72)
    fp.plot_mean_tracks(tracks, field, bbox, "tracks.png")
"""

import argparse
import os
import sys
import warnings
import numpy as np

import lagrangian_tracker as lt
import lyapunov as ly

R_EARTH = lt.R_EARTH
DEG = np.pi / 180.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _regular_grid(bbox, n_points):
    """Regularly spaced lon/lat vectors covering bbox with ~n_points along
    the longer side (spacing roughly equal in metres in both directions)."""
    lon_min, lon_max, lat_min, lat_max = bbox
    kx = R_EARTH * np.cos(0.5 * (lat_min + lat_max) * DEG) * DEG
    ky = R_EARTH * DEG
    w, h = (lon_max - lon_min) * kx, (lat_max - lat_min) * ky
    step = max(w, h) / max(n_points - 1, 1)
    nx, ny = max(int(round(w / step)) + 1, 3), max(int(round(h / step)) + 1, 3)
    return np.linspace(lon_min, lon_max, nx), np.linspace(lat_min, lat_max, ny)


def sample_on_grid(field, lon_g, lat_g, when, depth=None):
    """Velocity (m/s) on a regular lon/lat grid at one moment; NaN outside
    the mesh. `when` is anything np.datetime64 accepts."""
    t = float((np.datetime64(when) - field.times[0]) / np.timedelta64(1, "s"))
    LON, LAT = np.meshgrid(lon_g, lat_g)
    u, v = ly.sample_velocity(field, LON.ravel(), LAT.ravel(), t, depth)
    return u.reshape(LON.shape), v.reshape(LON.shape)


def _check_record(field, t_a, t_b):
    lo, hi = field.times[0], field.times[-1]
    if np.datetime64(t_a) < lo or np.datetime64(t_b) > hi:
        raise ValueError(f"Requested {t_a} .. {t_b} is outside the velocity record "
                         f"{lo} .. {hi}")


def _speed_norm(speeds, floor=1e-6):
    from matplotlib.colors import Normalize
    vals = np.concatenate([s[np.isfinite(s)].ravel() for s in speeds])
    vmax = float(np.percentile(vals, 99)) if vals.size else 1.0
    return Normalize(vmin=0.0, vmax=max(vmax, floor))


# ---------------------------------------------------------------------------
# 1. streamlines in time
# ---------------------------------------------------------------------------

def _draw_streamlines(ax, lon_g, lat_g, u, v, norm, density, cmap="viridis"):
    """Streamlines of (u, v) m/s on a lon/lat axes. Geometry uses the
    velocity converted to degrees/s so lines follow the true flow direction
    on the (equal-metre) map; colour and width show the speed in m/s."""
    LAT = np.meshgrid(lon_g, lat_g)[1]
    speed = np.hypot(u, v)
    ud = u / (R_EARTH * np.cos(LAT * DEG)) / DEG
    vd = v / R_EARTH / DEG
    bad = ~(np.isfinite(ud) & np.isfinite(vd))
    ud, vd = np.ma.array(ud, mask=bad), np.ma.array(vd, mask=bad)
    if bad.all():
        return None
    lw = 0.4 + 1.4 * np.clip(np.nan_to_num(speed) / norm.vmax, 0, 1)
    return ax.streamplot(lon_g, lat_g, ud, vd, color=np.ma.array(speed, mask=bad), cmap=cmap,
                         norm=norm, linewidth=lw, density=density, arrowsize=0.9)


def plot_streamlines_in_time(field, bbox, times, out_path, n_points=100, depth=None,
                             density=1.6, ncols=3):
    """One streamline panel per time in `times`, on one shared speed scale."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable

    times = [np.datetime64(t) for t in times]
    _check_record(field, min(times), max(times))
    lon_g, lat_g = _regular_grid(bbox, n_points)
    snaps = [sample_on_grid(field, lon_g, lat_g, t, depth) for t in times]
    norm = _speed_norm([np.hypot(u, v) for u, v in snaps])

    n = len(times)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    mean_lat = 0.5 * (bbox[2] + bbox[3])
    aspect = ((bbox[3] - bbox[2]) / ((bbox[1] - bbox[0]) * np.cos(np.deg2rad(mean_lat))))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.6 * ncols + 1.2, nrows * (5.6 * aspect + 1.0) + 0.6),
                             squeeze=False, constrained_layout=True)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax, t, (u, v) in zip(axes.ravel(), times, snaps):
        ax.set_facecolor("0.86")                       # outside the mesh
        _draw_streamlines(ax, lon_g, lat_g, u, v, norm, density)
        hrs = (t - times[0]) / np.timedelta64(1, "h")
        ax.set_title(f"{str(t)[:16]}   (+{hrs:g} h)", fontsize=10)
        ax.set_xlim(bbox[0], bbox[1])
        ax.set_ylim(bbox[2], bbox[3])
        ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
    fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=axes.ravel().tolist(),
                 shrink=0.8, label="speed (m/s)")
    fig.suptitle("Instantaneous streamlines" + (f" at {depth:g} m depth" if depth is not None else ""),
                 fontsize=12)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Saved streamline panels to {out_path}")


def animate_streamlines(field, bbox, times, out_path, n_points=90, depth=None,
                        density=1.5, fps=4):
    """GIF of the instantaneous streamlines through `times` (speed scale
    fixed over the whole animation)."""
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.cm import ScalarMappable

    times = [np.datetime64(t) for t in times]
    _check_record(field, min(times), max(times))
    lon_g, lat_g = _regular_grid(bbox, n_points)
    snaps = [sample_on_grid(field, lon_g, lat_g, t, depth) for t in times]
    norm = _speed_norm([np.hypot(u, v) for u, v in snaps])
    mean_lat = 0.5 * (bbox[2] + bbox[3])

    fig, ax = plt.subplots(figsize=(7.4, 5.4), constrained_layout=True)
    fig.colorbar(ScalarMappable(norm=norm, cmap="viridis"), ax=ax, label="speed (m/s)")

    def draw(k):
        ax.clear()
        ax.set_facecolor("0.86")
        u, v = snaps[k]
        _draw_streamlines(ax, lon_g, lat_g, u, v, norm, density)
        hrs = (times[k] - times[0]) / np.timedelta64(1, "h")
        ax.set_title(f"{str(times[k])[:16]}   (+{hrs:g} h)")
        ax.set_xlim(bbox[0], bbox[1])
        ax.set_ylim(bbox[2], bbox[3])
        ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")

    anim = animation.FuncAnimation(fig, draw, frames=len(times), interval=1000 / fps)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    anim.save(out_path, writer=animation.PillowWriter(fps=fps), dpi=100)
    plt.close(fig)
    print(f"Saved streamline animation to {out_path} ({len(times)} frames)")


# ---------------------------------------------------------------------------
# 2. average particle tracks
# ---------------------------------------------------------------------------

def compute_mean_tracks(field, bbox, t_ref, hours, direction="forward", n_x=6, n_y=None,
                        cluster_size=3, spacing_km=None, dt_seconds=1800.0, depth=None):
    """
    Mean tracks of particle clusters released over a coarse grid.

    Returns a dict: mean_lon/mean_lat (steps, n_clusters) -- NaN once any
    member has left the mesh; spread_km (steps, n_clusters) -- RMS distance
    of the members from their mean; end_lon/end_lat (n_clusters, members);
    elapsed_h (steps,), plus the settings.
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    kx = R_EARTH * np.cos(0.5 * (lat_min + lat_max) * DEG) * DEG
    ky = R_EARTH * DEG
    if spacing_km is None:                      # 1% of the longer box side
        spacing_km = 0.01 * max((lon_max - lon_min) * kx, (lat_max - lat_min) * ky) / 1000.0
    cx, cy, mem_lon, mem_lat = ly.cluster_seeds(bbox, n_x, n_y, cluster_size, spacing_km * 1000.0)
    plon, plat, elapsed = ly.advect_paths(field, mem_lon.ravel(), mem_lat.ravel(), t_ref, hours,
                                          direction, dt_seconds, depth)
    steps, ns, m = plon.shape[0], len(cx), cluster_size ** 2
    plon, plat = plon.reshape(steps, ns, m), plat.reshape(steps, ns, m)
    with np.errstate(invalid="ignore"):
        mean_lon, mean_lat = plon.mean(axis=2), plat.mean(axis=2)
        d = ly._pair_distance(mean_lon[:, :, None], mean_lat[:, :, None], plon, plat)
        spread = np.sqrt((d ** 2).mean(axis=2)) / 1000.0
    return dict(mean_lon=mean_lon, mean_lat=mean_lat, spread_km=spread,
                end_lon=plon[-1], end_lat=plat[-1], elapsed_h=elapsed / 3600.0,
                start_lon=cx, start_lat=cy, direction=direction, cluster_size=cluster_size,
                spacing_km=spacing_km, hours=hours, t_ref=np.datetime64(t_ref), depth=depth)


def plot_mean_tracks(tracks, field, bbox, out_path, n_points=90, n_bg_times=6):
    """Mean tracks coloured by elapsed time over the time-mean speed, with
    each cluster's spread against time alongside."""
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize

    sign = 1.0 if tracks["direction"] == "forward" else -1.0
    t0 = tracks["t_ref"]
    span = np.timedelta64(int(tracks["hours"] * 3600), "s")
    moments = [t0 + sign * span * f for f in np.linspace(0, 1, n_bg_times)]
    lon_g, lat_g = _regular_grid(bbox, n_points)
    speeds = [np.hypot(*sample_on_grid(field, lon_g, lat_g, t, tracks["depth"])) for t in moments]
    with warnings.catch_warnings():               # all-NaN cells (outside the mesh) are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_speed = np.nanmean(np.stack(speeds), axis=0)

    show_spread = tracks["cluster_size"] > 1
    fig, axes = plt.subplots(1, 2 if show_spread else 1, squeeze=False,
                             figsize=(13.5 if show_spread else 7.6, 5.6),
                             gridspec_kw=dict(width_ratios=[2.1, 1]) if show_spread else None)
    ax = axes[0, 0]
    ax.set_facecolor("0.86")
    ax.pcolormesh(lon_g, lat_g, mean_speed, cmap="Greys", vmin=0,
                  vmax=np.nanpercentile(mean_speed, 99) * 1.6, shading="auto", alpha=0.65)

    norm = Normalize(0, tracks["hours"])
    segs, vals = [], []
    for i in range(tracks["mean_lon"].shape[1]):
        lo, la = tracks["mean_lon"][:, i], tracks["mean_lat"][:, i]
        ok = np.isfinite(lo) & np.isfinite(la)
        n_ok = int(ok.sum())
        if n_ok < 2:
            continue
        pts = np.column_stack([lo[:n_ok], la[:n_ok]])
        segs.append(np.stack([pts[:-1], pts[1:]], axis=1))
        vals.append(0.5 * (tracks["elapsed_h"][:n_ok - 1] + tracks["elapsed_h"][1:n_ok]))
        ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=6,
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=0.8, mutation_scale=9,
                                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")]))
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_aspect(1.0 / np.cos(np.deg2rad(0.5 * (bbox[2] + bbox[3]))))
    if segs:
        lc = LineCollection(np.concatenate(segs), cmap="plasma", norm=norm, linewidths=2.0,
                            path_effects=[pe.withStroke(linewidth=3.6, foreground="white")], zorder=4)
        lc.set_array(np.concatenate(vals))
        ax.add_collection(lc)
    ax.scatter(tracks["start_lon"], tracks["start_lat"], s=26, facecolor="white", edgecolor="k",
               linewidths=0.9, zorder=7, label="start")
    if show_spread:
        ax.scatter(tracks["end_lon"], tracks["end_lat"], s=5, color="0.25", alpha=0.5,
                   linewidths=0, zorder=5, label="end positions of the cluster's particles")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=8, frameon=False)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    m = tracks["cluster_size"]
    what = (f"mean of {m}x{m}-particle clusters ({tracks['spacing_km']:.1f} km apart)"
            if show_spread else "single particles")
    ax.set_title(f"{tracks['direction']} tracks: {what}\n{tracks['hours']:g} h from "
                 f"{str(t0)[:16]}" + (f", {tracks['depth']:g} m" if tracks["depth"] is not None else "")
                 + "   (grey: time-mean speed)", fontsize=9.5)

    if show_spread:
        ax2 = axes[0, 1]
        sp = tracks["spread_km"]
        for i in range(sp.shape[1]):
            ax2.plot(tracks["elapsed_h"], sp[:, i], color="0.55", lw=0.8, alpha=0.7)
        with np.errstate(invalid="ignore"):
            ax2.plot(tracks["elapsed_h"], np.nanmedian(sp, axis=1), color="crimson", lw=2.2,
                     label="median over clusters")
        ax2.axhline(float(np.nanmedian(sp[0])), color="k", ls="--", lw=1,
                    label="initial spread")
        ax2.set_xlabel("hours since release" if sign > 0 else "hours before t_ref")
        ax2.set_ylabel("cluster spread: RMS distance from mean (km)")
        ax2.set_title("Where the mean track is representative:\nspread stays small", fontsize=9.5)
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8)
        ax2.set_yscale("log")
    plt.tight_layout()
    if show_spread and segs:                      # make room between the map's colourbar and the spread panel
        p2 = axes[0, 1].get_position()
        axes[0, 1].set_position([p2.x0 + 0.06, p2.y0, p2.width - 0.06, p2.height])
    if segs:
        # place the colourbar from the map's *actual* (aspect-adjusted) position so it hugs the map
        fig.canvas.draw()
        pos = ax.get_position()
        cax = fig.add_axes([pos.x1 + 0.008, pos.y0, 0.011, pos.height])
        fig.colorbar(lc, cax=cax, label="hours since release" if sign > 0
                     else "hours before t_ref (backward)")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved mean-track plot to {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", help="NetCDF/Zarr/OPeNDAP source (any grid the tracker reads)")
    p.add_argument("--synthetic", action="store_true", help="Use the time-dependent double-gyre demo")
    p.add_argument("--grid-type", default="auto",
                   choices=["auto", "unstructured", "roms", "doppio", "hycom", "rectilinear"])
    p.add_argument("--engine", default=None, choices=["zarr", "netcdf4", "h5netcdf"])
    p.add_argument("--zarr-anon", action="store_true")
    p.add_argument("--u-var", default=None)
    p.add_argument("--v-var", default=None)
    p.add_argument("--bbox", type=float, nargs=4, default=None,
                   metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"))
    p.add_argument("--load-pad", type=float, default=2.0,
                   help="Structured grids: extra degrees read around --bbox")
    p.add_argument("--time", default=None, help="Start of the window, ISO8601")
    p.add_argument("--duration-hours", type=float, default=None,
                   help="Window length (default 72; 240 for the synthetic demo)")
    p.add_argument("--depth", type=float, default=None,
                   help="Depth (m, positive down) for fields with vertical levels")
    p.add_argument("--n-panels", type=int, default=6, help="Streamline snapshots across the window")
    p.add_argument("--n-points", type=int, default=100, help="Streamline grid points on the longer side")
    p.add_argument("--density", type=float, default=1.6, help="Streamline density")
    p.add_argument("--animate", action="store_true", help="Also write a streamline GIF")
    p.add_argument("--n-frames", type=int, default=24)
    p.add_argument("--fps", type=float, default=4.0)
    p.add_argument("--direction", choices=["forward", "backward"], default="forward",
                   help="Direction of the particle tracks (backward ends at the window's end)")
    p.add_argument("--dt-seconds", type=float, default=1800.0)
    p.add_argument("--n-traj-x", type=int, default=6)
    p.add_argument("--n-traj-y", type=int, default=None)
    p.add_argument("--cluster-size", type=int, default=3,
                   help="Each track is the mean of N x N particles (1 = single particles)")
    p.add_argument("--cluster-spacing-km", type=float, default=None)
    p.add_argument("--out-prefix", default="flow")
    args = p.parse_args()

    if args.synthetic or not args.source:
        if not args.synthetic:
            print("No --source given; using the synthetic double gyre.", file=sys.stderr)
        lon, lat, tri, times, u, v = ly.synthetic_double_gyre()
        field = lt.UnstructuredVelocityField(lon, lat, tri, times, u, v)
        if args.bbox is None:
            m = 0.003
            args.bbox = (lon.min() + m, lon.max() - m, lat.min() + m, lat.max() - m)
        t_start = np.datetime64(args.time) if args.time else field.times[0]
        hours = args.duration_hours or 240.0
    else:
        if args.bbox is None or args.time is None:
            p.error("--bbox and --time are required with --source")
        t_start = np.datetime64(args.time)
        hours = args.duration_hours or 72.0
        t_end = t_start + np.timedelta64(int(hours * 3600), "s")
        args.depth = args.depth                      # (ly._load_field reads args.depth)
        field = ly._load_field(args, t_start, t_end, tuple(args.bbox))
    bbox = tuple(args.bbox)
    span = np.timedelta64(int(hours * 3600), "s")
    t_end = t_start + span

    moments = [t_start + span * f for f in np.linspace(0, 1, args.n_panels)]
    plot_streamlines_in_time(field, bbox, moments, f"{args.out_prefix}_streamlines.png",
                             n_points=args.n_points, depth=args.depth, density=args.density)
    if args.animate:
        frames = [t_start + span * f for f in np.linspace(0, 1, args.n_frames)]
        animate_streamlines(field, bbox, frames, f"{args.out_prefix}_streamlines.gif",
                            depth=args.depth, fps=args.fps)

    t_ref = t_start if args.direction == "forward" else t_end
    tracks = compute_mean_tracks(field, bbox, t_ref, hours, args.direction, args.n_traj_x,
                                 args.n_traj_y, args.cluster_size, args.cluster_spacing_km,
                                 args.dt_seconds, args.depth)
    np.savez_compressed(f"{args.out_prefix}_mean_tracks.npz",
                        **{k: tracks[k] for k in ("mean_lon", "mean_lat", "spread_km", "elapsed_h",
                                                  "start_lon", "start_lat", "end_lon", "end_lat")})
    plot_mean_tracks(tracks, field, bbox, f"{args.out_prefix}_mean_tracks.png")


if __name__ == "__main__":
    main()
