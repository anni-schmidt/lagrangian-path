"""
Finite-Time (FTLE) and Finite-Size (FSLE) Lyapunov exponents
for the same velocity fields the Lagrangian tracker uses
==============================================================

Works on any field built by lagrangian_tracker.py -- unstructured meshes
(FVCOM/SCHISM/...), and structured ROMS/DOPPIO or HYCOM grids, which that
module converts to triangle meshes -- including its depth levels.

Method
------
A regular grid of tracer particles (spacing `spacing_m`, uniform in a local
east/north plane) is seeded over a box and advected with RK4, all
particles at once, through the offline velocity field. One pass gives both:

  FTLE  sigma = ln( sqrt(lambda_max(C)) ) / |T|
        C = (grad Phi)^T (grad Phi) is the Cauchy-Green strain tensor of the
        flow map Phi over the fixed time T; grad Phi comes from central
        differences of the final particle positions.

  FSLE  lambda = ln(delta_f / delta_0) / tau
        delta_0 is the initial separation (the grid spacing) and tau the
        time it takes the largest separation between a particle and its
        four grid neighbours to grow to delta_f. Unlike the FTLE the time
        is variable and the length scale is fixed. Where the separation
        never reaches delta_f within T the value is NaN.

forward  = integrate from the reference time t0 to t0 + T
           (ridges of the field mark repelling Lagrangian coherent structures)
backward = integrate from t0 back to t0 - T
           (ridges mark attracting structures: fronts, filaments)

Results are in 1/day. Points whose particles leave the mesh (or whose
neighbours do) before the end are NaN, as are seeds outside the mesh and the
outermost ring of the grid (it has no neighbours on one side).

Usage
-----
    python lyapunov.py --synthetic                      # time-dependent double gyre demo

    python lyapunov.py --source fvcom_out.nc \\
        --bbox -71.0 -69.5 41.5 42.5 --time 2026-08-19T00:00 --integration-hours 48

    python lyapunov.py --grid-type roms --source <DOPPIO url or .nc> \\
        --bbox -74.5 -72.5 38.5 40.0 --time 2020-01-05T00:00 --depth 20

Besides <prefix>.png (FTLE and FSLE maps) it writes <prefix>_ridges.png: the FTLE
ridges (repelling = forward, attracting = backward) with the mean paths of small
particle clusters seeded across the box (--no-ridge-plot to skip).

Python:
    from lyapunov import compute_lyapunov
    res = compute_lyapunov(field, bbox, t_ref="2020-01-05T00:00",
                           integration_hours=72, direction="forward")
    res["ftle"], res["fsle"], res["lon"], res["lat"]
"""

import argparse
import os
import sys
import time as _time
import numpy as np

import lagrangian_tracker as lt

R_EARTH = lt.R_EARTH
DEG = np.pi / 180.0


# ---------------------------------------------------------------------------
# 1. Vectorized velocity sampling (many points at once)
# ---------------------------------------------------------------------------

def _locate_many(field, x, y):
    """Containing triangle and barycentric weights for arrays of points.
    Returns (tri_index (N,), bary (N, 3)); tri_index = -1 outside the mesh."""
    n = len(x)
    ti = np.full(n, -1, dtype=np.int64)
    bary = np.zeros((n, 3))
    ok = np.isfinite(x) & np.isfinite(y)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return ti, bary
    if field.locator is not None:                      # overlapping-mesh fallback
        for i in idx:
            t, b = field.locator.find(x[i], y[i])
            if t >= 0:
                ti[i], bary[i] = t, b
        return ti, bary
    found = np.asarray(field.trifinder(x[idx], y[idx]), dtype=np.int64)
    inside = found >= 0
    idx, found = idx[inside], found[inside]
    tri = field.triangles[found]
    xa, xb, xc = field.lon[tri[:, 0]], field.lon[tri[:, 1]], field.lon[tri[:, 2]]
    ya, yb, yc = field.lat[tri[:, 0]], field.lat[tri[:, 1]], field.lat[tri[:, 2]]
    det = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc)
    l1 = ((yb - yc) * (x[idx] - xc) + (xc - xb) * (y[idx] - yc)) / det
    l2 = ((yc - ya) * (x[idx] - xc) + (xa - xc) * (y[idx] - yc)) / det
    ti[idx] = found
    bary[idx] = np.stack([l1, l2, 1.0 - l1 - l2], axis=1)
    return ti, bary


def _vertical_interp(prof, levels, bottom, d):
    """Interpolate vertical profiles to depth d.

    prof: (n_level, N, 3) values at the 3 triangle corners; bottom: (N, 3)
    seabed depth or None. Same rule as the tracker: levels that are NaN or
    below the seabed are ignored, and a depth beyond the valid range takes
    the nearest valid level (a particle too deep rides along the bottom);
    no valid level gives 0."""
    nlev = len(levels)
    lev = levels[:, None, None]
    valid = np.isfinite(prof)
    if bottom is not None:
        valid &= lev <= bottom[None] + 1e-6
    k = np.arange(nlev)[:, None, None]
    lo = np.where(valid & (lev <= d), k, -1).max(axis=0)
    hi = np.where(valid & (lev >= d), k, nlev).min(axis=0)
    has_lo, has_hi = lo >= 0, hi < nlev
    lo_c, hi_c = np.clip(lo, 0, nlev - 1), np.clip(hi, 0, nlev - 1)
    v_lo = np.take_along_axis(prof, lo_c[None], axis=0)[0]
    v_hi = np.take_along_axis(prof, hi_c[None], axis=0)[0]
    d_lo, d_hi = levels[lo_c], levels[hi_c]
    span = d_hi - d_lo
    w = np.where(span > 0, (d - d_lo) / np.where(span > 0, span, 1.0), 0.0)
    return np.where(has_lo & has_hi, v_lo * (1 - w) + v_hi * w,
                    np.where(has_lo, v_lo, np.where(has_hi, v_hi, 0.0)))


def sample_velocity(field, lon, lat, t_sec, depth=None):
    """(u, v) in m/s at arrays of points for one time (seconds since the
    field's first snapshot) and, for fields with levels, one depth (m,
    positive down). NaN outside the mesh or the time record. Matches
    UnstructuredVelocityField.velocity_at, but for many points at once."""
    lon = np.asarray(lon, dtype=float).ravel()
    lat = np.asarray(lat, dtype=float).ravel()
    out_u = np.full(lon.shape, np.nan)
    out_v = np.full(lon.shape, np.nan)
    ts = field.t_seconds
    if t_sec < ts[0] or t_sec > ts[-1]:
        return out_u, out_v

    ti, bary = _locate_many(field, lon, lat)
    ok = ti >= 0
    if not ok.any():
        return out_u, out_v
    nodes = field.triangles[ti[ok]]                     # (M, 3)
    b = bary[ok]

    i1 = min(max(int(np.searchsorted(ts, t_sec, side="right")), 1), len(ts) - 1)
    i0 = i1 - 1
    w = 0.0 if ts[i1] == ts[i0] else (t_sec - ts[i0]) / (ts[i1] - ts[i0])

    if field.u.ndim == 2:
        cu = (1 - w) * field.u[i0][nodes] + w * field.u[i1][nodes]
        cv = (1 - w) * field.v[i0][nodes] + w * field.v[i1][nodes]
    else:
        d = field.level_depths[0] if depth is None else depth
        bottom = None if field.bottom_depth is None else field.bottom_depth[nodes]
        pu = (1 - w) * field.u[i0][:, nodes] + w * field.u[i1][:, nodes]   # (nlev, M, 3)
        pv = (1 - w) * field.v[i0][:, nodes] + w * field.v[i1][:, nodes]
        cu = _vertical_interp(pu, field.level_depths, bottom, d)
        cv = _vertical_interp(pv, field.level_depths, bottom, d)
    out_u[ok] = (b * cu).sum(axis=1)
    out_v[ok] = (b * cv).sum(axis=1)
    return out_u, out_v


# ---------------------------------------------------------------------------
# 2. Flow map integration + FTLE / FSLE
# ---------------------------------------------------------------------------

def _rhs(field, lon, lat, t, depth):
    u, v = sample_velocity(field, lon, lat, t, depth)
    return (u / (R_EARTH * np.cos(lat * DEG))) / DEG, (v / R_EARTH) / DEG   # deg/s


def _rk4_step(field, lon, lat, t, h, depth):
    """One RK4 step of length h seconds (negative = backward) from time t."""
    k1 = _rhs(field, lon, lat, t, depth)
    k2 = _rhs(field, lon + 0.5 * h * k1[0], lat + 0.5 * h * k1[1], t + 0.5 * h, depth)
    k3 = _rhs(field, lon + 0.5 * h * k2[0], lat + 0.5 * h * k2[1], t + 0.5 * h, depth)
    k4 = _rhs(field, lon + h * k3[0], lat + h * k3[1], t + h, depth)
    return (lon + (h / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
            lat + (h / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))


def _pair_distance(lon1, lat1, lon2, lat2):
    """Distance in metres between points (small separations: local
    equirectangular metric at the mean latitude)."""
    lm = 0.5 * (lat1 + lat2) * DEG
    return R_EARTH * DEG * np.hypot((lon2 - lon1) * np.cos(lm), lat2 - lat1)


def _max_neighbour_separation(lon2d, lat2d):
    """For every grid particle, the largest distance to its (up to four)
    grid neighbours, ignoring neighbours that no longer exist."""
    ny, nx = lon2d.shape
    seps = np.full((4, ny, nx), np.nan)
    dx = _pair_distance(lon2d[:, :-1], lat2d[:, :-1], lon2d[:, 1:], lat2d[:, 1:])
    dy = _pair_distance(lon2d[:-1], lat2d[:-1], lon2d[1:], lat2d[1:])
    seps[0, :, :-1], seps[1, :, 1:] = dx, dx
    seps[2, :-1], seps[3, 1:] = dy, dy
    return np.fmax.reduce(seps, axis=0)


def _cauchy_green_ftle(X, Y, delta0, T_sec):
    """FTLE (1/s) from final plane positions X, Y (ny, nx) of a regular
    grid with initial spacing delta0 (axis 1 = east, axis 0 = north)."""
    dXdy, dXdx = np.gradient(X, delta0)
    dYdy, dYdx = np.gradient(Y, delta0)
    a = dXdx ** 2 + dYdx ** 2
    b = dXdx * dXdy + dYdx * dYdy
    c = dXdy ** 2 + dYdy ** 2
    lam_max = 0.5 * (a + c) + np.sqrt(0.25 * (a - c) ** 2 + b ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * np.log(lam_max) / abs(T_sec)


def compute_lyapunov(field, bbox, t_ref, integration_hours=48.0, direction="forward",
                     spacing_km=None, n_points=120, dt_seconds=1800.0, depth=None,
                     delta_f_km=None, verbose=True):
    """
    FTLE and FSLE on a regular grid over bbox = (lon_min, lon_max, lat_min, lat_max).

    t_ref : reference time (anything np.datetime64 accepts). Forward uses
        [t_ref, t_ref + T], backward [t_ref - T, t_ref]; the record must cover it.
    spacing_km : grid spacing (= FSLE delta_0). Default: chosen so the longer
        side of the box has about n_points points.
    delta_f_km : FSLE final separation (default 10 x spacing).
    depth : fixed particle depth (m, positive down) for fields with levels.

    Returns a dict: lon, lat (ny, nx); ftle, fsle in 1/day; plus the parameters.
    """
    if direction not in ("forward", "backward"):
        raise ValueError("direction must be 'forward' or 'backward'")
    sign = 1.0 if direction == "forward" else -1.0
    lon_min, lon_max, lat_min, lat_max = bbox
    lon_ref, lat_ref = 0.5 * (lon_min + lon_max), 0.5 * (lat_min + lat_max)
    kx = R_EARTH * np.cos(lat_ref * DEG) * DEG        # m per degree lon at the box centre
    ky = R_EARTH * DEG                                # m per degree lat
    width, height = (lon_max - lon_min) * kx, (lat_max - lat_min) * ky
    if spacing_km is None:
        spacing_km = max(width, height) / 1000.0 / max(n_points - 1, 1)
    delta0 = spacing_km * 1000.0
    delta_f = (delta_f_km if delta_f_km is not None else 10.0 * spacing_km) * 1000.0
    if delta_f <= delta0:
        raise ValueError("delta_f must be larger than the grid spacing")

    # seeds: uniform in the local east/north plane
    nxg = max(int(np.floor(width / delta0)) + 1, 3)
    nyg = max(int(np.floor(height / delta0)) + 1, 3)
    xs = (np.arange(nxg) - (nxg - 1) / 2.0) * delta0
    ys = (np.arange(nyg) - (nyg - 1) / 2.0) * delta0
    X0, Y0 = np.meshgrid(xs, ys)
    lon2d = lon_ref + X0 / kx
    lat2d = lat_ref + Y0 / ky

    t0 = float((np.datetime64(t_ref) - field.times[0]) / np.timedelta64(1, "s"))
    T = integration_hours * 3600.0
    if verbose:
        print(f"{direction} Lyapunov: {nyg} x {nxg} = {nyg * nxg} particles, spacing "
              f"{spacing_km:.2f} km, T = {integration_hours:g} h, dt = {dt_seconds:g} s")

    lon, lat = lon2d.ravel().copy(), lat2d.ravel().copy()
    shape = lon2d.shape
    tau = np.full(shape, np.nan)
    sep_prev = np.full(shape, delta0)
    n_steps = int(np.ceil(T / dt_seconds))
    started = _time.time()
    for n in range(n_steps):
        h = min(dt_seconds, T - n * dt_seconds) * sign
        t = t0 + sign * n * dt_seconds
        lon, lat = _rk4_step(field, lon, lat, t, h, depth)

        sep = _max_neighbour_separation(lon.reshape(shape), lat.reshape(shape))
        with np.errstate(invalid="ignore"):
            crossed = np.isnan(tau) & (sep >= delta_f) & np.isfinite(sep_prev)
        if crossed.any():
            frac = (delta_f - sep_prev[crossed]) / (sep[crossed] - sep_prev[crossed])
            tau[crossed] = n * dt_seconds + frac * abs(h)
        sep_prev = np.where(np.isfinite(sep), sep, sep_prev)

    lon_f, lat_f = lon.reshape(shape), lat.reshape(shape)
    Xf, Yf = (lon_f - lon_ref) * kx, (lat_f - lat_ref) * ky
    ftle = _cauchy_green_ftle(Xf, Yf, delta0, T) * 86400.0
    # a seed that never made it to the end has no flow map; neighbours' stencils
    # that include it were already poisoned by NaN in np.gradient
    ftle[~np.isfinite(Xf)] = np.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        fsle = np.log(delta_f / delta0) / tau * 86400.0

    # the outermost ring of seeds has only 2-3 neighbours (one-sided gradients),
    # so its values are not comparable with the interior -- mask it
    for arr in (ftle, fsle):
        arr[0, :] = arr[-1, :] = arr[:, 0] = arr[:, -1] = np.nan

    if verbose:
        print(f"  done in {_time.time() - started:.1f} s; FTLE valid at "
              f"{np.isfinite(ftle).mean() * 100:.0f}% of points, FSLE at "
              f"{np.isfinite(fsle).mean() * 100:.0f}%")
    return dict(lon=lon2d, lat=lat2d, ftle=ftle, fsle=fsle, direction=direction,
                t_ref=np.datetime64(t_ref), integration_hours=integration_hours,
                spacing_km=spacing_km, delta_f_km=delta_f / 1000.0, depth=depth,
                final_lon=lon_f, final_lat=lat_f)


# ---------------------------------------------------------------------------
# 3. FTLE ridges (Lagrangian coherent structures) and representative trajectories
# ---------------------------------------------------------------------------

def _smooth_nan(a, sigma_cells):
    """Gaussian smoothing that ignores NaN (normalized convolution)."""
    if sigma_cells <= 0:
        return a.copy()
    r = max(int(np.ceil(3 * sigma_cells)), 1)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_cells) ** 2)
    k /= k.sum()
    w = np.isfinite(a).astype(float)
    x = np.where(np.isfinite(a), a, 0.0)

    def conv(arr):
        for axis in (0, 1):
            pad = [(0, 0), (0, 0)]
            pad[axis] = (r, r)
            p = np.pad(arr, pad, mode="edge")
            arr = sum(k[i] * np.take(p, np.arange(i, i + arr.shape[axis]), axis=axis)
                      for i in range(len(k)))
        return arr
    num, den = conv(x), conv(w)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(w > 0, out, np.nan)


def _bilinear(arr, rr, cc):
    """Sample arr at fractional (row, col) positions; NaN outside the array."""
    ny, nx = arr.shape
    bad = ~np.isfinite(rr) | ~np.isfinite(cc)
    rr, cc = np.where(bad, 0.0, rr), np.where(bad, 0.0, cc)
    r0, c0 = np.floor(rr).astype(int), np.floor(cc).astype(int)
    ok = (r0 >= 0) & (r0 < ny - 1) & (c0 >= 0) & (c0 < nx - 1) & ~bad
    r0c, c0c = np.clip(r0, 0, ny - 2), np.clip(c0, 0, nx - 2)
    fr, fc = rr - r0c, cc - c0c
    v = ((1 - fr) * (1 - fc) * arr[r0c, c0c] + (1 - fr) * fc * arr[r0c, c0c + 1]
         + fr * (1 - fc) * arr[r0c + 1, c0c] + fr * fc * arr[r0c + 1, c0c + 1])
    return np.where(ok, v, np.nan)


def extract_ftle_ridges(ftle, spacing_m, smooth_cells=2.0, min_ftle_fraction=0.4,
                        min_curvature_fraction=0.02, min_length_cells=10,
                        min_contrast_fraction=0.2):
    """
    Ridges of an FTLE field = candidate Lagrangian coherent structures.

    A pixel is a ridge point if (second-derivative / Hessian criterion):
      * the FTLE is a local maximum across the ridge: it is >= its value one
        cell either side along n, the direction of strongest concavity (the
        eigenvector of the Hessian's most negative eigenvalue);
      * the field is concave there (that eigenvalue is clearly negative);
      * the FTLE stands out: it is above the field's median by at least
        min_ftle_fraction of the median-to-99th-percentile range.
    A field whose 99th percentile exceeds its median by less than
    min_contrast_fraction of itself is treated as structureless (no ridges).
    The field is lightly smoothed first (smooth_cells, Gaussian), because
    second derivatives amplify noise. Short fragments (< min_length_cells
    connected pixels) are removed.

    Returns a dict: mask (bool, same shape as ftle), strength (-lambda_min,
    1/day per m^2), ftle_smooth.
    """
    sm = _smooth_nan(np.asarray(ftle, dtype=float), smooth_cells)
    gy, gx = np.gradient(sm, spacing_m)
    gyy, gyx = np.gradient(gy, spacing_m)
    gxy, gxx = np.gradient(gx, spacing_m)
    a, c, b = gxx, gyy, 0.5 * (gxy + gyx)
    root = np.sqrt(0.25 * (a - c) ** 2 + b ** 2)
    lam = 0.5 * (a + c) - root                        # most negative eigenvalue
    nx_, ny_ = b, lam - a                             # its eigenvector (x = columns, y = rows)
    norm = np.hypot(nx_, ny_)
    degenerate = norm < 1e-30 * max(np.nanmax(np.abs(lam)) if np.isfinite(lam).any() else 1.0, 1e-300)
    nx_ = np.where(degenerate, np.where(a <= c, 1.0, 0.0), nx_ / np.where(degenerate, 1.0, norm))
    ny_ = np.where(degenerate, np.where(a <= c, 0.0, 1.0), ny_ / np.where(degenerate, 1.0, norm))

    rows, cols = np.indices(sm.shape).astype(float)
    ahead = _bilinear(sm, rows + ny_, cols + nx_)
    behind = _bilinear(sm, rows - ny_, cols - nx_)
    with np.errstate(invalid="ignore"):
        local_max = (sm >= ahead) & (sm >= behind)
        valid = np.isfinite(sm) & np.isfinite(lam)
        if not valid.any():
            return dict(mask=np.zeros(sm.shape, bool), strength=-lam, ftle_smooth=sm)
        p99, med = np.nanpercentile(sm[valid], 99), np.nanmedian(sm[valid])
        if p99 - med < min_contrast_fraction * abs(p99):      # nearly uniform field: no structure
            return dict(mask=np.zeros(sm.shape, bool), strength=-lam, ftle_smooth=sm)
        ftle_thr = med + min_ftle_fraction * (p99 - med)
        concave = valid & (lam < 0)
        curv_ref = np.nanpercentile(-lam[concave], 99) if concave.any() else np.inf
        mask = local_max & valid & (lam < -min_curvature_fraction * curv_ref) & (sm >= ftle_thr)

    # drop isolated pixels and short fragments
    try:
        from scipy import ndimage
        labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
        if n:
            sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
            keep = np.isin(labels, 1 + np.where(sizes >= min_length_cells)[0])
            mask = mask & keep
    except ImportError:
        pad = np.pad(mask, 1)
        neighbours = sum(np.roll(np.roll(pad, i, 0), j, 1) for i in (-1, 0, 1)
                         for j in (-1, 0, 1) if (i, j) != (0, 0))[1:-1, 1:-1]
        mask = mask & (neighbours > 0)
    return dict(mask=mask, strength=-lam, ftle_smooth=sm)


def advect_paths(field, lon0, lat0, t_ref, integration_hours, direction="forward",
                 dt_seconds=1800.0, depth=None):
    """Full paths of many particles (all advected together). Returns
    (lon_path, lat_path, elapsed_seconds); paths have shape (n_steps+1, N)
    and turn NaN once a particle leaves the mesh."""
    sign = 1.0 if direction == "forward" else -1.0
    lon = np.asarray(lon0, dtype=float).ravel().copy()
    lat = np.asarray(lat0, dtype=float).ravel().copy()
    t0 = float((np.datetime64(t_ref) - field.times[0]) / np.timedelta64(1, "s"))
    T = integration_hours * 3600.0
    n_steps = int(np.ceil(T / dt_seconds))
    plon = np.full((n_steps + 1, lon.size), np.nan)
    plat = np.full_like(plon, np.nan)
    elapsed = np.zeros(n_steps + 1)
    plon[0], plat[0] = lon, lat
    for n in range(n_steps):
        h = min(dt_seconds, T - n * dt_seconds) * sign
        lon, lat = _rk4_step(field, lon, lat, t0 + sign * n * dt_seconds, h, depth)
        plon[n + 1], plat[n + 1] = lon, lat
        elapsed[n + 1] = elapsed[n] + abs(h)
    return plon, plat, elapsed


def cluster_seeds(bbox, n_x, n_y=None, cluster_size=3, spacing_m=2000.0, margin=0.06):
    """
    Seed locations for representative trajectories: an n_x x n_y grid of
    cluster centres over bbox (rows chosen from the aspect ratio if n_y is
    None), inset by `margin` of the box size. Each centre carries a
    cluster_size x cluster_size cluster of particles spacing_m apart.
    Returns centre lon, lat (n_seed,) and member lon, lat (n_seed, cluster_size**2).
    """
    lon_min, lon_max, lat_min, lat_max = bbox
    lat_c = 0.5 * (lat_min + lat_max)
    kx, ky = R_EARTH * np.cos(lat_c * DEG) * DEG, R_EARTH * DEG
    if n_y is None:
        n_y = max(2, int(round(n_x * (lat_max - lat_min) * ky / ((lon_max - lon_min) * kx))))
    mx, my = margin * (lon_max - lon_min), margin * (lat_max - lat_min)
    cx, cy = np.meshgrid(np.linspace(lon_min + mx, lon_max - mx, n_x),
                         np.linspace(lat_min + my, lat_max - my, n_y))
    cx, cy = cx.ravel(), cy.ravel()
    off = (np.arange(cluster_size) - (cluster_size - 1) / 2.0) * spacing_m
    ox, oy = np.meshgrid(off, off)
    return (cx, cy, cx[:, None] + ox.ravel()[None] / kx, cy[:, None] + oy.ravel()[None] / ky)


def compute_ridges_and_trajectories(field, result, bbox, n_traj_x=5, n_traj_y=None,
                                    cluster_size=3, cluster_spacing_km=None,
                                    dt_seconds=1800.0, depth=None, **ridge_kwargs):
    """
    Ridges of result['ftle'] plus representative trajectories over the box.

    A coarse grid of n_traj_x x n_traj_y seed locations is released over the
    box; each seed is a small cluster (cluster_size x cluster_size particles,
    cluster_spacing_km apart) and the *mean* path of the cluster is the
    "average trajectory". The cluster's final particle positions are kept
    too, because where a ridge runs through a cluster the average hides how
    far apart its members end up. A mean path ends when any member of its
    cluster leaves the mesh. cluster_size=1 gives individual trajectories.
    Seeds are advected in the same direction as the result.
    """
    ridges = extract_ftle_ridges(result["ftle"], result["spacing_km"] * 1000.0, **ridge_kwargs)
    spacing = (cluster_spacing_km if cluster_spacing_km is not None
               else 2.0 * result["spacing_km"]) * 1000.0
    cx, cy, mem_lon, mem_lat = cluster_seeds(bbox, n_traj_x, n_traj_y, cluster_size, spacing)

    plon, plat, elapsed = advect_paths(field, mem_lon.ravel(), mem_lat.ravel(), result["t_ref"],
                                       result["integration_hours"], result["direction"],
                                       dt_seconds, depth)
    steps, ns, m = plon.shape[0], len(cx), cluster_size ** 2
    plon, plat = plon.reshape(steps, ns, m), plat.reshape(steps, ns, m)
    with np.errstate(invalid="ignore"):
        mean_lon, mean_lat = plon.mean(axis=2), plat.mean(axis=2)     # NaN once any member exits
    return dict(ridge_mask=ridges["mask"], ridge_strength=ridges["strength"],
                ftle_smooth=ridges["ftle_smooth"], mean_lon=mean_lon, mean_lat=mean_lat,
                member_end_lon=plon[-1], member_end_lat=plat[-1], elapsed=elapsed,
                direction=result["direction"], cluster_size=cluster_size)


def plot_ridges_and_trajectories(results, extras, out_path):
    """
    FTLE (grey) with its ridges and the average trajectories on top. One
    panel per direction; with both directions, a third panel overlays the
    forward ridges (repelling structures) and backward ridges (attracting
    structures) on the forward FTLE with the forward trajectories.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.lines import Line2D
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    ridge_color = {"forward": "crimson", "backward": "royalblue"}
    ridge_label = {"forward": "forward-FTLE ridges (repelling)",
                   "backward": "backward-FTLE ridges (attracting)"}
    by_dir = {r["direction"]: (r, e) for r, e in zip(results, extras)}
    panels = [([d], d) for d in by_dir]
    if len(by_dir) == 2:
        panels.append((["forward", "backward"], "forward"))
    mean_lat = float(np.mean(results[0]["lat"]))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.6 * len(panels), 5.6), squeeze=False)

    for ax, (ridge_dirs, base_dir) in zip(axes[0], panels):
        r, e = by_dir[base_dir]
        vmax = np.nanpercentile(r["ftle"], 99) if np.isfinite(r["ftle"]).any() else 1.0
        mesh = ax.pcolormesh(r["lon"], r["lat"], r["ftle"], cmap="Greys", vmin=0, vmax=vmax,
                             shading="auto")
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.08)
        plt.colorbar(mesh, cax=cax, label=f"{base_dir} FTLE (1/day)")
        handles = []
        for d in ridge_dirs:
            mask = by_dir[d][1]["ridge_mask"]
            # one marker per ridge cell, slightly larger than the cell so the ridge reads as a line
            ax.scatter(r["lon"][mask], r["lat"][mask], s=13, marker="s", color=ridge_color[d],
                       linewidths=0, zorder=6)
            handles.append(Line2D([0], [0], color=ridge_color[d], lw=3, label=ridge_label[d]))

        n_seed = e["mean_lon"].shape[1]
        order = np.argsort(np.argsort(e["mean_lon"][0] + 1e-3 * e["mean_lat"][0]))
        cmap = plt.get_cmap("viridis")
        for i in range(n_seed):
            lo, la = e["mean_lon"][:, i], e["mean_lat"][:, i]
            ok = np.isfinite(lo) & np.isfinite(la)
            if ok.sum() < 2:
                continue
            col = cmap(order[i] / max(n_seed - 1, 1))
            last = np.where(ok)[0].max()
            ax.plot(lo[:last + 1], la[:last + 1], color=col, lw=1.1,
                    path_effects=[pe.withStroke(linewidth=2.3, foreground="white")], zorder=4)
            ax.scatter(lo[0], la[0], s=22, color=col, edgecolor="k", linewidths=0.6, zorder=5)
            ax.annotate("", xy=(lo[last], la[last]), xytext=(lo[max(last - 1, 0)], la[max(last - 1, 0)]),
                        arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0, mutation_scale=8,
                                        path_effects=[pe.withStroke(linewidth=2.2, foreground="white")]),
                        zorder=5)
            if e["cluster_size"] > 1:
                ax.scatter(e["member_end_lon"][i], e["member_end_lat"][i], s=5, color=col,
                           alpha=0.45, linewidths=0, zorder=3)
        handles += [Line2D([0], [0], color="0.3", lw=1.4, marker="o", ms=4,
                           label=("mean path of each particle cluster" if e["cluster_size"] > 1
                                  else "particle trajectory")),
                    Line2D([0], [0], color="0.5", marker=".", ls="", ms=4,
                           label="cluster end positions")] if e["cluster_size"] > 1 else [
                    Line2D([0], [0], color="0.3", lw=1.4, marker="o", ms=4, label="particle trajectory")]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2),
                  fontsize=7.5, frameon=False, ncol=1)
        title = {"forward": "forward", "backward": "backward"}[base_dir]
        if len(ridge_dirs) == 2:
            title = "repelling + attracting ridges, forward trajectories"
        ax.set_title(f"{title}, T = {r['integration_hours']:g} h\nt_ref = {str(r['t_ref'])[:16]}",
                     fontsize=9)
        ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ridge/trajectory plot to {out_path}")


# ---------------------------------------------------------------------------
# 4. Synthetic test field: time-dependent double gyre on a triangle mesh
# ---------------------------------------------------------------------------

def synthetic_double_gyre(L_km=100.0, tau_hours=24.0, A=0.1, eps=0.25,
                          omega=2 * np.pi / 10.0, spacing_km=2.5, n_days=30,
                          snapshot_hours=6.0, lon_c=-74.0, lat_c=39.0, seed=0):
    """Shadden's double gyre (a standard Lagrangian-coherent-structure test)
    on a jittered triangle mesh. Domain 2L x L, one non-dimensional time
    unit = tau_hours. Returns the same tuple as the tracker's synthetic
    generators: lon, lat, triangles, times, u, v."""
    import matplotlib.tri as mtri
    L = L_km * 1000.0
    kx = R_EARTH * np.cos(lat_c * DEG) * DEG
    ky = R_EARTH * DEG
    nx, ny = int(2 * L_km / spacing_km) + 1, int(L_km / spacing_km) + 1
    gx, gy = np.meshgrid(np.linspace(0, 2, nx), np.linspace(0, 1, ny))
    rng = np.random.default_rng(seed)
    jit = 0.3 * (2.0 / (nx - 1))
    interior = (gx > 0) & (gx < 2) & (gy > 0) & (gy < 1)
    gx = gx + np.where(interior, rng.uniform(-jit, jit, gx.shape), 0.0)
    gy = gy + np.where(interior, rng.uniform(-jit, jit, gy.shape), 0.0)
    x, y = gx.ravel(), gy.ravel()
    lon = lon_c + (x - 1.0) * L / kx
    lat = lat_c + (y - 0.5) * L / ky
    tri = mtri.Triangulation(lon, lat).triangles

    n_t = int(n_days * 24 / snapshot_hours) + 1
    t_nd = np.arange(n_t) * snapshot_hours / tau_hours          # non-dimensional time
    times = np.datetime64("2020-01-01T00:00") + (np.arange(n_t) * snapshot_hours * 3600).astype(
        "timedelta64[s]")
    speed = L / (tau_hours * 3600.0)                            # m/s per non-dimensional unit
    u = np.empty((n_t, len(x)), dtype=np.float32)
    v = np.empty_like(u)
    for k, tn in enumerate(t_nd):
        a = eps * np.sin(omega * tn)
        f = a * x ** 2 + (1 - 2 * a) * x
        dfdx = 2 * a * x + 1 - 2 * a
        u[k] = -np.pi * A * np.sin(np.pi * f) * np.cos(np.pi * y) * speed
        v[k] = np.pi * A * np.cos(np.pi * f) * np.sin(np.pi * y) * dfdx * speed
    return lon, lat, tri, times, u, v


# ---------------------------------------------------------------------------
# 5. Plot / save / CLI
# ---------------------------------------------------------------------------

def plot_lyapunov(results, out_path, field=None):
    """One panel per (direction, quantity): FTLE and FSLE, forward/backward."""
    import matplotlib.pyplot as plt
    os_dir = os.path.dirname(os.path.abspath(out_path)) or "."
    os.makedirs(os_dir, exist_ok=True)
    panels = [(r, q) for q in ("ftle", "fsle") for r in results]
    fig, axes = plt.subplots(2 if len(results) > 1 else 1, len(results) if len(results) > 1 else 2,
                             figsize=(6.2 * max(len(results), 1) + 1, 5.2 * (2 if len(results) > 1 else 1)),
                             squeeze=False)
    axes = axes.T.ravel() if len(results) > 1 else axes.ravel()
    mean_lat = float(np.mean(results[0]["lat"]))
    for ax, (r, q) in zip(axes, panels):
        vals = r[q]
        vmax = np.nanpercentile(vals, 99) if np.isfinite(vals).any() else 1.0
        m = ax.pcolormesh(r["lon"], r["lat"], vals, cmap="inferno", vmin=0, vmax=vmax, shading="auto")
        plt.colorbar(m, ax=ax, fraction=0.046, label="1/day")
        name = "FTLE" if q == "ftle" else f"FSLE ($\\delta_f$={r['delta_f_km']:.0f} km)"
        ax.set_title(f"{r['direction']} {name}, T={r['integration_hours']:g} h\n"
                     f"t_ref = {str(r['t_ref'])[:16]}" + (f", depth {r['depth']:g} m" if r["depth"] is not None else ""),
                     fontsize=9)
        ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
    for ax in axes[len(panels):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved Lyapunov plot to {out_path}")


def save_npz(results, path, extras=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    data = {}
    for r, e in zip(results, extras or [None] * len(results)):
        d = r["direction"]
        if e is not None:
            data[f"ridge_mask_{d}"] = e["ridge_mask"]
            data[f"mean_traj_lon_{d}"] = e["mean_lon"]
            data[f"mean_traj_lat_{d}"] = e["mean_lat"]
            data[f"traj_elapsed_s_{d}"] = e["elapsed"]
        for key in ("lon", "lat", "ftle", "fsle"):
            data[f"{key}_{d}" if key not in ("lon", "lat") else key] = r[key]
        data[f"params_{d}"] = np.array([r["integration_hours"], r["spacing_km"], r["delta_f_km"]])
        data["t_ref"] = np.datetime64(r["t_ref"], "s").astype(str)
    np.savez_compressed(path, **data)
    print(f"Saved arrays to {path}")


def _load_field(args, t_lo, t_hi, bbox):
    """Build a velocity field from --source using the tracker's loaders."""
    storage_options = {"anon": True} if args.zarr_anon else None
    gt = lt.normalize_grid_type(args.grid_type)
    if gt == "auto":
        gt = lt.detect_grid_type(args.source, engine=args.engine, storage_options=storage_options)
        print(f"Detected grid type: {gt}")
    depth_range = (0.0, args.depth) if args.depth else None
    common = dict(t_start=t_lo, t_end=t_hi, depth_range=depth_range, return_vertical=True,
                  engine=args.engine, storage_options=storage_options,
                  u_var=args.u_var, v_var=args.v_var)
    if gt == "unstructured":
        lon, lat, tri, times, u, v, lev, bot = lt.load_unstructured_currents(args.source, **common)
    else:
        pad = args.load_pad
        read_box = (bbox[0] - pad, bbox[1] + pad, bbox[2] - pad, bbox[3] + pad)
        if gt == "roms":
            targets = lt.schedule_target_depths([args.depth], args.depth) if args.depth else None
            lon, lat, tri, times, u, v, lev, bot = lt.load_roms_currents(
                args.source, bbox=read_box, target_depths=targets, **common)
        else:
            lon, lat, tri, times, u, v, lev, bot = lt.load_rectilinear_currents(
                args.source, bbox=read_box, **common)
    return lt.UnstructuredVelocityField(lon, lat, tri, times, u, v,
                                        level_depths=lev, bottom_depth=bot)


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
                   metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
                   help="Region to compute over (required with --source)")
    p.add_argument("--load-pad", type=float, default=2.0,
                   help="Structured grids: extra degrees read around --bbox so particles that "
                        "drift out of the box are still inside the data")
    p.add_argument("--time", default=None, help="Reference time t0, ISO8601")
    p.add_argument("--n-times", type=int, default=1,
                   help="Compute at this many reference times (--time-step-hours apart)")
    p.add_argument("--time-step-hours", type=float, default=24.0)
    p.add_argument("--integration-hours", type=float, default=48.0)
    p.add_argument("--direction", choices=["forward", "backward", "both"], default="both")
    p.add_argument("--dt-seconds", type=float, default=1800.0)
    p.add_argument("--spacing-km", type=float, default=None)
    p.add_argument("--n-points", type=int, default=120,
                   help="Grid points along the longer side when --spacing-km is not given")
    p.add_argument("--delta-f-km", type=float, default=None,
                   help="FSLE final separation (default 10 x spacing)")
    p.add_argument("--depth", type=float, default=None,
                   help="Fixed depth (m, positive down) for fields with vertical levels")
    p.add_argument("--out-prefix", default="lyapunov")
    p.add_argument("--no-plot", action="store_true")
    g = p.add_argument_group("ridges and trajectories (the *_ridges.png plot)")
    g.add_argument("--no-ridge-plot", action="store_true",
                   help="Skip the FTLE-ridge + average-trajectory plot")
    g.add_argument("--n-traj-x", type=int, default=5,
                   help="Trajectory seed locations across the box (rows chosen from the aspect ratio)")
    g.add_argument("--n-traj-y", type=int, default=None)
    g.add_argument("--cluster-size", type=int, default=3,
                   help="Each seed is a cluster of N x N particles whose mean path is drawn "
                        "(1 = single trajectories)")
    g.add_argument("--cluster-spacing-km", type=float, default=None,
                   help="Particle spacing inside a cluster (default 2 x grid spacing)")
    g.add_argument("--ridge-ftle-fraction", type=float, default=0.4,
                   help="Ridges must rise above the median FTLE by this fraction of the "
                        "median-to-99th-percentile range (lower = more, weaker ridges)")
    g.add_argument("--ridge-curvature-fraction", type=float, default=0.02,
                   help="Minimum ridge sharpness, as a fraction of the strongest")
    g.add_argument("--ridge-smooth-cells", type=float, default=2.0,
                   help="Gaussian smoothing (grid cells) applied before taking derivatives")
    args = p.parse_args()

    dirs = ["forward", "backward"] if args.direction == "both" else [args.direction]
    T = np.timedelta64(int(args.integration_hours * 3600), "s")

    if args.synthetic or not args.source:
        if not args.synthetic:
            print("No --source given; using the synthetic double gyre.", file=sys.stderr)
        # a shallow double gyre run for ~3 non-dimensional periods
        lon, lat, tri, times, u, v = synthetic_double_gyre()
        field = lt.UnstructuredVelocityField(lon, lat, tri, times, u, v)
        if args.bbox is None:
            m = 0.003
            args.bbox = (lon.min() + m, lon.max() - m, lat.min() + m, lat.max() - m)
        t_ref = np.datetime64(args.time) if args.time else field.times[0] + (
            T if "backward" in dirs else np.timedelta64(0, "s"))
        if args.integration_hours == 48.0:
            print("(the double-gyre demo is usually shown with T = 15 days; "
                  "try --integration-hours 360)")
    else:
        if args.bbox is None or args.time is None:
            p.error("--bbox and --time are required with --source")
        t_ref = np.datetime64(args.time)
        span = np.timedelta64(int((args.n_times - 1) * args.time_step_hours * 3600), "s")
        t_lo = t_ref - (T if "backward" in dirs else np.timedelta64(0, "s"))
        t_hi = t_ref + span + (T if "forward" in dirs else np.timedelta64(0, "s"))
        field = _load_field(args, t_lo, t_hi, tuple(args.bbox))

    for i in range(args.n_times):
        t_i = t_ref + np.timedelta64(int(i * args.time_step_hours * 3600), "s")
        results = [compute_lyapunov(field, tuple(args.bbox), t_i, args.integration_hours, d,
                                    spacing_km=args.spacing_km, n_points=args.n_points,
                                    dt_seconds=args.dt_seconds, depth=args.depth,
                                    delta_f_km=args.delta_f_km) for d in dirs]
        tag = "" if args.n_times == 1 else f"_{i:03d}"
        extras = None
        if not args.no_ridge_plot:
            extras = [compute_ridges_and_trajectories(
                field, r, tuple(args.bbox), n_traj_x=args.n_traj_x, n_traj_y=args.n_traj_y,
                cluster_size=args.cluster_size, cluster_spacing_km=args.cluster_spacing_km,
                dt_seconds=args.dt_seconds, depth=args.depth,
                smooth_cells=args.ridge_smooth_cells,
                min_ftle_fraction=args.ridge_ftle_fraction,
                min_curvature_fraction=args.ridge_curvature_fraction) for r in results]
        save_npz(results, f"{args.out_prefix}{tag}.npz", extras)
        if not args.no_plot:
            plot_lyapunov(results, f"{args.out_prefix}{tag}.png")
            if extras is not None:
                plot_ridges_and_trajectories(results, extras, f"{args.out_prefix}{tag}_ridges.png")


if __name__ == "__main__":
    main()
