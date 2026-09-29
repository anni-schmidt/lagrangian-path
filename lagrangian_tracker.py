"""
Offline Lagrangian particle tracker for unstructured-grid ocean currents
==========================================================================

Advects a particle (or several) through a pre-computed u/v current field
defined on an unstructured triangular mesh (e.g. FVCOM, SCHISM, ADCIRC-style
output), starting from a user-specified release longitude/latitude/time.
"Offline" means it works entirely from saved model output -- it does not
run or couple to the ocean model itself.

Method
------
  * Spatial interpolation: matplotlib.tri.LinearTriInterpolator on the
    mesh's Delaunay/model triangulation (barycentric interpolation within
    the containing triangle, located via a TriFinder). A particle that
    leaves the meshed domain (interpolator returns NaN) is treated as
    exited/beached and tracking stops there.
  * Temporal interpolation: linear blend between the two nearest saved
    time snapshots that bracket the particle's current time.
  * Time integration: classic 4th-order Runge-Kutta (RK4) on the
    velocity field, with m/s converted to deg/s locally via the great
    circle metric (R_earth, cos(latitude)).

Requirements
------------
    pip install numpy matplotlib xarray netCDF4
    pip install zarr fsspec                # for local/cloud Zarr stores
    pip install s3fs                       # for s3:// Zarr stores
    pip install gcsfs                      # for gs:// Zarr stores

xarray/netCDF4 are only needed to load a real unstructured-grid NetCDF
file; zarr/fsspec (+s3fs or gcsfs for cloud paths) are only needed if
your source is a Zarr store rather than a .nc file. The tracker itself
(interpolation, RK4, plotting) only needs numpy/matplotlib. Without a
real file the script demonstrates itself on a synthetic rotating-gyre
mesh so the whole pipeline is testable offline.

Usage
-----
    python lagrangian_tracker.py --synthetic \
        --lon -74.3 --lat 39.2 --time 2020-01-01T00:00:00 \
        --duration-hours 48

    python lagrangian_tracker.py --source my_fvcom_output.nc \
        --lon -74.3 --lat 39.2 --time 2020-01-01T00:00:00 \
        --duration-hours 72 --dt-seconds 300

Multiple release points: repeat --lon/--lat/--time in matching order,
e.g. --lon -74.3 -73.9 --lat 39.2 39.0 --time 2020-01-01T00:00:00 2020-01-01T06:00:00
"""

import argparse
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

R_EARTH = 6371000.0  # m


def validate_mesh(lon, lat, triangles):
    """
    Check a (lon, lat, triangles) mesh for the issues that commonly make
    matplotlib.tri.Triangulation raise "Triangulation is invalid", and
    raise a specific, actionable error instead of the generic one.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    triangles = np.asarray(triangles)
    n_nodes = len(lon)

    if lon.shape != lat.shape:
        raise ValueError(f"lon and lat have different shapes: {lon.shape} vs {lat.shape}")

    if np.nanmax(np.abs(lat)) > 90 or np.nanmax(np.abs(lon)) > 360:
        raise ValueError(
            f"Node coordinates don't look like degrees (lon range "
            f"{np.nanmin(lon):.4g}..{np.nanmax(lon):.4g}, lat range "
            f"{np.nanmin(lat):.4g}..{np.nanmax(lat):.4g}). D-Flow FM / "
            f"flexible-mesh output is often in projected metres -- use the "
            f"spherical lon/lat node variables, or reproject to lon/lat first.")

    bad_coords = np.where(~np.isfinite(lon) | ~np.isfinite(lat))[0]
    if len(bad_coords):
        raise ValueError(
            f"{len(bad_coords)} node(s) have NaN/Inf lon or lat "
            f"(e.g. node indices {bad_coords[:5].tolist()}) -- check for "
            f"an unmasked fill value in the source file's coordinates.")

    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(
            f"triangles must have shape (n_triangles, 3); got {triangles.shape}. "
            f"If your connectivity variable is (3, n_elements) it needs transposing.")
        

    tmin, tmax = int(triangles.min()), int(triangles.max())
    if tmin < 0 or tmax >= n_nodes:
        raise ValueError(
            f"triangle connectivity references node index {tmax} (min {tmin}), "
            f"but only {n_nodes} nodes were loaded (valid range 0..{n_nodes - 1}). "
            f"This almost always means the node coordinates were spatially "
            f"subset/trimmed without renumbering the triangle connectivity to "
            f"match -- re-load the full node set, or remap triangle indices "
            f"after subsetting instead of subsetting lon/lat directly. It can "
            f"also mean the connectivity array is still 1-based (FVCOM 'nv' "
            f"convention) and the 1-based-to-0-based conversion was skipped.")

    p0, p1, p2 = lon[triangles[:, 0]], lon[triangles[:, 1]], lon[triangles[:, 2]]
    q0, q1, q2 = lat[triangles[:, 0]], lat[triangles[:, 1]], lat[triangles[:, 2]]
    signed_area2 = (p1 - p0) * (q2 - q0) - (p2 - p0) * (q1 - q0)
    degenerate = np.where(np.abs(signed_area2) < 1e-12)[0]
    if len(degenerate):
        raise ValueError(
            f"{len(degenerate)} triangle(s) are degenerate/zero-area "
            f"(e.g. triangle indices {degenerate[:5].tolist()}), usually from "
            f"duplicate/coincident node coordinates. Check for repeated "
            f"points in lon/lat, e.g. via "
            f"np.unique(np.stack([lon, lat], axis=1), axis=0).")

    dup_rows = triangles.copy()
    dup_rows.sort(axis=1)
    _, counts = np.unique(dup_rows, axis=0, return_counts=True)
    if (counts > 1).any():
        raise ValueError(
            f"{int((counts > 1).sum())} triangle(s) appear more than once in "
            f"the connectivity array -- check for accidental concatenation "
            f"of connectivity from multiple grids/time steps.")

    edges = np.concatenate([
        triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]
    ])
    edges = np.sort(edges, axis=1)
    _, edge_counts = np.unique(edges, axis=0, return_counts=True)
    if (edge_counts > 2).any():
        raise ValueError(
            f"{int((edge_counts > 2).sum())} edge(s) are shared by more than "
            f"2 triangles -- the mesh is non-manifold (likely duplicated or "
            f"overlapping triangles somewhere in the connectivity array).")


# ---------------------------------------------------------------------------
# 1. Velocity field container (unstructured mesh + time-varying u, v)
# ---------------------------------------------------------------------------

def diagnose_mesh(lon, lat, triangles):
    """
    Summarize mesh sanity in one or two lines: triangle edge lengths
    (spurious triangles bridging distant nodes show up as extreme
    outliers) and total triangle area vs. the bounding box (a valid,
    non-overlapping mesh can never cover more area than its bounding box).
    Returns a human-readable string.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    kx = np.cos(np.deg2rad(np.nanmean(lat)))
    x, y = lon[triangles] * kx, lat[triangles]
    e = np.stack([np.hypot(x[:, 0] - x[:, 1], y[:, 0] - y[:, 1]),
                  np.hypot(x[:, 1] - x[:, 2], y[:, 1] - y[:, 2]),
                  np.hypot(x[:, 2] - x[:, 0], y[:, 2] - y[:, 0])], axis=1)
    longest = e.max(axis=1)
    med = float(np.median(longest))
    n_long = int((longest > 50 * med).sum())
    area = 0.5 * np.abs((x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0])
                        - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0]))
    bbox = (np.ptp(lon) * kx) * np.ptp(lat)
    ratio = float(area.sum() / bbox) if bbox > 0 else np.nan
    msg = (f"longest triangle edge: median {med:.4f} deg, max {longest.max():.3f} deg; "
           f"{n_long} triangle(s) have an edge >50x the median; "
           f"total triangle area / bounding-box area = {ratio:.2f} "
           f"(must be <= 1 for a non-overlapping mesh)")
    if n_long:
        msg += (". Extreme long edges usually mean triangle connectivity "
                "does not match the node coordinates (e.g. nodes re-ordered "
                "or subset without remapping the triangle indices)")
    elif ratio > 1.0:
        msg += ". Area > bounding box => triangles genuinely overlap"
    return msg


class KDTriangleLocator:
    """
    Point-in-triangle search that tolerates overlapping triangles (unlike
    matplotlib's TrapezoidMapTriFinder, which refuses them). Candidates
    come from the triangles touching the nearest nodes plus the triangles
    with the nearest centroids; of those containing the point, the
    smallest wins. Requires scipy.
    """

    def __init__(self, lon, lat, triangles):
        from scipy.spatial import cKDTree
        self.lon, self.lat, self.tri = lon, lat, np.asarray(triangles)
        self.kx = float(np.cos(np.deg2rad(np.nanmean(lat))))
        cx = lon[self.tri].mean(axis=1) * self.kx
        cy = lat[self.tri].mean(axis=1)
        self.ctree = cKDTree(np.column_stack([cx, cy]))
        self.ntree = cKDTree(np.column_stack([lon * self.kx, lat]))
        flat = self.tri.ravel()
        order = np.argsort(flat, kind="stable")
        self.adj = order // 3                       # node -> incident triangles (CSR)
        self.ptr = np.concatenate([[0], np.cumsum(np.bincount(flat, minlength=len(lon)))])

    def _bary(self, cands, x, y):
        t = self.tri[cands]
        xa, xb, xc = self.lon[t[:, 0]], self.lon[t[:, 1]], self.lon[t[:, 2]]
        ya, yb, yc = self.lat[t[:, 0]], self.lat[t[:, 1]], self.lat[t[:, 2]]
        det = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc)
        with np.errstate(divide="ignore", invalid="ignore"):
            l1 = ((yb - yc) * (x - xc) + (xc - xb) * (y - yc)) / det
            l2 = ((yc - ya) * (x - xc) + (xa - xc) * (y - yc)) / det
        return np.column_stack([l1, l2, 1.0 - l1 - l2]), np.abs(det)

    def find(self, x, y):
        """Return (triangle_index, barycentric[3]) or (-1, None) if outside."""
        p = [x * self.kx, y]
        for n_nodes, n_cent in ((4, 24), (16, 256)):
            _, nn = self.ntree.query(p, k=n_nodes)
            _, cc = self.ctree.query(p, k=n_cent)
            parts = [self.adj[self.ptr[n]:self.ptr[n + 1]] for n in np.atleast_1d(nn)
                     if n < len(self.ptr) - 1]
            cands = np.unique(np.concatenate(parts + [np.atleast_1d(cc)]))
            bary, size = self._bary(cands, x, y)
            with np.errstate(invalid="ignore"):
                inside = np.nanmin(bary, axis=1) >= -1e-9
            if inside.any():
                # if several (overlapping) triangles contain the point, the
                # smallest one is the most local / least likely to be bogus
                pick = np.where(inside)[0][np.argmin(size[inside])]
                return int(cands[pick]), bary[pick]
        return -1, None


class UnstructuredVelocityField:
    """
    Wraps an unstructured triangular-mesh current field and provides
    velocity_at(lon, lat, t) with linear spatial (barycentric) and
    linear temporal interpolation.

    The containing triangle is found with matplotlib's (fast) trifinder
    when the mesh is clean, and with a KD-tree search (KDTriangleLocator)
    if matplotlib rejects the mesh as invalid/overlapping.

    Parameters
    ----------
    lon, lat : (n_nodes,) node coordinates, degrees
    triangles : (n_tri, 3) 0-based node connectivity
    times : (n_time,) np.datetime64 array of snapshot times
    u, v : (n_time, n_nodes) node-centered velocity components, m/s
           (use elements_to_nodes() first if your source data is
           element/cell-centered, as FVCOM's u/v typically are)
    """

    def __init__(self, lon, lat, triangles, times, u, v):
        self.lon = np.asarray(lon, dtype=float)
        self.lat = np.asarray(lat, dtype=float)
        self.triangles = np.asarray(triangles)
        validate_mesh(self.lon, self.lat, self.triangles)
        self.triangulation = mtri.Triangulation(self.lon, self.lat, self.triangles)
        self.trifinder = None
        self.locator = None
        try:
            self.trifinder = self.triangulation.get_trifinder()
        except RuntimeError as e:
            print(f"WARNING: matplotlib rejected the mesh ({e}); some triangles "
                  f"overlap/intersect.\n  Mesh diagnostics: "
                  f"{diagnose_mesh(self.lon, self.lat, self.triangles)}\n"
                  f"  Falling back to a KD-tree point locator (needs scipy). "
                  f"Results are only trustworthy where the overlap is "
                  f"harmless -- check the diagnostics above.", file=sys.stderr)
            try:
                self.locator = KDTriangleLocator(self.lon, self.lat, self.triangles)
            except ImportError as ie:
                raise ValueError(
                    "The mesh has overlapping triangles and the fallback "
                    "locator needs scipy (pip install scipy).") from ie
        self.times = np.asarray(times, dtype="datetime64[ns]")
        self.t_seconds = (self.times - self.times[0]) / np.timedelta64(1, "s")
        self.u = np.asarray(u, dtype=np.float32 if np.asarray(u).dtype == np.float32 else float)
        self.v = np.asarray(v, dtype=np.float32 if np.asarray(v).dtype == np.float32 else float)

    def _locate(self, x, y):
        """Triangle index and barycentric weights for a point, or (-1, None)."""
        if self.locator is not None:
            return self.locator.find(x, y)
        ti = int(self.trifinder(x, y))
        if ti < 0:
            return -1, None
        t = self.triangles[ti]
        xa, xb, xc = self.lon[t]
        ya, yb, yc = self.lat[t]
        det = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc)
        l1 = ((yb - yc) * (x - xc) + (xc - xb) * (y - yc)) / det
        l2 = ((yc - ya) * (x - xc) + (xa - xc) * (y - yc)) / det
        return ti, np.array([l1, l2, 1.0 - l1 - l2])

    def velocity_at(self, lon_pt, lat_pt, t_sec):
        """Return (u, v) in m/s at a point/time, or (nan, nan) if the
        point is outside the mesh or the time is outside the record."""
        if t_sec < self.t_seconds[0] or t_sec > self.t_seconds[-1]:
            return np.nan, np.nan

        ti, bary = self._locate(lon_pt, lat_pt)
        if ti < 0:
            return np.nan, np.nan
        nodes = self.triangles[ti]

        idx1 = int(np.searchsorted(self.t_seconds, t_sec, side="right"))
        idx1 = min(max(idx1, 1), len(self.t_seconds) - 1)
        idx0 = idx1 - 1
        t0, t1 = self.t_seconds[idx0], self.t_seconds[idx1]
        w = 0.0 if t1 == t0 else (t_sec - t0) / (t1 - t0)

        u = sum(bary[k] * ((1 - w) * self.u[idx0, nodes[k]] + w * self.u[idx1, nodes[k]])
                for k in range(3))
        v = sum(bary[k] * ((1 - w) * self.v[idx0, nodes[k]] + w * self.v[idx1, nodes[k]])
                for k in range(3))
        return float(u), float(v)


def faces_to_triangles(conn_da, n_nodes):
    """
    Convert a face->node connectivity array into an (n_tri, 3) 0-based
    triangle array, plus `tri_face` mapping each triangle back to the
    face/element it came from.

    Handles: (3, n) FVCOM 'nv' layout or (n, k) UGRID/D-Flow FM layout,
    1-based or 0-based indices (uses a start_index attribute if present),
    fill values (NaN, negatives, or the _FillValue attribute) used to pad
    faces with fewer than k nodes, and quads/polygons (fan-triangulated).
    """

    conn = np.asarray(conn_da.values, dtype=float)
    if conn.ndim == 3:
        conn=conn[0,:,:]
        print('detected 3 dimensions for connectivity, used index 0 of axis 0')
        print('Expected if model is FVCOM')
        
    if conn.ndim != 2:
        raise ValueError(f"connectivity must be 2-D, got shape {conn.shape}")
        raise ValueError("used first index of first dimension")
    if conn.shape[0] <= 10 and conn.shape[0] < conn.shape[1]:
        conn = conn.T  # (k, n_faces) -> (n_faces, k)

    fill = conn_da.attrs.get("_FillValue", None)
    invalid = ~np.isfinite(conn) | (conn < 0)
    if fill is not None:
        invalid |= conn == fill
    invalid |= conn >= 1e9  # typical unmasked fill sentinels

    valid_vals = conn[~invalid]
    start_index = conn_da.attrs.get("start_index", None)
    if start_index is None:
        start_index = 1 if valid_vals.size and valid_vals.min() >= 1 and \
            valid_vals.max() == n_nodes else 0
    conn = conn - int(start_index)
    invalid |= conn < 0
    conn = np.where(invalid, -1, conn).astype(np.int64)

    tris, faces = [], []
    n_valid = (conn >= 0).sum(axis=1)
    for k in np.unique(n_valid):
        if k < 3:
            continue
        idx = np.where(n_valid == k)[0]
        rows = conn[idx][:, :k]
        # valid nodes are assumed to be packed to the left of each row
        for j in range(1, k - 1):
            tris.append(np.stack([rows[:, 0], rows[:, j], rows[:, j + 1]], axis=1))
            faces.append(idx)
    if not tris:
        raise ValueError("No valid faces (>=3 nodes) found in connectivity array")
    triangles = np.concatenate(tris)
    tri_face = np.concatenate(faces)
    order = np.argsort(tri_face, kind="stable")
    return triangles[order], tri_face[order]


def repair_mesh(lon, lat, triangles):
    """
    Fix the mesh defects that make matplotlib's TriFinder fail with
    "Triangulation is invalid": nodes with identical coordinates are
    merged, and triangles that become degenerate (repeated vertex or
    ~zero area) or duplicated are dropped.

    Returns (triangles, keep_mask, info) where keep_mask indexes into the
    *input* triangle array.
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    tri = np.asarray(triangles).copy()
    info = {}

    coords = np.round(np.stack([lon, lat], axis=1), 9)
    _, first_idx, inverse = np.unique(coords, axis=0, return_index=True,
                                       return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)
    n_merged = len(lon) - len(first_idx)
    if n_merged:
        tri = first_idx[inverse][tri]
    info["merged_duplicate_nodes"] = int(n_merged)

    keep = np.ones(len(tri), dtype=bool)
    repeated = (tri[:, 0] == tri[:, 1]) | (tri[:, 1] == tri[:, 2]) | (tri[:, 0] == tri[:, 2])
    keep &= ~repeated
    info["repeated_vertex_triangles"] = int(repeated.sum())

    x, y = lon[tri], lat[tri]
    area2 = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    zero_area = np.abs(area2) < 1e-14
    keep &= ~zero_area
    info["zero_area_triangles"] = int((zero_area & ~repeated).sum())

    key = np.sort(tri, axis=1)
    _, first_tri = np.unique(key, axis=0, return_index=True)
    is_first = np.zeros(len(tri), dtype=bool)
    is_first[first_tri] = True
    dup = ~is_first & keep
    keep &= ~dup
    info["duplicate_triangles"] = int(dup.sum())

    return tri[keep], keep, info


def elements_to_nodes(values_elem, triangles, n_nodes):
    """
    Average element/cell-centered values onto mesh nodes (simple
    unweighted mean of the surrounding elements). Needed because many
    unstructured models (e.g. FVCOM) store u, v at element centers,
    while LinearTriInterpolator needs node-centered values.

    values_elem: (..., n_elements) array (any number of leading dims,
                 e.g. time); triangles: (n_elements, 3) connectivity.
    """
    values_elem = np.asarray(values_elem)
    orig_shape = values_elem.shape
    flat = values_elem.reshape(-1, orig_shape[-1])  # (n_frames, n_elem)

    node_sum = np.zeros((flat.shape[0], n_nodes))
    node_count = np.zeros(n_nodes)
    for corner in range(3):
        np.add.at(node_count, triangles[:, corner], 1)
    for f in range(flat.shape[0]):
        vals = np.zeros(n_nodes)
        for corner in range(3):
            np.add.at(vals, triangles[:, corner], flat[f])
        node_sum[f] = vals
    node_count[node_count == 0] = 1
    node_vals = node_sum / node_count
    return node_vals.reshape(orig_shape[:-1] + (n_nodes,))


# ---------------------------------------------------------------------------
# 2. Data loading
# ---------------------------------------------------------------------------

def load_unstructured_currents(source, lon_var=None, lat_var=None,
                                connectivity_var=None, u_var=None, v_var=None,
                                time_var="time", layer_index=None,
                                engine=None, consolidated=None,
                                storage_options=None,
                                t_start=None, t_end=None):
    """
    Load an unstructured-grid current field from a NetCDF or Zarr source
    (FVCOM / SCHISM / similar). Variable names are auto-detected from
    common conventions but can be overridden if your file differs.

    `source` can be:
      - a local NetCDF file path (.nc)
      - a local Zarr store path (.zarr directory) or any path when
        engine="zarr" is forced
      - a cloud-hosted Zarr store URL, e.g. "s3://bucket/store.zarr" or
        "gs://bucket/store.zarr" (requires `s3fs`/`gcsfs` installed;
        pass credentials/options via storage_options, e.g.
        {"anon": True} for public buckets)
      - an http(s) URL to a Zarr store

    engine : None (auto-detect from the extension/URL scheme), "zarr",
        or "netcdf4"/"h5netcdf" to force a particular reader.
    consolidated : passed to xr.open_zarr when engine is zarr; None lets
        xarray auto-detect consolidated metadata.

    Handles both node-centered and element-centered u/v (element-centered
    is averaged onto nodes via elements_to_nodes()). If u/v carry a
    vertical/sigma-layer dimension, layer_index selects which layer
    (default 0, typically the surface).
    """
    import xarray as xr

    is_zarr = (engine == "zarr") or (
        engine is None and (
            str(source).rstrip("/").endswith(".zarr")
            or "://" in str(source) and str(source).rstrip("/").endswith(".zarr")
        )
    )

    if is_zarr:
        open_kwargs = {}
        if consolidated is not None:
            open_kwargs["consolidated"] = consolidated
        if storage_options is not None:
            open_kwargs["storage_options"] = storage_options
        # Some Zarr stores carry stray non-array/group objects (e.g. a
        # leftover "_temp" scratch folder from whatever wrote/synced the
        # store). zarr-python (v3) warns about these while walking the
        # store's hierarchy; it's harmless -- the real variables still
        # open fine -- so it's filtered out here rather than left to
        # clutter the console on every load.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*not recognized as a component of a Zarr hierarchy.*")
            ds = xr.open_zarr(source, **open_kwargs)
    else:
        open_kwargs = {}
        if engine is not None:
            open_kwargs["engine"] = engine
        ds = xr.open_dataset(source, **open_kwargs)

    lon_var = lon_var or next((n for n in (
        "lon", "x", "mesh2d_node_x", "node_x", "lonc") if n in ds), None)
    lat_var = lat_var or next((n for n in (
        "lat", "y", "mesh2d_node_y", "node_y", "latc") if n in ds), None)
    connectivity_var = connectivity_var or next((n for n in (
        "nv", "triangles", "tri", "element", "faces", "cells",
        "mesh2d_face_nodes", "face_nodes", "face_node_connectivity")
        if n in ds), None)
    u_var = u_var or next((n for n in (
        "u", "ua", "water_u", "mesh2d_ucx", "ucx") if n in ds), None)
    v_var = v_var or next((n for n in (
        "v", "va", "water_v", "mesh2d_ucy", "ucy") if n in ds), None)

    if None in (lon_var, lat_var, connectivity_var, u_var, v_var):
        raise ValueError(
            "Could not auto-detect required variables in this file. "
            "Pass lon_var/lat_var/connectivity_var/u_var/v_var explicitly. "
            f"Found so far: lon={lon_var}, lat={lat_var}, "
            f"connectivity={connectivity_var}, u={u_var}, v={v_var}. "
            f"Variables in file: {list(ds.variables)[:40]}")

    # read only the time window needed (files are often many GB)
    all_times = ds[time_var].values
    if t_start is not None or t_end is not None:
        i0, i1 = 0, len(all_times) - 1
        if t_start is not None:
            i0 = max(int(np.searchsorted(all_times, np.datetime64(t_start), "right")) - 1, 0)
        if t_end is not None:
            i1 = min(int(np.searchsorted(all_times, np.datetime64(t_end), "left")),
                     len(all_times) - 1)
        if i1 < i0:
            raise ValueError(
                f"Requested window {t_start} .. {t_end} is outside the file's "
                f"time range {all_times[0]} .. {all_times[-1]}")
        ds = ds.isel({time_var: slice(i0, i1 + 1)})

    lon = ds[lon_var].values
    lat = ds[lat_var].values
    n_nodes = lon.size

    # connectivity -> triangles (fan-triangulates quads/polygons, drops
    # fill values, converts 1-based to 0-based)
    triangles, tri_face = faces_to_triangles(
        ds[connectivity_var], n_nodes=n_nodes)
    n_faces = int(tri_face.max()) + 1

    # repair coincident nodes / degenerate / duplicate triangles
    triangles, keep, info = repair_mesh(lon, lat, triangles)
    tri_face = tri_face[keep]
    if any(info.values()):
        print(f"Mesh repair: {info}")

    u_da, v_da = ds[u_var], ds[v_var]
    for dim in ("sigma", "siglay", "siglev", "layer", "lev", "level",
                "depth", "nmesh2d_layer", "laydim"):
        if dim in u_da.dims:
            if layer_index is None:
                # auto-pick the surface layer using z_levels if available
                idx = 0
                if "z_levels" in ds:
                    z = np.asarray(ds["z_levels"].values, dtype=float)
                    idx = int(np.argmax(z) if z.max() <= 0 else np.argmin(z))
                    print(f"Vertical levels ({dim}): {np.round(z, 2).tolist()}")
                else:
                    print(f"No z_levels variable; using {dim} index 0")
                print(f"Using {dim} index {idx} as the surface layer "
                      f"(override with --layer-index)")
            else:
                idx = layer_index
            u_da = u_da.isel({dim: idx})
            v_da = v_da.isel({dim: idx})

    times = ds[time_var].values
    u = np.asarray(u_da.values, dtype=float)
    v = np.asarray(v_da.values, dtype=float)
    u = np.where(np.abs(u) > 1e5, np.nan, u)  # unmasked fill values
    v = np.where(np.abs(v) > 1e5, np.nan, v)

    if u.shape[-1] == n_faces and u.shape[-1] != n_nodes:
        # element/face-centered -> node-centered (via the triangles)
        u = elements_to_nodes(u[..., tri_face], triangles, n_nodes)
        v = elements_to_nodes(v[..., tri_face], triangles, n_nodes)
    elif u.shape[-1] != n_nodes:
        raise ValueError(
            f"u/v last dimension ({u.shape[-1]}) matches neither the number "
            f"of nodes ({n_nodes}) nor faces/elements ({n_faces}). If your "
            f"variables are (node, time) instead of (time, node), transpose "
            f"them; if they have a layer dimension not in the recognized "
            f"list, subset it before loading.")

    if "wet" in ds and ds["wet"].shape == u.shape:
        wet = np.asarray(ds["wet"].values) > 0
        u = np.where(wet, u, 0.0)   # no flow at dry nodes
        v = np.where(wet, v, 0.0)

    # NaNs in u/v (e.g. dry cells) would make the tracker think the
    # particle left the domain -- treat them as zero velocity instead
    u = np.nan_to_num(u, nan=0.0)
    v = np.nan_to_num(v, nan=0.0)

    return lon, lat, triangles, times, u, v


def synthetic_unstructured_currents(n_points=800, n_time=25, seed=0,
                                     lon_center=-74.0, lat_center=39.0,
                                     domain_deg=2.0, period_hours=48,
                                     speed_m_s=0.3):
    """
    Build a synthetic unstructured (Delaunay-triangulated) mesh covering
    a small domain with a time-varying solid-body rotation ("gyre") plus
    a modest steady drift, so the tracker can be exercised and sanity
    checked (a released particle should trace out a roughly circular
    path) without needing a real model file.
    """
    rng = np.random.default_rng(seed)
    lon = lon_center + domain_deg * (rng.random(n_points) - 0.5)
    lat = lat_center + domain_deg * (rng.random(n_points) - 0.5) * 0.8
    triangulation = mtri.Triangulation(lon, lat)  # Delaunay via qhull
    triangles = triangulation.triangles

    start = np.datetime64("2020-01-01T00:00:00")
    times = start + np.arange(n_time) * np.timedelta64(
        int(period_hours * 3600 / (n_time - 1)), "s")

    omega = 2 * np.pi / (period_hours * 3600.0)  # rad/s -- every radius
    # completes one full loop in exactly period_hours (solid-body rotation)
    dx_deg = lon - lon_center
    dy_deg = lat - lat_center
    # convert deg offsets to meters for a physically-scaled rotation speed
    dx_m = dx_deg * R_EARTH * np.cos(np.deg2rad(lat_center)) * np.pi / 180
    dy_m = dy_deg * R_EARTH * np.pi / 180

    u = np.zeros((n_time, n_points))
    v = np.zeros((n_time, n_points))
    for k in range(n_time):
        # solid-body rotation (tangential velocity = omega x r) + a small
        # steady drift so trajectories aren't perfectly closed loops
        u[k] = -omega * dy_m + 0.15 * speed_m_s
        v[k] = omega * dx_m + 0.05 * speed_m_s

    return lon, lat, triangles, times, u, v


# ---------------------------------------------------------------------------
# 3. RK4 particle tracking
# ---------------------------------------------------------------------------

def _deg_per_sec(u_m_s, v_m_s, lat_deg):
    """Convert m/s velocity components to deg/s at a given latitude."""
    dlon_dt = u_m_s / (R_EARTH * np.cos(np.deg2rad(lat_deg))) * (180.0 / np.pi)
    dlat_dt = v_m_s / R_EARTH * (180.0 / np.pi)
    return dlon_dt, dlat_dt


def track_particle(field, lon0, lat0, release_time, duration_hours,
                    dt_seconds=600, output_interval_seconds=3600):
    """
    Advect one particle released at (lon0, lat0, release_time) for
    duration_hours using RK4 integration of the velocity field.

    Returns a dict with arrays 'times' (datetime64), 'lon', 'lat',
    plus 'status' ('completed' or 'exited_domain') and, if exited,
    'exit_time'.
    """
    release_time = np.datetime64(release_time)
    t0_sec = float((release_time - field.times[0]) / np.timedelta64(1, "s"))
    duration_sec = duration_hours * 3600.0
    t_end_sec = t0_sec + duration_sec
    record_end = float(field.t_seconds[-1])
    truncated = t_end_sec > record_end
    t_end_sec = min(t_end_sec, record_end)

    out_lon, out_lat, out_times = [lon0], [lat0], [release_time]
    lon, lat, t = lon0, lat0, t0_sec
    next_output = t0_sec + output_interval_seconds
    status = "completed"
    exit_time = None

    while t < t_end_sec:
        dt = min(dt_seconds, t_end_sec - t)

        u1, v1 = field.velocity_at(lon, lat, t)
        if np.isnan(u1):
            status, exit_time = "exited_domain", field.times[0] + np.timedelta64(int(t), "s")
            break
        dlon1, dlat1 = _deg_per_sec(u1, v1, lat)

        u2, v2 = field.velocity_at(lon + 0.5 * dt * dlon1, lat + 0.5 * dt * dlat1, t + 0.5 * dt)
        if np.isnan(u2):
            status, exit_time = "exited_domain", field.times[0] + np.timedelta64(int(t + 0.5 * dt), "s")
            break
        dlon2, dlat2 = _deg_per_sec(u2, v2, lat + 0.5 * dt * dlat1)

        u3, v3 = field.velocity_at(lon + 0.5 * dt * dlon2, lat + 0.5 * dt * dlat2, t + 0.5 * dt)
        if np.isnan(u3):
            status, exit_time = "exited_domain", field.times[0] + np.timedelta64(int(t + 0.5 * dt), "s")
            break
        dlon3, dlat3 = _deg_per_sec(u3, v3, lat + 0.5 * dt * dlat2)

        u4, v4 = field.velocity_at(lon + dt * dlon3, lat + dt * dlat3, t + dt)
        if np.isnan(u4):
            status, exit_time = "exited_domain", field.times[0] + np.timedelta64(int(t + dt), "s")
            break
        dlon4, dlat4 = _deg_per_sec(u4, v4, lat + dt * dlat3)

        lon += (dt / 6.0) * (dlon1 + 2 * dlon2 + 2 * dlon3 + dlon4)
        lat += (dt / 6.0) * (dlat1 + 2 * dlat2 + 2 * dlat3 + dlat4)
        t += dt

        if t >= next_output or t >= t_end_sec:
            out_lon.append(lon)
            out_lat.append(lat)
            out_times.append(field.times[0] + np.timedelta64(int(round(t)), "s"))
            next_output += output_interval_seconds

    if status == "completed" and truncated:
        status = "end_of_record"

    return {
        "times": np.array(out_times, dtype="datetime64[ns]"),
        "lon": np.array(out_lon),
        "lat": np.array(out_lat),
        "status": status,
        "exit_time": exit_time,
    }


# ---------------------------------------------------------------------------
# 4. Output: CSV + plot
# ---------------------------------------------------------------------------

def save_trajectory_csv(traj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("time,lon,lat\n")
        for t, lo, la in zip(traj["times"], traj["lon"], traj["lat"]):
            f.write(f"{str(t)},{lo:.6f},{la:.6f}\n")
    print(f"Saved trajectory CSV to {path}  (status: {traj['status']})")


def plot_trajectories(field, trajectories, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))

    # only draw the part of the mesh near the trajectories (big meshes have
    # hundreds of thousands of triangles)
    all_lon = np.concatenate([t["lon"] for t in trajectories])
    all_lat = np.concatenate([t["lat"] for t in trajectories])
    pad_lon = max(0.15 * np.ptp(all_lon), 0.05)
    pad_lat = max(0.15 * np.ptp(all_lat), 0.05)
    x0, x1 = all_lon.min() - pad_lon, all_lon.max() + pad_lon
    y0, y1 = all_lat.min() - pad_lat, all_lat.max() + pad_lat
    in_box = (field.lon >= x0) & (field.lon <= x1) & (field.lat >= y0) & (field.lat <= y1)
    tri_in = in_box[field.triangles].all(axis=1)
    if tri_in.any():
        ax.triplot(field.lon, field.lat, field.triangles[tri_in],
                   linewidth=0.2, color="0.75", zorder=1)
    nodes = np.where(in_box)[0]
    nodes = nodes[::max(1, len(nodes) // 400)]
    ax.quiver(field.lon[nodes], field.lat[nodes],
              field.u[0, nodes], field.v[0, nodes],
              color="0.6", width=0.002, zorder=2, label="velocity @ first snapshot")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

    cmap = plt.get_cmap("tab10")
    for i, traj in enumerate(trajectories):
        color = cmap(i % 10)
        ax.plot(traj["lon"], traj["lat"], "-", color=color, linewidth=1.6, zorder=3)
        ax.scatter(traj["lon"], traj["lat"], s=10, color=color, zorder=4)
        ax.scatter(traj["lon"][0], traj["lat"][0], marker="o", s=70,
                   facecolor=color, edgecolor="k", zorder=5,
                   label=f"start {i}" if len(trajectories) > 1 else "release")
        end_label = (f"end {i} ({traj['status']})" if len(trajectories) > 1
                     else f"end ({traj['status']})")
        if traj["status"] == "exited_domain":
            ax.scatter(traj["lon"][-1], traj["lat"][-1], marker="x", s=90,
                       color=color, linewidths=2.5, zorder=5, label=end_label)
        else:
            ax.scatter(traj["lon"][-1], traj["lat"][-1], marker="s", s=90,
                       facecolor=color, edgecolor="k", zorder=5, label=end_label)

    mean_lat = float(np.mean(all_lat))
    ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Offline Lagrangian particle trajectory")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved trajectory plot to {out_path}")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", help="NetCDF file or Zarr store (local path, "
                    "or s3://, gs://, http(s):// URL) with unstructured-grid u/v currents")
    p.add_argument("--engine", choices=["zarr", "netcdf4", "h5netcdf"], default=None,
                    help="Force a reader instead of auto-detecting from --source")
    p.add_argument("--zarr-anon", action="store_true",
                    help="Use anonymous/unsigned access for a cloud Zarr store (public bucket)")
    p.add_argument("--synthetic", action="store_true",
                    help="Use a synthetic rotating-gyre mesh instead of --source")
    p.add_argument("--lon", type=float, nargs="+", required=True,
                    help="Release longitude(s), degrees")
    p.add_argument("--lat", type=float, nargs="+", required=True,
                    help="Release latitude(s), degrees")
    p.add_argument("--time", type=str, nargs="+", required=True,
                    help="Release time(s), ISO8601, e.g. 2020-01-01T00:00:00")
    p.add_argument("--duration-hours", type=float, default=48.0)
    p.add_argument("--dt-seconds", type=float, default=600.0,
                    help="RK4 integration time step")
    p.add_argument("--output-interval-hours", type=float, default=1.0,
                    help="How often to record a trajectory point")
    p.add_argument("--layer-index", type=int, default=None,
                    help="Vertical layer index (default: auto-pick the surface "
                         "layer via z_levels if present, else 0)")
    p.add_argument("--out-plot", default="trajectory.png")
    p.add_argument("--out-csv", default="trajectory.csv")
    args = p.parse_args()

    n = len(args.lon)
    if not (len(args.lat) == n and len(args.time) == n):
        p.error("--lon, --lat, and --time must all have the same number of values")

    if args.synthetic or not args.source:
        if not args.synthetic:
            print("No --source given; using synthetic demo current field.", file=sys.stderr)
        lon, lat, triangles, times, u, v = synthetic_unstructured_currents()
    else:
        try:
            print(f"Loading unstructured currents from {args.source} ...")
            storage_options = {"anon": True} if args.zarr_anon else None
            rel = [np.datetime64(t) for t in args.time]
            win_start = min(rel)
            win_end = max(rel) + np.timedelta64(int(args.duration_hours * 3600), "s")
            lon, lat, triangles, times, u, v = load_unstructured_currents(
                args.source, layer_index=args.layer_index, engine=args.engine,
                storage_options=storage_options, t_start=win_start, t_end=win_end)
        except Exception as e:
            print(f"Could not load {args.source} ({e}).\n"
                  f"Falling back to synthetic demo data so you can still "
                  f"test the tracker.", file=sys.stderr)
            lon, lat, triangles, times, u, v = synthetic_unstructured_currents()

    print(f"Mesh: {len(lon)} nodes, {len(triangles)} triangles, "
          f"{len(times)} time snapshots ({times[0]} to {times[-1]})")

    field = UnstructuredVelocityField(lon, lat, triangles, times, u, v)

    trajectories = []
    for lon0, lat0, t0 in zip(args.lon, args.lat, args.time):
        print(f"Tracking particle released at ({lon0}, {lat0}) at {t0} "
              f"for {args.duration_hours} h ...")
        traj = track_particle(field, lon0, lat0, t0, args.duration_hours,
                               dt_seconds=args.dt_seconds,
                               output_interval_seconds=args.output_interval_hours * 3600)
        if traj["status"] == "exited_domain":
            print(f"  -> exited the meshed domain at {traj['exit_time']}")
        elif traj["status"] == "end_of_record":
            print(f"  -> reached the end of the velocity record "
                  f"({field.times[-1]}) before {args.duration_hours} h elapsed; "
                  f"final position ({traj['lon'][-1]:.4f}, {traj['lat'][-1]:.4f})")
        else:
            print(f"  -> completed: final position ({traj['lon'][-1]:.4f}, "
                  f"{traj['lat'][-1]:.4f})")
        trajectories.append(traj)

    if n == 1:
        save_trajectory_csv(trajectories[0], args.out_csv)
    else:
        for i, traj in enumerate(trajectories):
            base, ext = os.path.splitext(args.out_csv)
            save_trajectory_csv(traj, f"{base}_{i}{ext}")

    plot_trajectories(field, trajectories, args.out_plot)


if __name__ == "__main__":
    main()
