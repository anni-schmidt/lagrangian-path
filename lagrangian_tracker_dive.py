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

# names a vertical (level/layer) dimension of u/v may have
LEVEL_DIMS = ("sigma", "siglay", "siglev", "layer", "lev", "level",
              "depth", "nmesh2d_layer", "laydim")


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

    def __init__(self, lon, lat, triangles, times, u, v,
                 level_depths=None, bottom_depth=None):
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
        self.u = self._as_float(u)
        self.v = self._as_float(v)
        self.level_depths = (None if level_depths is None
                             else np.asarray(level_depths, dtype=float))
        self.bottom_depth = (None if bottom_depth is None
                             else np.asarray(bottom_depth, dtype=float))
        if self.u.ndim == 3:
            if self.level_depths is None or len(self.level_depths) != self.u.shape[1]:
                raise ValueError(
                    "3-D u/v (time, level, node) need level_depths with one "
                    "positive-down depth (m) per level, in increasing order")
        elif self.u.ndim != 2:
            raise ValueError(f"u/v must be (time, node) or (time, level, node); got {self.u.shape}")

    @staticmethod
    def _as_float(a):
        a = np.asarray(a)
        return a if a.dtype == np.float32 else a.astype(float)

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

    def plane_at(self, time_idx, depth=0.0):
        """Node-wise (u, v) at one snapshot and the model level nearest
        `depth` -- used for plotting."""
        if self.u.ndim == 3:
            lev = int(np.argmin(np.abs(self.level_depths - (depth or 0.0))))
            return np.nan_to_num(self.u[time_idx, lev]), np.nan_to_num(self.v[time_idx, lev])
        return self.u[time_idx], self.v[time_idx]

    def _profile_value(self, prof, node, depth):
        """Interpolate one node's vertical profile to `depth`. Levels that
        are NaN or below the local seabed are ignored, so a particle
        requested deeper than the water column uses the deepest valid
        level (i.e. it rides along just above the bottom)."""
        ok = np.isfinite(prof)
        if self.bottom_depth is not None:
            ok &= self.level_depths <= self.bottom_depth[node] + 1e-6
        if not ok.any():
            return 0.0
        return float(np.interp(depth, self.level_depths[ok], prof[ok]))

    def velocity_at(self, lon_pt, lat_pt, t_sec, depth=None):
        """Return (u, v) in m/s at a point/time (and depth in m, positive
        down, if the field has levels), or (nan, nan) if the point is
        outside the mesh or the time is outside the record."""
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

        if self.u.ndim == 2:
            u = sum(bary[k] * ((1 - w) * self.u[idx0, nodes[k]] + w * self.u[idx1, nodes[k]])
                    for k in range(3))
            v = sum(bary[k] * ((1 - w) * self.v[idx0, nodes[k]] + w * self.v[idx1, nodes[k]])
                    for k in range(3))
            return float(u), float(v)

        d = self.level_depths[0] if depth is None else depth
        U = (1 - w) * self.u[idx0][:, nodes] + w * self.u[idx1][:, nodes]   # (n_level, 3)
        V = (1 - w) * self.v[idx0][:, nodes] + w * self.v[idx1][:, nodes]
        u = sum(bary[k] * self._profile_value(U[:, k], nodes[k], d) for k in range(3))
        v = sum(bary[k] * self._profile_value(V[:, k], nodes[k], d) for k in range(3))
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
    if conn.ndim != 2:
        raise ValueError(f"connectivity must be 2-D, got shape {conn.shape}")
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
                                t_start=None, t_end=None,
                                depth_range=None, return_vertical=False,
                                use_bottom_depth=True):
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
    is averaged onto nodes via elements_to_nodes()).

    Vertical: if u/v carry a level dimension, then
      * depth_range=None -> a single layer is used (layer_index, or the
        surface layer chosen from z_levels), returning 2-D (time, node) u/v;
      * depth_range=(dmin, dmax) -> every level from the surface down to
        the first level at/below dmax is loaded, returning 3-D
        (time, level, node) u/v so the tracker can interpolate in depth.
        Set return_vertical=True to also get (level_depths, bottom_depth)
        as two extra return values (level depths in metres, positive down,
        increasing; bottom_depth per node or None).
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
    level_dim = next((d for d in LEVEL_DIMS if d in u_da.dims), None)
    level_depths = None
    bottom_depth = None

    if level_dim is not None and depth_range is not None:
        if "z_levels" not in ds:
            raise ValueError(
                "A depth schedule needs the depth of each vertical level, "
                "expected in a 'z_levels' variable, but the file has none.")
        z = np.asarray(ds["z_levels"].values, dtype=float)
        depth_pos = -z if z.max() <= 0 else z          # positive-down metres
        order = np.argsort(depth_pos)
        d_sorted = depth_pos[order]
        hi = min(int(np.searchsorted(d_sorted, depth_range[1], side="left")),
                 len(d_sorted) - 1)
        sel = order[:hi + 1]
        level_depths = d_sorted[:hi + 1]
        print(f"Model levels ({level_dim}), depth in m: {np.round(d_sorted, 2).tolist()}")
        print(f"Loading {len(sel)} levels ({level_depths[0]:g}-{level_depths[-1]:g} m) "
              f"to cover the depth schedule ({depth_range[0]:g}-{depth_range[1]:g} m)")
        u_da = u_da.isel({level_dim: sel})
        v_da = v_da.isel({level_dim: sel})
    elif level_dim is not None:
        if layer_index is None:
            # auto-pick the surface layer using z_levels if available
            idx = 0
            if "z_levels" in ds:
                z = np.asarray(ds["z_levels"].values, dtype=float)
                idx = int(np.argmax(z) if z.max() <= 0 else np.argmin(z))
                print(f"Vertical levels ({level_dim}): {np.round(z, 2).tolist()}")
            else:
                print(f"No z_levels variable; using {level_dim} index 0")
            print(f"Using {level_dim} index {idx} as the surface layer "
                  f"(override with --layer-index)")
        else:
            idx = layer_index
        u_da = u_da.isel({level_dim: idx})
        v_da = v_da.isel({level_dim: idx})

    times = ds[time_var].values
    u = np.asarray(u_da.values)
    v = np.asarray(v_da.values)
    if not np.issubdtype(u.dtype, np.floating):
        u, v = u.astype(np.float32), v.astype(np.float32)
    if level_depths is not None:                       # -> (time, level, node)
        dims = list(u_da.dims)
        t_ax = dims.index(time_var) if time_var in dims else 0
        u = np.moveaxis(u, [t_ax, dims.index(level_dim)], [0, 1])
        v = np.moveaxis(v, [t_ax, dims.index(level_dim)], [0, 1])
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

    if "wet" in ds:
        wet = np.asarray(ds["wet"].values) > 0
        if u.ndim == 3 and wet.ndim == 2 and wet.shape == (u.shape[0], u.shape[2]):
            wet = wet[:, None, :]
        if wet.ndim == u.ndim and all(w in (1, s) for w, s in zip(wet.shape, u.shape)):
            u = np.where(wet, u, 0.0)   # no flow at dry nodes
            v = np.where(wet, v, 0.0)

    if level_depths is None:
        # NaNs in u/v (e.g. dry cells) would make the tracker think the
        # particle left the domain -- treat them as zero velocity instead
        u = np.nan_to_num(u, nan=0.0)
        v = np.nan_to_num(v, nan=0.0)
    elif use_bottom_depth and "depth" in ds and ds["depth"].shape == (n_nodes,):
        bottom_depth = np.asarray(ds["depth"].values, dtype=float)
        if np.nanmedian(bottom_depth) < 0:              # elevation convention
            bottom_depth = -bottom_depth
        print(f"Using 'depth' as seabed depth per node (median "
              f"{np.nanmedian(bottom_depth):.1f} m, positive down); levels "
              f"below the seabed are ignored (disable with --no-bottom-mask)")

    if return_vertical:
        return lon, lat, triangles, times, u, v, level_depths, bottom_depth
    return lon, lat, triangles, times, u, v


def synthetic_unstructured_currents(n_points=800, n_time=25, seed=0,
                                     lon_center=-74.0, lat_center=39.0,
                                     domain_deg=2.0, period_hours=48,
                                     speed_m_s=0.3, vertical=False):
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

    if vertical:
        # sheared gyre: rotation weakens with depth, vanishes at 50 m and
        # reverses below; a steady drift acts at every depth. Seabed shoals
        # towards the south (20 m) and deepens towards the north (150 m).
        levels = np.array([0, 5, 10, 20, 30, 50, 75, 100], dtype=float)
        shear = 1.0 - levels / 50.0
        u = np.zeros((n_time, len(levels), n_points))
        v = np.zeros_like(u)
        for k in range(n_time):
            u[k] = shear[:, None] * (-omega * dy_m)[None, :] + 0.15 * speed_m_s
            v[k] = shear[:, None] * (omega * dx_m)[None, :] + 0.05 * speed_m_s
        bottom = 20.0 + 130.0 * (lat - lat.min()) / np.ptp(lat)
        below = levels[:, None] > bottom[None, :]
        u[:, below] = np.nan           # no data below the seabed
        v[:, below] = np.nan
        return lon, lat, triangles, times, u, v, levels, bottom

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

class DepthSchedule:
    """
    A particle's depth (m, positive down) as a user-specified function of
    elapsed time since release -- e.g. a diel vertical migration pattern,
    or a one-off descent to a fixed depth. This *prescribes* depth; it
    does not simulate buoyancy or vertical velocity. The horizontal
    currents used to advect the particle are looked up at whatever depth
    the schedule says the particle is at, at each moment.

    Parameters
    ----------
    hours, depths : breakpoints (elapsed hours since release, depth in m,
        positive down). Need not start at hour 0 -- depth is held constant
        at depths[0] for any time before hours[0].
    mode : "step" (depth jumps instantly at each breakpoint, held constant
        in between -- the default, matching "change depth at these
        intervals") or "linear" (ramps smoothly between breakpoints).
    cycle_hours : if given, the schedule repeats with this period (e.g. a
        24 h schedule for diel vertical migration repeated every day),
        using elapsed-time modulo cycle_hours.
    """

    def __init__(self, hours, depths, mode="step", cycle_hours=None):
        hours = np.asarray(hours, dtype=float)
        depths = np.asarray(depths, dtype=float)
        if len(hours) != len(depths) or len(hours) == 0:
            raise ValueError("depth schedule needs matching, non-empty hours/depths")
        order = np.argsort(hours)
        self.hours, self.depths = hours[order], depths[order]
        if mode not in ("step", "linear"):
            raise ValueError(f"depth schedule mode must be 'step' or 'linear', got {mode!r}")
        self.mode = mode
        self.cycle_hours = cycle_hours
        if cycle_hours is not None and self.hours[-1] > cycle_hours:
            raise ValueError(
                f"depth schedule's last breakpoint ({self.hours[-1]} h) is "
                f"past --depth-cycle-hours ({cycle_hours} h)")

    def depth_at(self, elapsed_seconds):
        h = elapsed_seconds / 3600.0
        if self.cycle_hours:
            h = h % self.cycle_hours
        if self.mode == "linear":
            return float(np.interp(h, self.hours, self.depths))
        idx = np.searchsorted(self.hours, h, side="right") - 1
        idx = int(np.clip(idx, 0, len(self.hours) - 1))
        return float(self.depths[idx])

    @classmethod
    def from_cli_tokens(cls, tokens, mode="step", cycle_hours=None):
        """Parse ["0:0", "6:20", "12:50"] (hour:depth_m) tokens."""
        hours, depths = [], []
        for tok in tokens:
            if ":" not in tok:
                raise ValueError(
                    f"depth schedule entry {tok!r} must be HOURS:DEPTH_M, e.g. '6:20'")
            h_str, d_str = tok.split(":", 1)
            try:
                hours.append(float(h_str))
                depths.append(float(d_str))
            except ValueError:
                raise ValueError(f"depth schedule entry {tok!r} is not HOURS:DEPTH_M")
        return cls(hours, depths, mode=mode, cycle_hours=cycle_hours)


def _deg_per_sec(u_m_s, v_m_s, lat_deg):
    """Convert m/s velocity components to deg/s at a given latitude."""
    dlon_dt = u_m_s / (R_EARTH * np.cos(np.deg2rad(lat_deg))) * (180.0 / np.pi)
    dlat_dt = v_m_s / R_EARTH * (180.0 / np.pi)
    return dlon_dt, dlat_dt


def track_particle(field, lon0, lat0, release_time, duration_hours,
                    dt_seconds=600, output_interval_seconds=3600,
                    depth_fn=None):
    """
    Advect one particle released at (lon0, lat0, release_time) for
    duration_hours using RK4 integration of the velocity field.

    depth_fn(elapsed_seconds) -> depth_m (positive down), if given,
    prescribes the particle's depth as a function of time since release
    (see DepthSchedule) -- horizontal currents are then looked up at that
    depth at every RK4 stage. None means stay at the surface/single loaded
    layer (unchanged prior behavior).

    Returns a dict with arrays 'times' (datetime64), 'lon', 'lat',
    'depth' (m, positive down; zeros if depth_fn is None), plus 'status'
    ('completed', 'end_of_record', or 'exited_domain') and, if exited,
    'exit_time'.
    """
    release_time = np.datetime64(release_time)
    t0_sec = float((release_time - field.times[0]) / np.timedelta64(1, "s"))
    duration_sec = duration_hours * 3600.0
    t_end_sec = t0_sec + duration_sec
    record_end = float(field.t_seconds[-1])
    truncated = t_end_sec > record_end
    t_end_sec = min(t_end_sec, record_end)

    def depth_of(elapsed_sec):
        return depth_fn(elapsed_sec) if depth_fn is not None else None

    d0 = depth_of(t0_sec)
    out_lon, out_lat, out_depth, out_times = [lon0], [lat0], [d0 or 0.0], [release_time]
    lon, lat, t = lon0, lat0, t0_sec
    next_output = t0_sec + output_interval_seconds
    status = "completed"
    exit_time = None

    while t < t_end_sec:
        dt = min(dt_seconds, t_end_sec - t)

        d_stage = depth_of(t - t0_sec)
        u1, v1 = field.velocity_at(lon, lat, t, depth=d_stage)
        if np.isnan(u1):
            status, exit_time = "exited_domain", field.times[0] + np.timedelta64(int(t), "s")
            break
        dlon1, dlat1 = _deg_per_sec(u1, v1, lat)

        d_stage = depth_of(t + 0.5 * dt - t0_sec)
        u2, v2 = field.velocity_at(lon + 0.5 * dt * dlon1, lat + 0.5 * dt * dlat1,
                                    t + 0.5 * dt, depth=d_stage)
        if np.isnan(u2):
            status, exit_time = "exited_domain", field.times[0] + np.timedelta64(int(t + 0.5 * dt), "s")
            break
        dlon2, dlat2 = _deg_per_sec(u2, v2, lat + 0.5 * dt * dlat1)

        u3, v3 = field.velocity_at(lon + 0.5 * dt * dlon2, lat + 0.5 * dt * dlat2,
                                    t + 0.5 * dt, depth=d_stage)
        if np.isnan(u3):
            status, exit_time = "exited_domain", field.times[0] + np.timedelta64(int(t + 0.5 * dt), "s")
            break
        dlon3, dlat3 = _deg_per_sec(u3, v3, lat + 0.5 * dt * dlat2)

        d_stage = depth_of(t + dt - t0_sec)
        u4, v4 = field.velocity_at(lon + dt * dlon3, lat + dt * dlat3, t + dt, depth=d_stage)
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
            out_depth.append(depth_of(t - t0_sec) or 0.0)
            out_times.append(field.times[0] + np.timedelta64(int(round(t)), "s"))
            next_output += output_interval_seconds

    if status == "completed" and truncated:
        status = "end_of_record"

    return {
        "times": np.array(out_times, dtype="datetime64[ns]"),
        "lon": np.array(out_lon),
        "lat": np.array(out_lat),
        "depth": np.array(out_depth),
        "status": status,
        "exit_time": exit_time,
    }


# ---------------------------------------------------------------------------
# 4. Output: CSV + plot
# ---------------------------------------------------------------------------

def save_trajectory_csv(traj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("time,lon,lat,depth_m\n")
        for t, lo, la, d in zip(traj["times"], traj["lon"], traj["lat"], traj["depth"]):
            f.write(f"{str(t)},{lo:.6f},{la:.6f},{d:.2f}\n")
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
    u0, v0 = field.plane_at(0, depth=trajectories[0]["depth"][0])
    ax.quiver(field.lon[nodes], field.lat[nodes], u0[nodes], v0[nodes],
              color="0.6", width=0.002, zorder=2,
              label=f"velocity @ first snapshot, {trajectories[0]['depth'][0]:.0f} m")
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

    varying_depth = any(np.ptp(t["depth"]) > 1e-6 for t in trajectories)
    if varying_depth:
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        for i, traj in enumerate(trajectories):
            color = cmap(i % 10)
            hours = (traj["times"] - traj["times"][0]) / np.timedelta64(1, "h")
            ax2.plot(hours, traj["depth"], "-o", color=color, ms=3,
                    label=f"particle {i}" if len(trajectories) > 1 else None)
        ax2.invert_yaxis()  # depth increases downward
        ax2.set_xlabel("hours since release")
        ax2.set_ylabel("depth (m)")
        ax2.set_title("Prescribed particle depth over time")
        ax2.grid(alpha=0.3)
        if len(trajectories) > 1:
            ax2.legend(loc="best", fontsize=8)
        plt.tight_layout()
        depth_path = out_path.replace(".png", "_depth.png")
        plt.savefig(depth_path, dpi=150)
        print(f"Saved depth-vs-time plot to {depth_path}")


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
                         "layer via z_levels if present, else 0). Ignored if "
                         "--depth-schedule is given.")
    p.add_argument("--depth-schedule", type=str, nargs="+", default=None,
                    help="Change the particle's depth over the run: one or "
                         "more HOURS:DEPTH_M tokens (elapsed hours since "
                         "release, depth in m positive-down), e.g. "
                         "--depth-schedule 0:0 6:20 18:50. Depth is held at "
                         "the first value before the first breakpoint and at "
                         "the last value after the last one. Same schedule "
                         "applies to every released particle.")
    p.add_argument("--depth-interp", choices=["step", "linear"], default="step",
                    help="'step' (default): depth jumps at each breakpoint and "
                         "is held constant in between. 'linear': ramps "
                         "smoothly between breakpoints.")
    p.add_argument("--depth-cycle-hours", type=float, default=None,
                    help="Repeat --depth-schedule with this period, e.g. 24 "
                         "for a daily diel-vertical-migration pattern.")
    p.add_argument("--no-bottom-mask", action="store_true",
                    help="Don't use the file's seabed 'depth' variable to mask "
                         "levels below the bottom at each node")
    p.add_argument("--out-plot", default="trajectory.png")
    p.add_argument("--out-csv", default="trajectory.csv")
    args = p.parse_args()

    n = len(args.lon)
    if not (len(args.lat) == n and len(args.time) == n):
        p.error("--lon, --lat, and --time must all have the same number of values")

    depth_schedule = None
    level_depths = bottom_depth = None
    if args.depth_schedule is not None:
        depth_schedule = DepthSchedule.from_cli_tokens(
            args.depth_schedule, mode=args.depth_interp,
            cycle_hours=args.depth_cycle_hours)
        max_depth = float(depth_schedule.depths.max())
        print(f"Depth schedule ({args.depth_interp}): "
              f"{list(zip(depth_schedule.hours.tolist(), depth_schedule.depths.tolist()))} "
              f"h:m" + (f", repeating every {args.depth_cycle_hours} h" if args.depth_cycle_hours else ""))

    if args.synthetic or not args.source:
        if not args.synthetic:
            print("No --source given; using synthetic demo current field.", file=sys.stderr)
        if depth_schedule is not None:
            lon, lat, triangles, times, u, v, level_depths, bottom_depth = \
                synthetic_unstructured_currents(vertical=True)
        else:
            lon, lat, triangles, times, u, v = synthetic_unstructured_currents()
    else:
        try:
            print(f"Loading unstructured currents from {args.source} ...")
            storage_options = {"anon": True} if args.zarr_anon else None
            rel = [np.datetime64(t) for t in args.time]
            win_start = min(rel)
            win_end = max(rel) + np.timedelta64(int(args.duration_hours * 3600), "s")
            if depth_schedule is not None:
                lon, lat, triangles, times, u, v, level_depths, bottom_depth = \
                    load_unstructured_currents(
                        args.source, engine=args.engine, storage_options=storage_options,
                        t_start=win_start, t_end=win_end,
                        depth_range=(0.0, max_depth), return_vertical=True,
                        use_bottom_depth=not args.no_bottom_mask)
            else:
                lon, lat, triangles, times, u, v = load_unstructured_currents(
                    args.source, layer_index=args.layer_index, engine=args.engine,
                    storage_options=storage_options, t_start=win_start, t_end=win_end)
        except Exception as e:
            print(f"Could not load {args.source} ({e}).\n"
                  f"Falling back to synthetic demo data so you can still "
                  f"test the tracker.", file=sys.stderr)
            if depth_schedule is not None:
                lon, lat, triangles, times, u, v, level_depths, bottom_depth = \
                    synthetic_unstructured_currents(vertical=True)
            else:
                lon, lat, triangles, times, u, v = synthetic_unstructured_currents()

    print(f"Mesh: {len(lon)} nodes, {len(triangles)} triangles, "
          f"{len(times)} time snapshots ({times[0]} to {times[-1]})")

    field = UnstructuredVelocityField(lon, lat, triangles, times, u, v,
                                      level_depths=level_depths, bottom_depth=bottom_depth)

    trajectories = []
    for lon0, lat0, t0 in zip(args.lon, args.lat, args.time):
        print(f"Tracking particle released at ({lon0}, {lat0}) at {t0} "
              f"for {args.duration_hours} h ...")
        depth_fn = depth_schedule.depth_at if depth_schedule is not None else None
        traj = track_particle(field, lon0, lat0, t0, args.duration_hours,
                               dt_seconds=args.dt_seconds,
                               output_interval_seconds=args.output_interval_hours * 3600,
                               depth_fn=depth_fn)
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