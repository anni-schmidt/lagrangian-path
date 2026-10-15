# lagrangian-path
Doppio_weekly runs uses openDrift to calculate the trajectory of a single particle released at the same location weekly for n weeks. 

Doppio_weeklyRuns_grid_multiDepth uses openDrift to calculate the trajectories of single particles released at locations on an n by n grid, released weekly for n weeks, at the surface, mid depth, and near bottom. 

waypointTransit_tidal_inOrder_dijsktra uses parcels to calculate the trajectories of single particles that are following user defined waypoints. The path back to the waypoint is found using Dijkstra to choose the most efficient path. 

flowDirectionMapping is the file where various plots using doppio data to analyze current patterns are created

lagrangian_tracker, developed with Claude, command line run and uses the zarr directly as input
Example usage: python lagrangian_tracker.py --source C:/Users/annis_rgr7tey/Documents/TideTest/amr3d_shakedown_z_v2.zarr --lon -70.3 --lat 41.5 --time 2026-08-18T00:00:00 --duration-hours 48.0
Output is a csv with time, lat and lon, and a figure of the trajectory. 

lagrangian_tracker_dive is an expansion of lagrangian_tracker which includes changing depth cycles
example usage:
python lagrangian_tracker_dive.py --source C:/Users/annis_rgr7tey/Documents/TideTest/amr3d_shakedown_z_v2.zarr --lon -70.66265 --lat 41.506975 --time 2026-08-18T00:00:00 --duration-hours 72.0 --depth-schedule 0:0 6:20 18:50 --depth-cycle-hours 24


flow_plots and lyapunov use various ways of plotting flow patterns. Lyapunov uses the finite time lyapunov exponent, plots the map and highlights the attracting/repelling ridges. Flow_plots plots flow trajectories and streamlines in the area for the time specified. (known bugs, cannot edit number of points)

Example usage:
python flow_plots.py --source https://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/bgc_oa/013/avg_1d --bbox -74 -67 40 43 --time 2008-01-01T00:00 --duration-hours 72 --animate 

python lyapunov.py --source https://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/bgc_oa/013/avg_1d --bbox -74 -67 40 43 --time 2008-01-01T00:00 --integration-hours 72 [--depth 10] [--grid-type roms|hycom]
