# lagrangian-path
Doppio_weekly runs uses openDrift to calculate the trajectory of a single particle released at the same location weekly for n weeks. 

Doppio_weeklyRuns_grid_multiDepth uses openDrift to calculate the trajectories of single particles released at locations on an n by n grid, released weekly for n weeks, at the surface, mid depth, and near bottom. 

waypointTransit_tidal_inOrder_dijsktra uses parcels to calculate the trajectories of single particles that are following user defined waypoints. The path back to the waypoint is found using Dijkstra to choose the most efficient path. 

flowDirectionMapping is the file where various plots using doppio data to analyze current patterns are created

lagrangian_tracker, developed with Claude, command line run and uses the zarr directly as input
Example usage: python lagrangian_tracker.py --source C:/Users/annis_rgr7tey/Documents/TideTest/amr3d_shakedown_z_v2.zarr --lon -70.3 --lat 41.5 --time 2026-08-18T00:00:00 --duration-hours 48.0
Output is a csv with time, lat and lon, and a figure of the trajectory. 
