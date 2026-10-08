# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 08:40:14 2026
doppio variation over time
Release a drifter at the same point every day (or week?) for a year or whatever and compare end points
@author: annis_rgr7tey
"""
# import os
# from datetime import timedelta, datetime
# from opendrift.models.oceandrift import OceanDrift
# from opendrift.readers import reader_ROMS_native
# import numpy as np
# import xarray as xr

# figPath=r'C:\Users\annis_rgr7tey\Documents\BIS_VS_domainRuns\doppio_WeeklyRun_figures'
# opDir=r'C:\Users\annis_rgr7tey\Documents\BIS_VS_domainRuns\doppio_WeeklyRun_Customdepths_2wks_2007-2009'
# outPath=r'C:\Users\annis_rgr7tey\Documents\BIS_VS_domainRuns\doppio_WeeklyRun_Customdepths_2wks_2007-2009'
# os.makedirs(outPath,exist_ok=True)
# os.makedirs(opDir,exist_ok=True)
# os.makedirs(figPath,exist_ok=True)

# os.chdir(opDir)
#below is just a straight run
#need to add in changing depth, resetting to waypoints, transitting from shore
#and recording all the transits
import os
import argparse

# outPath=os.getcwd()

# latlim=[40,43]
# lonlim=[-67,-74]

# ndep=3

# nby=25

# Nruns=104
# runT=7

# https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
def haversine(lat1, lon1, lat2, lon2):
      from math import radians, cos, sin, asin, sqrt, atan2, degrees
      R = 6372.8 # 3959.87433 is in miles.  For Earth radius in kilometers use 6372.8 km

      dLat = radians(lat2 - lat1)
      dLon = radians(lon2 - lon1)
      lat1 = radians(lat1)
      lat2 = radians(lat2)

      a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
      c = 2*asin(sqrt(a))

      # lat1 = radians(lat1)
      # lon1 = radians(lon1)
      # lat2 = radians(lat2)
      # lon2 = radians(lon2)
      # y = sin(lon2 - lon1) * cos(lat2)
      # x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2 - lon1)
      # intbearing = atan2(y, x)
      # intbearing = degrees(intbearing)
      # intbearing = (intbearing + 360) % 360
    
      return R * c

#https://www.askpython.com/python/examples/calculate-gps-distance-using-haversine-formula
def bearing(lati1, long1, lati2, long2):
    import math
    lati1 = math.radians(lati1)
    long1 = math.radians(long1)
    lati2 = math.radians(lati2)
    long2 = math.radians(long2)
    y = math.sin(long2 - long1) * math.cos(lati2)
    x = math.cos(lati1) * math.sin(lati2) - math.sin(lati1) * math.cos(
        lati2
    ) * math.cos(long2 - long1)
    intbearing = math.atan2(y, x)
    intbearing = math.degrees(intbearing)
    intbearing = (intbearing + 360) % 360
    return intbearing

def compass2uv(spd,drctn):
    import numpy as np
    spd=np.array(spd)
    drctn=np.array(drctn)
    u=spd*np.sin(np.radians(drctn))
    v=spd*np.cos(np.radians(drctn))
    return u,v

def main():
    import os
    from datetime import timedelta, datetime
    from opendrift.models.oceandrift import OceanDrift
    from opendrift.readers import reader_ROMS_native
    import numpy as np
    import xarray as xr
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--opDir", help="Operating directory, if different from current directory ",default=os.getcwd())
    p.add_argument("--latlim",help="list of latitude [lower, upper] limits for grid")
    p.add_argument("--lonlim",help="list of longitude [eastern, western] limits for grid")
    p.add_argument("--ndep",help="number of water column depths to run particles at",default=1)
    p.add_argument("--nby",help="value of edge nodes, n, for an n by n grid")
    p.add_argument("--Nruns",help="number of iterations to run")
    p.add_argument("--runT",help="run time of each iteration in DAYS")
    p.add_argument("--runname",help='run title under which opendrift will save the runs')
    p.add_argument("--bathymetry",help="location of GEBCO bathymetry file",default=r'C:\Users\annis_rgr7tey\Documents\GEBCO_Bathy\gebco_2025_n43.643_s40.551_w-71.668_e-68.629.nc')
    args = p.parse_args()
    
    tds='https://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/bgc_oa/013/avg_1d'
    ds_doppio=xr.open_dataset(tds)
    
    lats=np.linspace(args.latlim[0],args.latlim[1],args.nby)
    lons=np.linspace(args.lonlim[0],args.lonlim[1],args.nby)
    
    y,x=np.meshgrid(lats,lons)
    
    latA=y.flatten()
    lonA=x.flatten()
    
    lonA=np.append(lonA,-1*(69+15/60))
    latA=np.append(latA,41+45/60)
    
    # bathyfile=r'C:\Users\annis_rgr7tey\Documents\GEBCO_Bathy\gebco_2025_n43.643_s40.551_w-71.668_e-68.629.nc'
    bathyxr=xr.open_dataset(args.bathyfile)
                
    # vectorized nearest-neighbor lookup for all points at once
    lat_idx = xr.DataArray(latA, dims="points")
    lon_idx = xr.DataArray(lonA, dims="points")
    bathyvals = bathyxr.sel(lat=lat_idx, lon=lon_idx, method="nearest").elevation.values
    
    landmask = bathyvals >= 0
    landinds = np.where(landmask)[0]
    
    seavals = bathyvals[~landmask]
    z = np.stack([np.zeros_like(seavals), seavals / 2, seavals - 5], axis=1).ravel()
            
    latA=np.delete(latA,landinds)
    lonA=np.delete(lonA,landinds)
    
    df=[1]*len(latA)*args.ndep #drift factor
    
    # z=z*len(latA)
    # zcolor=['k','g','b']*len(latA)
    latA=np.repeat(latA,args.ndep)
    lonA=np.repeat(lonA,args.ndep)
    
    from pathlib import Path
    
    start_time = datetime(2007, 1, 1, 12)
    
    reader_doppio = reader_ROMS_native.Reader(tds)  # build once, reuse every run
    
    for n in range(args.Nruns + 1):
        print("running run " + str(n) +' of ' +str(args.Nruns))
        
        o = OceanDrift(loglevel=0)
        o.set_config('environment:fallback:land_binary_mask', 0)
        o.add_reader(reader_doppio)
    
        o.seed_elements(lon=lonA, lat=latA, time=start_time, current_drift_factor=df, z=z)
        o.run(time_step=timedelta(minutes=15), duration=timedelta(days=args.runT), outfile=args.runname+'_'+str(n))
    
        start_time += timedelta(days=args.runT)
    
    lastlat = []
    lastlon = []
    all_lats_list = []
    all_lons_list = []
    dist=[]
    bear=[]
    
    for f in range(0, args.Nruns):
        outfile=args.runname+'_'+str(n)
        ds = xr.open_dataset(args.outpath+'//'+outfile+'.nc')
        lon = ds['lon'].values   # shape (n_traj, n_time)
        lat = ds['lat'].values
        n_time = lon.shape[1]
    
        # first NaN index per trajectory (0 if none found -- fixed below)
        nan_mask = np.isnan(lon)
        has_nan = nan_mask.any(axis=1)
        first_nan = np.argmax(nan_mask, axis=1)
        lastvalid = np.where(has_nan, first_nan - 1, -1)   # same semantics as your lastvalid
    
        # last valid lon/lat per trajectory (vectorized fancy indexing)
        rows = np.arange(lon.shape[0])
        lastlat_f = lat[rows, lastvalid]
        lastlon_f = lon[rows, lastvalid]
        lastlat.append(lastlat_f)
        lastlon.append(lastlon_f)
    
        # flag any trajectory whose "last valid" point is itself NaN
        for t in np.where(np.isnan(lastlat_f))[0]:
            print(f'f is {f}')
            print(f't is {t}')
    
        # vectorized equivalent of ds['lon'][t][0:lastvalid[t]] for every t at once
        col_idx = np.arange(n_time)
        cutoff = lastvalid % n_time          # turns the -1 sentinel into n_time-1
        mask = col_idx[None, :] < cutoff[:, None]
        all_lons_list.append(lon[mask])
        all_lats_list.append(lat[mask])
        
        distint = haversine(latA, lonA, lastlat_f, lastlon_f)
        bearint = bearing(latA, lonA, lastlat_f, lastlon_f)
        dist.append(np.asarray(distint).tolist())
        bear.append(np.asarray(bearint).tolist())
    
        ds.close()
    
    all_lons = np.concatenate(all_lons_list)
    all_lats = np.concatenate(all_lats_list)
    
    distresh=np.array(dist).reshape(len(latA),args.Nruns)
    bearresh=np.array(bear).reshape(len(latA),args.Nruns)
    # distresh=np.array(dist).reshape(len(latA),20)
    # bearresh=np.array(bear).reshape(len(latA),20)

    u,v=compass2uv(dist,bear) 

    mean_u=np.nanmean(u,axis=0).flatten()
    mean_v=np.nanmean(v,axis=0).flatten()
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    
    zcolor=['k','g','b']*len(latA)

    #for the quiver plot need the distance and bearing at each location, so add that back in
    plt.figure()
    m = Basemap(projection='merc', llcrnrlat=args.latlim[0], urcrnrlat=args.latlim[1], llcrnrlon=args.lonlim[1], urcrnrlon=args.lonlim[0], resolution='f')
    m.drawcoastlines()
    m.fillcontinents('k')
    xmap,ymap=m(lonA,latA)
    plt.title('surface (black), mid depth(green), near bottom(blue)')
    m.quiver(xmap,ymap,mean_u,mean_v,color=zcolor)
    m.fillcontinents('k')

if __name__ == "__main__":
    main()
