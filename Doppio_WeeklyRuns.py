# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 08:40:14 2026
doppio variation over time
Release a drifter at the same point every day (or week?) for a year or whatever and compare end points
@author: annis_rgr7tey
"""
import os
from datetime import timedelta, datetime
from opendrift.readers import reader_global_landmask
from opendrift.models.oceandrift import OceanDrift
from opendrift.readers.reader_netCDF_CF_generic import Reader
from opendrift.readers import reader_ROMS_native
import numpy as np
import xarray as xr
from scipy import spatial
from shapely.geometry import Point, LineString
from shapely import line_interpolate_point
from toolbox import haversine, bearing
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
from windrose import WindroseAxes
from scipy.spatial import KDTree

figPath='../doppio_WeeklyRun_figures'
opDir='../doppio_WeeklyRun2'
outPath='../\doppio_WeeklyRun2'
os.makedirs(outPath,exist_ok=True)
os.makedirs(opDir,exist_ok=True)
os.makedirs(figPath,exist_ok=True)

os.chdir(opDir)
#below is just a straight run
#need to add in changing depth, resetting to waypoints, transitting from shore
#and recording all the transits
latlim=[40,43]
lonlim=[-67,-74]

latA=(41+6/60)
lonA=-1*(71+15/60)


df=1 #drift factor

depth=[0,400] #in m
timeatdepth=[2,24] #in hours

z=0

NP=6
distThresh=10
transitspd=0.5
check_interval=3 #days in this case


Nruns=104 #run a week long for a year

runT=7
tds='https://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/bgc_oa/013/avg_1d'

n=0
newtime=datetime(2008,1,1,12)

while n<=Nruns:
    filename=outPath+'\\'+'Doppio_Week_' + str(n) + '.nc'
    
    # if n % 2: #if n is odd
    #     z=depth[1]
    #     runT=timeatdepth[1]
    # else: #n is even
    #     z=depth[0]
    #     runT=timeatdepth[0]

    o = OceanDrift(loglevel=0)
    o.set_config('environment:fallback:land_binary_mask', 0)
    reader_doppio = reader_ROMS_native.Reader(tds)
    o.add_reader(reader_doppio)    
    # o.seed_elements(lon=newlon, lat=newlat, number=1, time=newtime, current_drift_factor=df)
    # o.run(duration=timedelta(days=runT),outfile=filename)
    o.seed_elements(lon=lonA, lat=latA, number=1, time=newtime, current_drift_factor=df,z=z)
    o.run(time_step=timedelta(minutes=15),duration=timedelta(days=runT),outfile=filename)

    newtime=newtime+timedelta(days=7)
    n=n+1

lastlat=[]
lastlon=[]
all_lats=[]
all_lons=[]
dist=[]
bear=[]
for f in range(0,Nruns):
    ds=xr.open_dataset(outPath+'\\'+'Doppio_Week_' + str(f) + '.nc')
    lastvalid=np.where(ds['lon'][0].values==0)[0]
    if len(lastvalid)==0:
        lastvalid=-1
    else:
        lastvalid=lastvalid[0]
    all_lons=np.concatenate((all_lons,ds['lon'][0][0:lastvalid].values))
    all_lats=np.concatenate((all_lats,ds['lat'][0][0:lastvalid].values))
    
    lastlat.append(ds['lat'].values[0][lastvalid])
    lastlon.append(ds['lon'].values[0][lastvalid])
    
    dist.append(haversine(latA,lonA,lastlat[f],lastlon[f]))
    bear.append(bearing(latA,lonA,lastlat[f],lastlon[f]))
    
plt.figure()
m = Basemap(projection='merc', llcrnrlat=latlim[0], urcrnrlat=latlim[1], llcrnrlon=lonlim[1], urcrnrlon=lonlim[0], resolution='f')
m.drawcoastlines()

x,y=m(lastlon,lastlat)
xA,yA=m(lonA,latA)

m.plot(x,y,'.')
m.plot(xA,yA,'k*')

ax = WindroseAxes.from_ax()
ax.bar(np.array(bear), np.array(dist), normed=True, opening=0.8, edgecolor='white')
ax.set_legend()
ax.set_title('end point dist&bear')
plt.show()

ds_doppio=xr.open_dataset(tds)

dop_lat=ds_doppio.lat_u.values.flatten()
dop_lon=ds_doppio.lon_u.values.flatten()

dop_latv=ds_doppio.lat_v.values.flatten()
dop_lonv=ds_doppio.lon_v.values.flatten()

coords=[]
for c in range(0,len(dop_lat)):
    coords.append((dop_lat[c],dop_lon[c]))
    
coordsv=[]
for c in range(0,len(dop_latv)):
    coordsv.append((dop_latv[c],dop_lonv[c]))
    
    
tree = KDTree(coords)
query_point = [latA, lonA]

treev = KDTree(coordsv)

# Find nearest neighbor
distance, index = tree.query(query_point)
distance, indexv = treev.query(query_point)

doplatA=dop_lat[int(index)]
doplonA=dop_lon[int(index)]

doplatAv=dop_latv[int(indexv)]
doplonAv=dop_lonv[int(indexv)]

dopind=np.where((ds_doppio.lat_u==doplatA) & (ds_doppio.lon_u==doplonA))
dopindv=np.where((ds_doppio.lat_v==doplatAv) & (ds_doppio.lon_v==doplonAv))

ds_point=ds_doppio.isel(eta_u=[int(dopind[0][0])],xi_u=[int(dopind[1][0])],eta_v=[int(dopindv[0][0])],xi_v=[int(dopindv[1][0])]).squeeze()



# ds_point=ds_doppio.sel(lat_u=latA,lon_u=lonA,lat_v=latA,lon_v=lonA,method='nearest')

ds_point.dropna(dim='ocean_time')

spd=np.sqrt(ds_point.u[:,0]*ds_point.u[:,0]+ds_point.v[:,0]*ds_point.v[:,0])
drctn=np.degrees(np.arctan2(ds_point.v[:,0], ds_point.u[:,0])) %360

from metpy.calc import wind_direction
drctn_metpy=wind_direction(ds_point.u[:,0], ds_point.v[:,0], convention='to')


ax = WindroseAxes.from_ax()
ax.bar(np.array(drctn), np.array(spd), normed=True, opening=0.8, edgecolor='white')
ax.set_legend()
ax.set_title('start point')
plt.show()

ax = WindroseAxes.from_ax()
ax.bar(np.array(drctn_metpy), np.array(spd), normed=True, opening=0.8, edgecolor='white')
ax.set_legend()
ax.set_title('start point')
plt.show()
