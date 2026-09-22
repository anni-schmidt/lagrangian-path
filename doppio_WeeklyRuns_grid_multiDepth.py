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
os.chdir(r'C:\Users\annis_rgr7tey\OneDrive\Documents')
from toolbox import haversine, bearing #, compass2uv
def compass2uv(spd,drctn):
    import numpy as np
    spd=np.array(spd)
    drctn=np.array(drctn)
    u=spd*np.sin(np.radians(drctn))
    v=spd*np.cos(np.radians(drctn))
    return u,v
os.chdir(r'C:\Users\annis_rgr7tey')
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
from windrose import WindroseAxes
from scipy.spatial import KDTree

figPath=r'C:\Users\annis_rgr7tey\Documents\BIS_VS_domainRuns\doppio_WeeklyRun_figures'
opDir=r'C:\Users\annis_rgr7tey\Documents\BIS_VS_domainRuns\doppio_WeeklyRun_Customdepths_2wks_2007-2009'
outPath=r'C:\Users\annis_rgr7tey\Documents\BIS_VS_domainRuns\doppio_WeeklyRun_Customdepths_2wks_2007-2009'
os.makedirs(outPath,exist_ok=True)
os.makedirs(opDir,exist_ok=True)
os.makedirs(figPath,exist_ok=True)

os.chdir(opDir)
#below is just a straight run
#need to add in changing depth, resetting to waypoints, transitting from shore
#and recording all the transits
latlim=[40,43]
lonlim=[-67,-74]

#correct this to get at depth, mid depth, and surface for all pts 
# z=[0,-10,-500]

# ds_hycom=xr.open_dataset('http://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z',decode_times=False)

# #select latA/latB
# #plot some depth plots
# #maybe a 2D in time
# #see why there isnt variation

#check the landmask points?
# ds_hycom=xr.open_dataset('http://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z',decode_times=False)
#doppio
tds='https://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/bgc_oa/013/avg_1d'
ds_doppio=xr.open_dataset(tds)
# plt.figure()
# plt.pcolormesh(ds_hycom.lon,ds_hycom.lat,ds_hycom.water_u[0,0])
# plt.xlim(360-74,360-67)
# plt.ylim(40,43)
# latA=(41+6/60)
# lonA=-1*(71+15/60)

#make lat A and lon A a grid that covers the domain
#maybe, what, 100 by 100?
lats=np.linspace(latlim[0],latlim[1],25)
lons=np.linspace(lonlim[0],lonlim[1],25)

y,x=np.meshgrid(lats,lons)

latA=y.flatten()
lonA=x.flatten()

lonA=np.append(lonA,-1*(69+15/60))
latA=np.append(latA,41+45/60)

#pop all the points that are on land somehow
# ds_hycom_sel=ds_hycom.sel(lat=lats,lon=360+lons,depth=0,method='nearest').isel(time=0)
# ds_sel=ds_doppio.sel(lat)
mask=np.where(np.isnan(ds_doppio.alkalinity[0,0,:,:].values.flatten()))

# nanindlat=np.where(np.isnan(ds_hycom_sel.water_u.values))[0]
# nanindlon=np.where(np.isnan(ds_hycom_sel.water_u.values))[1]

# landlat=lats[nanindlat]
# landlon=lons[nanindlon]
# landlat=ds_doppio.lat_u[mask]
# landlon=ds_doppio.lat_u[mask]

# landinds=[]
# for l in range(0,len(landlat)):
#     try:
#         ind=np.where((latA==landlat[l]) & (lonA==landlon[l]))
#         landinds.append(ind[0][0])
#     except:
#         print('didnt work')
        
# latA=np.delete(latA,landinds)
# lonA=np.delete(lonA,landinds)

# z=[]
# for i in range(0,len(latA)):
#     ds_hycom_sel=ds_hycom.sel(lat=latA[i],lon=360+lonA[i],method='nearest')

#     depi=np.where(np.isnan(ds_hycom_sel.water_u[0,:].values))[0][0]-1
#     sitedep=ds_hycom_sel.depth[depi].values

#     z.extend([0,-1*int(sitedep/2),-1*int(sitedep)])
#get the doppio land mask and depths
#do it looping over each point, opening, if on land pop it, if not, add depth to list
#or just get the GEBCO depth and use a percentage??
bathyfile=r'C:\Users\annis_rgr7tey\Documents\GEBCO_Bathy\gebco_2025_n43.643_s40.551_w-71.668_e-68.629.nc'
bathyxr=xr.open_dataset(bathyfile)
z=[]
landinds=[]
for i in range(0,len(latA)):
    bathyval=bathyxr.sel(lat=latA[i],lon=lonA[i],method='nearest').elevation.values
    if bathyval>=0:
        landinds.append(i)
    else:
        z.extend([0,bathyval/2,bathyval-5])
        
latA=np.delete(latA,landinds)
lonA=np.delete(lonA,landinds)

ndep=3

df=[1]*len(latA)*ndep #drift factor

# depth=[0,400] #in m
# timeatdepth=[2,24] #in hours

# z=z*len(latA)
zcolor=['k','g','b']*len(latA)
latA=np.repeat(latA,ndep)
lonA=np.repeat(lonA,ndep)

NP=6
distThresh=10
transitspd=0.5
check_interval=3 #days in this case


# Nruns=52 #run a week long for a year
Nruns=104
runT=7

# o = OceanDrift(loglevel=0)
# o.set_config('environment:fallback:land_binary_mask', 0)
# # reader_doppio = reader_ROMS_native.Reader(tds)
# # o.add_reader(reader_doppio)    
# o.add_readers_from_list(['http://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z'])
# for n in range(0,len(latA)):
#     is_land = o.land_binary_mask(lonA[n], latA[n])
#     if ~is_land:
        
#try to get the ftle
#foreward is stable and backward unstable? unstable/attracting, stable/repelling?
o=OceanDrift(loglevel=0)
o.set_config('drift:advection_scheme','runge-kutta4')
reader_doppio = reader_ROMS_native.Reader(tds)
o.add_reader(reader_doppio) 
times=[]
for d in range(1,28):
    times.append(datetime(2007,1,d,12))
# times=[datetime(2007,1,1,12), datetime(2007,1,2,12),datetime(2007,1,3,12)]
lcs=o.calculate_ftle(time=times,time_step=timedelta(days=1),duration=timedelta(days=27),delta=0.1,domain=np.hstack([lonlim[::-1],latlim]).astype('float'))


newtime=[datetime(2007,1,1,12)]*len(latA)

o = o.clone()
o.seed_elements(lon=lonA[700], lat=latA[700], number=10, radius=10,time=newtime[0])
o.run(time_step=timedelta(minutes=5),duration=timedelta(days=5),outfile='lcsrunlong2')
o.plot(lcs=lcs, colorbar=True, vmin=1e-7, vmax=1e-5,show_elements=True,buffer=2)

#backward
ob=OceanDrift(loglevel=0)
ob.set_config('drift:advection_scheme','runge-kutta4')
reader_doppio = reader_ROMS_native.Reader(tds)
ob.add_reader(reader_doppio) 
times=[]
for d in range(1,28):
    times.append(datetime(2007,1,d,12))
times=times[::-1]
# times=[datetime(2007,1,1,12), datetime(2007,1,2,12),datetime(2007,1,3,12)][::-1]
lcsback=ob.calculate_ftle(time=times,time_step=timedelta(days=-1),duration=timedelta(days=27),delta=0.1,domain=np.hstack([lonlim[::-1],latlim]).astype('float'),RLCS=False)

ob = ob.clone()
ob.seed_elements(lon=lonA[700], lat=latA[700], number=10, radius=10,time=newtime[0])
ob.run(time_step=timedelta(minutes=5),duration=timedelta(days=5),outfile='lcsrunbacklong2')
ob.plot(lcs=lcsback, colorbar=True, vmin=1e-7, vmax=1e-5,show_elements=True,buffer=2)

# lcssum=lcs['ALCS']+lcsback['ALCS']
lcssum=lcs['ALCS']+lcs['RLCS']

# ob = ob.clone()
# ob.seed_elements(lon=lonA[700], lat=latA[700], number=10, radius=10,time=newtime[0])
# ob.run(time_step=timedelta(minutes=5),duration=timedelta(days=5),outfile='lcsrunsumlong')
# ob.plot(lcs=lcssum, colorbar=True, vmin=1e-7, vmax=1e-5,show_elements=True,buffer=2)

plt.figure()
plt.contourf(np.nanmean(lcssum,axis=0))

#that's showing all nans but maybe that's just because i have no wifi, try again later.

# tds='https://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/bgc_oa/013/avg_1d'
n=0
newtime=[datetime(2007,1,1,12)]*len(latA)
# newtime=[datetime(2020,1,1,12)]
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
    # o.add_readers_from_list(['http://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z'])
    # o.set_config('general:coastline_action', 'stranding')
    # o.set_config('general:use_auto_landmask', True)
    # o.seed_elements(lon=newlon, lat=newlat, number=1, time=newtime, current_drift_factor=df)
    # o.run(duration=timedelta(days=runT),outfile=filename)
    o.seed_elements(lon=lonA, lat=latA, time=newtime, current_drift_factor=df,z=z)
    o.run(time_step=timedelta(minutes=15),duration=timedelta(days=runT),outfile=filename)

    newtime=[newtime[0]+timedelta(days=runT)]*len(latA)
    n=n+1

lastlat=[]
lastlon=[]
all_lats=[]
all_lons=[]
dist=[]
bear=[]
# for f in range(0,Nruns):
for f in range(0,20):

    ds=xr.open_dataset(outPath+'\\'+'Doppio_Week_' + str(f) + '.nc')
    # lastvalid=np.where(ds['lon'][0].values==0)[0]
    # print(f)
    # if len(lastvalid)==0:
    #     lastvalid=-1
    # else:
    #     lastvalid=lastvalid[0]
    # print(ds.lon[:][lastvalid].values)

    distint=[]
    bearint=[]
    for t in range(0,len(latA)):
        lastvalid=np.where(np.isnan(ds['lon'][t].values))[0]
        if len(lastvalid)==0:
            lastvalid=-1
        else:
            lastvalid=lastvalid[0]-1
        
        all_lons=np.concatenate((all_lons,ds['lon'][t][0:lastvalid].values))
        all_lats=np.concatenate((all_lats,ds['lat'][t][0:lastvalid].values))
        
        lastlat.append(ds['lat'].values[t][lastvalid])
        lastlon.append(ds['lon'].values[t][lastvalid])
        
        if np.isnan(ds['lat'].values[t][lastvalid]):
            print('f is ' +str(f))
            print('t is '+str(t))
            
        di=haversine(latA[t],lonA[t],ds['lat'].values[t][lastvalid],ds['lon'].values[t][lastvalid])
        bi=bearing(latA[t],lonA[t],ds['lat'].values[t][lastvalid],ds['lon'].values[t][lastvalid])
        distint.append(di)
        bearint.append(bi)
    dist.append(distint)
    bear.append(bearint)
        
        
    # ax = WindroseAxes.from_ax()
    # ax.bar(np.array(bearint), np.array(distint), normed=True, opening=0.8, edgecolor='white')
    # ax.set_legend()
    # ax.set_title('end point dist&bear' + str(latA[t]) + ' ' + str(lonA[t]))
    # plt.show()
    
distresh=np.array(dist).reshape(len(latA),Nruns)
bearresh=np.array(bear).reshape(len(latA),Nruns)
# distresh=np.array(dist).reshape(len(latA),20)
# bearresh=np.array(bear).reshape(len(latA),20)

u,v=compass2uv(dist,bear) 

mean_u=np.nanmean(u,axis=0).flatten()
mean_v=np.nanmean(v,axis=0).flatten()
# mean_u=np.nanmean(u.reshape(len(latA),Nruns),axis=1)
# mean_v=np.nanmean(v.reshape(len(latA),Nruns),axis=1)

# mean_u=np.nanmedian(u.reshape(len(latA),Nruns),axis=1)
# mean_v=np.nanmedian(v.reshape(len(latA),Nruns),axis=1)

#doppio coords
latslcs=np.linspace(latlim[0],latlim[1],30)
lonslcs=np.linspace(lonlim[0],lonlim[1],70)
xx,yy=np.meshgrid(lonslcs,latslcs)
#quiver plot
plt.figure()
m = Basemap(projection='merc', llcrnrlat=latlim[0], urcrnrlat=latlim[1], llcrnrlon=lonlim[1], urcrnrlon=lonlim[0], resolution='f')
m.drawcoastlines()
m.fillcontinents('k')
xmap,ymap=m(lonA,latA)
xlcsmap,ylcsmap=m(xx,yy) #where do i get the lat lons for doppio?
plt.title('surface (black), mid depth(green), near bottom(blue)')
# c=m.contourf(xlcsmap,ylcsmap,np.nanmean(lcs['RLCS'],axis=0),cmap='gray_r')
c=m.contourf(xlcsmap,ylcsmap,np.nanmean(lcssum,axis=0),cmap='gray_r')

m.quiver(xmap,ymap,mean_u,mean_v,color=zcolor)
m.fillcontinents('k')

plt.colorbar(c)


#a streamplot of the above mean vectors would be similar to FTLE? it would show the areas of steady flow and the gaps
#maybe did a stupid thing with the depths and data structuring
#pull out just the surface data and try that, every 3rd point?
#then there's the increasing problem with the indices..imaginary number used in example?
plt.figure()
plt.streamplot(latA,lonA,mean_u,mean_v)


plt.figure()
m = Basemap(projection='merc', llcrnrlat=latlim[0], urcrnrlat=latlim[1], llcrnrlon=lonlim[1], urcrnrlon=lonlim[0], resolution='f')
m.drawcoastlines()

x,y=m(lastlon,lastlat)
xA,yA=m(lonA,latA)

m.plot(x,y,'.')
m.plot(xA,yA,'k*')


#compare to current roses from same pts same time
sti=np.where(ds_doppio.ocean_time==np.datetime64(datetime(2007,1,1,12)))[0][0]
ndi=np.where(ds_doppio.ocean_time==np.datetime64(newtime[0]))[0][0]
# ndi=-1
ds_doppio_sel=ds_doppio.isel(ocean_time=slice(int(sti),int(ndi)),s_rho=39,eta_u=slice(0,105),xi_v=slice(0,241))
ds_doppio_sel_dep=ds_doppio.isel(ocean_time=slice(int(sti),int(ndi)),s_rho=5,eta_u=slice(0,105),xi_v=slice(0,241))

mean_u_model=np.nanmean(ds_doppio_sel.u,axis=0)
mean_v_model=np.nanmean(ds_doppio_sel.v,axis=0)

mean_u_model_dep=np.array(np.nanmean(ds_doppio_sel_dep.u,axis=0)).flatten()
mean_v_model_dep=np.array(np.nanmean(ds_doppio_sel_dep.v,axis=0)).flatten()




plt.figure()
m = Basemap(projection='merc', llcrnrlat=latlim[0], urcrnrlat=latlim[1], llcrnrlon=lonlim[1], urcrnrlon=lonlim[0], resolution='f')
m.drawcoastlines()
x_model_map,y_model_map=m(ds_doppio_sel.lon_u,ds_doppio_sel.lat_u)
m.quiver(x_model_map,y_model_map,mean_u_model,mean_v_model,scale=2)#,color=zcolor)
plt.title('surface (black)') #, mid depth(green), near bottom(blue)
#make the vectors every like 5 or whatever, less dense

#get a map of like, where arrows are rougly perp? cross prod?
#so index 0 crossed with index 2, 3 with 5 etc
surfind=np.arange(0,len(mean_u),3)
depind=np.arange(2,len(mean_u),3)

cross=[]
for n in range(0,len(surfind)):
    cross.append(np.dot([mean_u[surfind[n]],mean_v[surfind[n]]],[mean_u[depind[n]],mean_v[depind[n]]]))
cross=np.array(cross)

from matplotlib.colors import ListedColormap
cmap = ListedColormap(["green", "red"])
 
plt.figure()
m = Basemap(projection='merc', llcrnrlat=latlim[0], urcrnrlat=latlim[1], llcrnrlon=lonlim[1], urcrnrlon=lonlim[0], resolution='f')
m.drawcoastlines()
c=m.scatter(xmap[surfind],ymap[surfind],c=cross/abs(cross),cmap=cmap)
plt.colorbar(c,label='cos theta')


mean_u_model=np.array(mean_u_model).flatten()
mean_v_model=np.array(mean_v_model).flatten()

modelcross=[]
for n in range(0,len(mean_u_model)):
    modelcross.append(np.dot([mean_u_model[n],mean_v_model[n]],[mean_u_model_dep[n],mean_v_model_dep[n]]))
modelcross=np.array(modelcross)

plt.figure()
m = Basemap(projection='merc', llcrnrlat=latlim[0], urcrnrlat=latlim[1], llcrnrlon=lonlim[1], urcrnrlon=lonlim[0], resolution='f')
m.drawcoastlines()
c=m.scatter(np.array(x_model_map).flatten(),np.array(y_model_map).flatten(),c=modelcross/abs(modelcross),cmap=cmap)
plt.colorbar(c,label='cos theta')


# ax = WindroseAxes.from_ax()
# ax.bar(np.array(bear), np.array(dist), normed=True, opening=0.8, edgecolor='white')
# ax.set_legend()
# ax.set_title('end point dist&bear')
# plt.show()

# ds_hycom=xr.open_dataset('http://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z',decode_times=False)

# # dop_lat=ds_doppio.lat_u.values.flatten()
# # dop_lon=ds_doppio.lon_u.values.flatten()

# # dop_latv=ds_doppio.lat_v.values.flatten()
# # dop_lonv=ds_doppio.lon_v.values.flatten()

# # coords=[]
# # for c in range(0,len(dop_lat)):
# #     coords.append((dop_lat[c],dop_lon[c]))
    
# # coordsv=[]
# # for c in range(0,len(dop_latv)):
# #     coordsv.append((dop_latv[c],dop_lonv[c]))
    
    
# # tree = KDTree(coords)
# # query_point = [latA, lonA]

# # treev = KDTree(coordsv)

# # # Find nearest neighbor
# # distance, index = tree.query(query_point)
# # distance, indexv = treev.query(query_point)

# # doplatA=dop_lat[int(index)]
# # doplonA=dop_lon[int(index)]

# # doplatAv=dop_latv[int(indexv)]
# # doplonAv=dop_lonv[int(indexv)]

# # dopind=np.where((ds_doppio.lat_u==doplatA) & (ds_doppio.lon_u==doplonA))
# # dopindv=np.where((ds_doppio.lat_v==doplatAv) & (ds_doppio.lon_v==doplonAv))

# ds_point=ds_hycom.sel(lat=latA,lon=lonA,method='nearest')



# # ds_point=ds_doppio.sel(lat_u=latA,lon_u=lonA,lat_v=latA,lon_v=lonA,method='nearest')

# ds_point.dropna(dim='time')

# spd=np.sqrt(ds_point.water_u[:,0]*ds_point.water_u[:,0]+ds_point.water_v[:,0]*ds_point.water_v[:,0])
# drctn=np.degrees(np.arctan2(ds_point.water_v[:,0], ds_point.water_u[:,0])) %360

# from metpy.calc import wind_direction
# drctn_metpy=wind_direction(ds_point.water_u[:,0], ds_point.water_v[:,0], convention='to')


# ax = WindroseAxes.from_ax()
# ax.bar(np.array(drctn), np.array(spd), normed=True, opening=0.8, edgecolor='white')
# ax.set_legend()
# ax.set_title('start point')
# plt.show()

# ax = WindroseAxes.from_ax()
# ax.bar(np.array(drctn_metpy), np.array(spd), normed=True, opening=0.8, edgecolor='white')
# ax.set_legend()
# ax.set_title('start point')
# plt.show()