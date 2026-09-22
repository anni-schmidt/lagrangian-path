# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 13:58:39 2026
find a way to plot diverging/converging currents
integrate the direction over the water column to show net transport directions
streamlines at surface and bottom maybe?
LCS for layers to show diverging/converging areas
@author: annis_rgr7tey
"""
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap
from datetime import datetime, timedelta
import netCDF4
import scipy

figPath='../FlowDirectionFigures'

setlim=True
latlim=[40,43]
lonlim=[-67,-74]

st=datetime(2013,1,1)
nd=datetime(2013,2,1)
    
timelim=[st,nd]    

#start with one model, then expand I guess
#doppio
# tds='https://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/bgc_oa/013/avg_1d'
# ds_doppio=xr.open_dataset(tds)

#deal with a proper interp later, not working today
ds=xr.open_dataset('https://tds.marine.rutgers.edu/thredds/dodsC/roms/doppio/bgc_oa/013/avg_1d')

x=ds.lon_rho.values
y=ds.lat_rho.values

ds_sel=ds.sel(ocean_time=slice(timelim[0],timelim[1]))#,lat_u=y,lon_u=x,lat_v=y,lon_v=x,method='linear')
#interp u and v to the rho grid
# u=griddata((ds_sel['lon_u'].values, ds_sel['lat_u'].values), ds_sel['u'].squeeze()[0,0,:,:], (x, y), method='linear')

spd=[]
for n in range(0,len(ds.ocean_time)):
    spd.append(np.sqrt(ds.u[n,0,0:105,:].values*ds.u[n,0,0:105,:].values+ds.v[n,0,:,0:241].values*ds.v[n,0,:,0:241].values))
# spd =spd0
# YY, XX = np.mgrid[np.nanmin(y):np.nanmax(y):500j, np.nanmin(x):np.nanmax(x):500j]
w=1
# X,Y= np.mgrid[0:w:105j, 0:w:241j]

X,Y = np.mgrid[latlim[0]:latlim[1]:105j, lonlim[1]:lonlim[0]:241j]    

plt.figure()
plt.subplot(1, 2, 1)
plt.streamplot(Y,X,np.nanmean(ds.u[:,0,0:105,:],axis=0),np.nanmean(ds.v[:,0,:,0:241].values,axis=0),color='k',density=0.6,broken_streamlines=False)
plt.title('surface')
plt.subplot(1, 2, 2)
plt.streamplot(Y,X,np.nanmean(ds.u[:,-1,0:105,:],axis=0),np.nanmean(ds.v[:,-1,:,0:241].values,axis=0),color='k',density=0.6,broken_streamlines=False)
plt.title('bottom')

plt.figure()
c=plt.pcolormesh(x,y,spd[0])
plt.colorbar(c,label='current (m/s)')


spdvar=np.nanvar(spd,axis=0)
spdmean=np.nanmean(spd,axis=0)


plt.figure()
c=plt.pcolormesh(x,y,spdvar,cmap='Blues_r')
plt.colorbar(c,label='variance (m2/s2)')
if setlim==True:
    plt.xlim(lonlim[1],lonlim[0])
    plt.ylim(latlim)
    
plt.figure()
c=plt.pcolormesh(x,y,spdvar/spdmean,cmap='Blues_r')
plt.colorbar(c,label='normed variance (m/s)')
if setlim==True:
    plt.xlim(lonlim[1],lonlim[0])
    plt.ylim(latlim)

#plot vorticity?
meanvort=np.nanmean(ds['rvorticity'].values,axis=0)
plt.figure()
c=plt.pcolormesh(x,y,meanvort[0,:,:],cmap='hsv')
plt.title('surface mean vorticity')
    
drctn0=np.degrees(np.arctan2(ds_sel.v.values[:,0,:,0:241], ds_sel.u.values[:,0,0:105,:]))
drctn0[drctn0<0]=drctn0[drctn0<0]+360

plt.figure()
c=plt.pcolormesh(x,y,drctn0[0,:,:],cmap='hsv')
plt.colorbar(c,label='drctn (deg)')
if setlim==True:
    plt.xlim(lonlim[1],lonlim[0])
    plt.ylim(latlim)
    
drctn=np.degrees(np.arctan2(ds_sel.v.values[:,:,:,0:241], ds_sel.u.values[:,:,0:105,:]))
drctn[drctn<0]=drctn[drctn<0]+360

#...you know you cant do the mean of a straight direction, dummy
#do the u and v vector mean
drctn_rad=np.radians(drctn)
drctn_int=scipy.stats.circmean(scipy.stats.circmean(drctn_rad,axis=0),axis=0)

drctn_int_deg=np.degrees(drctn_int)

plt.figure()
c=plt.pcolormesh(x,y,drctn_int,cmap='hsv')
plt.colorbar(c,label='mean drctn (deg)')
if setlim==True:
    plt.xlim(lonlim[1],lonlim[0])
    plt.ylim(latlim)
    
plt.figure()
c=plt.pcolormesh(x,y,drctn_int_deg,cmap='hsv')
plt.colorbar(c,label='mean drctn (deg)')
if setlim==True:
    plt.xlim(lonlim[1],lonlim[0])
    plt.ylim(latlim)
    
#now do surface and depth seperately

drctn_rad0=np.radians(drctn[:,0,:,:])
drctn_int0=scipy.stats.circmean(drctn_rad0,axis=0)

drctn_int_deg0=np.degrees(drctn_int0)

plt.figure()
c=plt.pcolormesh(x,y,drctn_int0,cmap='hsv')
plt.colorbar(c,label='mean drctn (deg)')
plt.title('surface')
if setlim==True:
    plt.xlim(lonlim[1],lonlim[0])
    plt.ylim(latlim)

plt.figure()
c=plt.pcolormesh(x,y,drctn_int_deg0,cmap='hsv')
plt.colorbar(c,label='mean drctn (deg)')
plt.title('surface')
if setlim==True:
    plt.xlim(lonlim[1],lonlim[0])
    plt.ylim(latlim)
    
    
#now at depth
    
### get the bottom most layer for each grid cell?

drctn_rad1=np.radians(drctn[:,-1,:,:])
drctn_int1=scipy.stats.circmean(drctn_rad1,axis=0)

drctn_int_deg1=np.degrees(drctn_int1)

plt.figure()
c=plt.pcolormesh(x,y,drctn_int_deg1,cmap='hsv')
plt.colorbar(c,label='mean drctn (deg)')
plt.title('depth')
if setlim==True:
    plt.xlim(lonlim[1],lonlim[0])
    plt.ylim(latlim)
    
#difference between surface and depth?
plt.figure()
c=plt.pcolormesh(x,y,abs(drctn_int_deg0-drctn_int_deg1),cmap='hsv')
plt.colorbar(c,label='mean drctn (deg)')
plt.title('abs(surface-depth deg)')
if setlim==True:
    plt.xlim(lonlim[1],lonlim[0])
    plt.ylim(latlim)
    
#maybe binary if theyre are roughly the same, or not
#diff color map?
from matplotlib.colors import LinearSegmentedColormap
colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # R -> G -> B
# n_bins = [3, 6, 10, 100]  # Discretizes the interpolation into bins
cmap_name = 'my_list'
cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=3)

plt.figure()
c=plt.pcolormesh(x,y,abs(drctn_int_deg0-drctn_int_deg1),cmap=cmap)
plt.colorbar(c,label='diff (deg)')
plt.title('abs(surface-depth deg)')

## calculate (vector) transport across each cell and map that


# map shear, du/dz
spd_all=[]
for d in range(0,len(ds.eta_u)):
    spd_dep=[]
    for n in range(0,len(ds.ocean_time)):
        spd_dep.append(np.sqrt(ds.u[n,d,0:105,:].values*ds.u[n,d,0:105,:].values+ds.v[n,d,:,0:241].values*ds.v[n,d,:,0:241].values))
    spd_all.append(spd_dep)
    
spd_grad=np.max(np.gradient(np.array(spd_all),axis=0),axis=0)
plt.figure()
c=plt.pcolormesh(x,y,np.mean(spd_grad,axis=0))
plt.colorbar(c)

#plot the depth of the max gradient, that'll be the pycocline?

grad_dep=np.gradient(np.array(spd_all),axis=0).argmax(axis=0)

plt.figure()
c=plt.pcolormesh(x,y,np.mean(grad_dep,axis=0))
plt.colorbar(c,label='index')
plt.title('index of max spd gradient in depth')

## do a transport across transect plot
#calc transport equation


## sea surface height, which is related to convergence/divergence
import copernicusmarine


dssat = copernicusmarine.open_dataset(dataset_id='c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D')
# data=xr.open_dataset(r'C:\Users\annis_rgr7tey\Documents\NOAA\c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0p25deg_P1D_CUT2020Jan18.nc')
# adt=data['adt']
adt=dssat['adt'].sel(latitude=slice(latlim[0],latlim[1]),longitude=slice(lonlim[1],lonlim[0])).sel(time=datetime(2007,1,1,12),method='nearest')
# adt=dssat['adt'].sel(latitude=slice(32,46),longitude=slice(-80,-60)).sel(time=datetime(2007,1,1,12),method='nearest')

adtgrad=np.sqrt(np.gradient(adt)[0]*np.gradient(adt)[0]+np.gradient(adt)[1]*np.gradient(adt)[1])
XX,YY=np.meshgrid(adt['longitude'],adt['latitude'])

#plot to make sure there is data!
plt.figure()
c=plt.pcolormesh(XX,YY,adtgrad)
plt.colorbar(c)
plt.title('ADT gradient')

plt.figure()
c=plt.pcolormesh(XX,YY,adt)
plt.colorbar(c)
plt.streamplot(Y,X,ds.u[0,0,0:105,:],ds.v[0,0,:,0:241].values,color='k',broken_streamlines=False)
plt.title('ADT')

