# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 09:51:42 2026
slightly offshore run for animations, with transits back to waypoints
@author: annis_rgr7tey
"""
# [import packages]
import os
import math
import datetime
from datetime import timedelta
import xarray as xr
import parcels
import pandas as pd
import numpy as np
os.chdir(r'C:\Users\annis_rgr7tey\OneDrive\Documents')
from toolbox import haversine, bearing
os.chdir(r'C:\Users\annis_rgr7tey\Documents\NECOFS_Parcels')
from NECOFS_Parcels_Functions import CheckOutOfBounds, CheckError, flag_land, remove_stranded, smagdiff, record_depth, record_amb_vel, set_displacement, displace,  makeFieldsetFVCOMMove
from getPath import DijkstraPath
import matplotlib.pyplot as plt
from geopy.distance import geodesic
from geopy.point import Point
from scipy.spatial import KDTree
from scipy.signal import find_peaks
from shapely.geometry import Point, LineString
from shapely import line_interpolate_point
from windrose import WindroseAxes
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import datetime
from datetime import timedelta

notes='return to release point from the final point of free floating run, check location every other tidal cycle'

dirPath=r'C:\Users\annis_rgr7tey\Documents\NECOFS_StellwagenBankRun\test1_gloucester_transit_hindcast1978'
os.makedirs(dirPath,exist_ok=True)
os.chdir(dirPath)

# cycles=50
depthlayer=8
surfacelayer=4
distThresh=10 #threshold in km of how far when reset, this is checked later depending on dist b/t waypoints
transitspd=0.5 #m/s
NP=3 #number of waypoints
setFree=False
driftfactor=1
model='NECOFS'

#add a line to set check frequency, and then adjust the loop accordingly
#maybe have a little true/false list so itll check every N tidal cycles
CF=2 #check frequency

# ptA
# latA=42+12/60+54/3600
# lonA=-1*(70+42/60+18/3600) 
# latA=42.25416666666667
# lonA=-70.6125
latA=42+38/60
lonA=-1*(70+32/60)
# latA=42.05144500732422
# lonA=-70.06498718261719
# ptB
latB=42+18/60
lonB=-1*(70+15/60)
# latB=42+38/60
# lonB=-1*(70+32/60)

# ds=xr.load_dataset('C:\\Users\\annis_rgr7tey\\Documents\\NECOFS\\concat\\Concat_avg_forecast_2025-11-07T00-00-00.000000000_to_2025-11-20T00-00-00.000000000.nc')

ds=xr.load_dataset('C:\\Users\\annis_rgr7tey\\Documents\\NECOFS\\concat\\Concat_avg_forecast_1978-01-01T00-00-00.000000000_to_1978-02-03T04-00-00.000000000.nc')

latlim=[41.5,43]
lonlim=[-69.5,-71]

Y=1978
M=1
D=1
# Y=2025
# M=11
# D=7
H=0

timeA=datetime.datetime(Y,M,D,H)
timestr=timeA.strftime("%Y-%m-%d %H:%M:%S")

file_obj = open("output.txt", "w", encoding="utf-8")

lines=['Release Lat \t %0.2f \n'%latA,
       'Release lon \t %0.2f \n'%lonA,
       'Goal Lat \t %0.2f \n'%latB,
       'Goal lon \t %0.2f \n'%lonB,
       'depth drift layer \t %int \n'%depthlayer,
       'surface drift layer \t %i \n'%surfacelayer, 
       'drift factor \t %0.2f \n'%driftfactor,
       'Distance threshold \t %i km \n'%distThresh,
       'Transit speed \t %0.1f m/s \n'%transitspd,
       'Number of waypoints \t %i \n'%NP,
       'Release time \t %i-%i-%i %i:00:00 \n'%(Y,M,D,H),
       'Location checked every %int tidal cycles \n' %CF,
       'Model \t %s \n'%model,
       notes
       
       ]

file_obj.writelines(lines)

file_obj.close()

#######################
##### tidal times #####
#######################

coords=[]
for c in range(0,len(ds.nele)):
    coords.append((ds.nele.latc.values[c],ds.nele.lonc.values[c]))
    
nodes=[]
for c in range(0,len(ds.node)):
    nodes.append((ds.node.lat.values[c],ds.node.lon.values[c]))
    
tree = KDTree(coords)
query_point = [latA, lonA]

# Find nearest neighbor
distance, index = tree.query(query_point)

    
nodetree = KDTree(nodes)
# Find nearest neighbor
distancenode, indexnode = nodetree.query(query_point)

ds_point=ds.isel(nele=index,node=indexnode)


ds_point.dropna(dim='time')

spd=np.sqrt(ds_point.u[:,0]*ds_point.u[:,0]+ds_point.v[:,0]*ds_point.v[:,0])
drctn=90-np.degrees(np.arctan2(ds_point.v[:,0], ds_point.u[:,0]))

# spd=np.sqrt(ds_point.u[:,4]*ds_point.u[:,4]+ds_point.v[:,4]*ds_point.v[:,4])
# drctn=90-np.degrees(np.arctan2(ds_point.v[:,4], ds_point.u[:,4]))

drctn[drctn<0]=drctn[drctn<0]+360

drctngrad=np.gradient(drctn)

# plt.figure()
# plt.plot(drctngrad)
# plt.plot(drctn)

peaks=find_peaks(drctngrad)[0]
switch_time=ds.time[peaks]

from scipy.signal import argrelextrema

mins=argrelextrema(np.array(spd), np.less)
switch_time_spd=ds.time[mins]

fig,ax1=plt.subplots()
ax1.plot(ds.time.values,drctn)
ax2=ax1.twinx()
ax2.plot(ds.time.values,spd,c='orange')
for n in range(0,len(switch_time)):
    ax2.plot([switch_time.values[n],switch_time.values[n]],[0,0.7],'k')
# ax2.plot([switch_time.values[1],switch_time.values[1]],[0,0.6],'k')
# ax2.plot([switch_time.values[2],switch_time.values[2]],[0,0.6],'k')
# ax2.plot([switch_time.values[3],switch_time.values[3]],[0,0.6],'k')


fig,ax1=plt.subplots(figsize=[12,6])
ax1.plot(ds.time.values,ds_point.h.values+ds_point.zeta,label='wlev')
ax1.set_ylim([35,40])
ax1.set_ylabel('water level above bottom (m)')
ax2=ax1.twinx()
ax2.plot(ds.time.values,spd,c='orange',label='speed')
ax2.set_ylabel('current speed (m/s)')
fig.legend()
for n in range(0,len(switch_time_spd)):
    ax2.plot([switch_time_spd.values[n],switch_time_spd.values[n]],[0,0.7],'--k')
    
fig,ax1=plt.subplots(figsize=[12,6])
ax1.plot(ds.time.values,ds_point.h.values+ds_point.zeta,label='wlev')
ax1.set_ylim([35,40])
ax1.set_ylabel('water level above bottom (m)')
ax2=ax1.twinx()
ax2.plot(ds.time.values,drctn,c='orange',label='dir')
ax2.set_ylabel('current direction (deg)')
fig.legend()
for n in range(0,len(switch_time)):
    ax2.plot([switch_time.values[n],switch_time.values[n]],[0,360],'--k')

ax = WindroseAxes.from_ax()
ax.bar(np.array(drctn), np.array(spd), normed=True, opening=0.8, edgecolor='white')
ax.set_legend()
ax.set_title('start point')
plt.show()

   
fig,ax=plt.subplots(nrows=3,sharex=True,figsize=(12,8))
ax[0].plot(ds.time.values,ds_point.h.values+ds_point.zeta,label='wlev')
ax[0].set_title('water level')
# ax[0].set_ylim([27.25,29])
ax[0].grid()
ax[1].plot(ds.time.values,spd,c='k',label='speed')
ax[1].set_title('speed')
ax[1].grid()
ax[2].plot(ds.time.values,drctn,c='orange',label='dir')
ax[2].set_title('direction')
ax[2].set_ylim([0,360])
ax[2].grid()

query_point = [latB, lonB]

# Find nearest neighbor
distance, index = tree.query(query_point)

    
nodetree = KDTree(nodes)
# Find nearest neighbor
distancenode, indexnode = nodetree.query(query_point)

ds_point=ds.isel(nele=index,node=indexnode)


ds_point.dropna(dim='time')
spd=np.sqrt(ds_point.u[:,0]*ds_point.u[:,0]+ds_point.v[:,0]*ds_point.v[:,0])
drctn=90-np.degrees(np.arctan2(ds_point.v[:,0], ds_point.u[:,0]))
drctn[drctn<0]=drctn[drctn<0]+360

ax = WindroseAxes.from_ax()
ax.bar(np.array(drctn), np.array(spd), normed=True, opening=0.8, edgecolor='white')
ax.set_legend()
ax.set_title('end point')
plt.show()

switch_time=switch_time_spd #comment this out for the switch to be based on current direction switch and not speed

# waypts
#just interp along the line between A and B
interplon=[]
interplat=[]
interpcoords=[]

pt1=Point(latA,lonA)
pt2=Point(latB,lonB)
line = LineString([pt1, pt2])
if line.length<3:
    pts = []
    for div in np.arange(line.length/NP,line.length,line.length/NP):
        pts.extend(line_interpolate_point(line,div).coords[:])
    interpcoords.extend(pts)
        
corecoords=[]  
corelat=[]
corelon=[]    
for i in interpcoords:
    interplon.append(i[1])
    interplat.append(i[0])
    corecoords.append([i[1],i[0]])
    corelat.append(i[0])
    corelon.append(i[1])
    
# corecoords.append([lonB,latB])
    
#check the path
plt.figure()
m = Basemap(projection='merc', llcrnrlat=42, urcrnrlat=43, llcrnrlon=-71.5, urcrnrlon=-69.5, resolution='f')
m.drawcoastlines()
# m.drawmeridians()
xA,yA=m(lonA,latA)
m.plot(xA,yA,'g*')
xB,yB=m(lonB,latB)
m.plot(xB,yB,'r*')
for c in corecoords:
    x,y=m(c[0],c[1])
    m.plot(x,y,'.')
    
#now, since you're going back and forth, append a bunch more core coords 
#probably overkill but whatever
# waypts=np.vstack([corecoords,(corecoords[::-1][1:-1]),corecoords,(corecoords[::-1][1:-1]),corecoords,(corecoords[::-1][1:-1]),corecoords,(corecoords[::-1][1:-1]),corecoords,(corecoords[::-1][1:-1]),corecoords,(corecoords[::-1][1:-1]),corecoords,(corecoords[::-1][1:-1])])
waypts=np.vstack([corecoords,[corecoords[-1]]*20,(corecoords[::-1])])

# plt.figure()
# plt.plot(lonA,latA,'g.')
# plt.plot(lonB,latB,'r.')
# plt.plot(corelon,corelat,'.')

#check distance between points and update threshold if dist<set thresh
distbtpt=haversine(corecoords[0][1],corecoords[0][0],corecoords[1][1],corecoords[1][0])
if distbtpt<distThresh:
    distThresh=distbtpt
    print('Distance between waypoints smaller than distance threshold')
    print('Distance updated to %0.2f km' %(distThresh))

###############
##### run #####
###############
#set up the loop to run tidal cycles
#update the waypoints to the next one in the list after moving to one
#check on each surfacing

siglaytide=[surfacelayer,depthlayer]*int(np.ceil(len(switch_time)/2))
checkcycle=np.array([0]*len(switch_time)) #a list of ones and zeros/true false for wther or not to check. every Nth = 1, have to make sure it's on the depth one
c=np.where(np.array(siglaytide)==depthlayer)[0]
checkcycle[c[::CF]]=1


newlon=lonA
newlat=latA
timeA=np.datetime64(timeA)
timeB=np.datetime64(timeA+timedelta(hours=24))
lastI=0
WP=0
dist=[]
layer=[]
reset_at_coords=[]
reset_to_coords=[]
files=[]
t=2
allLats=[]
allLons=[]
motormask=[]
runtimes=[]
allTimes=[]
driftLayer=[]
Done=False
# for t in range(1,len(switch_time)):
while t<len(switch_time):
    # timeA=switch_time.values[t]
    # runtime=24 #run for 1 day, reset after that
    runtime=int((switch_time.values[t]-np.datetime64(timeA))*2.7778E-13) #runtime between tide switches in hours
    if runtime<=0:
        t=t+1
        runtime=int((switch_time.values[t]-np.datetime64(timeA))*2.7778E-13) #runtime between tide switches in hours
        if runtime<=0:
            t=t+1
            runtime=int((switch_time.values[t]-np.datetime64(timeA))*2.7778E-13) #runtime between tide switches in hours
            
    runtimes.append(runtime)
    #figure out how to change layers based on tide within the 24 hour run? subruns? check everytime you go to surface, not 24 hours?
    siglay=siglaytide[t]

    outFile='WaypointExperiment2_cycle'+str(t)+'_depthlayer_'+str(siglay)
    files.append(outFile)
    
        
    fieldset,d2s=makeFieldsetFVCOMMove(ds,latlim,lonlim,siglay,np.datetime64(timeA)-np.timedelta64(1,'h'),timeB,rectangle=False,driftfactor=driftfactor)

    x = fieldset.U.grid.lon
    y = fieldset.U.grid.lat

    cell_areas = parcels.Field(
        name="cell_areas", data=fieldset.U.cell_areas(), lon=x, lat=y
    )
    fieldset.add_field(cell_areas)

    fieldset.add_constant("Cs", 0.1)
    
    #set up run
    Particle = parcels.JITParticle.add_variables(
        [
            parcels.Variable("t"),
            parcels.Variable("dU"),
            parcels.Variable("dV"),
            parcels.Variable("d2s", initial=1e3),
            parcels.Variable('dMoved',initial=0),
            parcels.Variable('driftdepth'),
            parcels.Variable('ambU'),
            parcels.Variable('ambV')
            
        ]
    )

    pset = parcels.ParticleSet(fieldset=fieldset, pclass=Particle, lon=newlon, lat=newlat, time=timeA)

    # define kernels to use
    # kernels=[displace, parcels.AdvectionRK4, record_depth, record_amb_vel, set_displacement, flag_land, remove_stranded]
    kernels=[displace, parcels.AdvectionRK4, smagdiff,  record_depth, record_amb_vel, set_displacement]

    # Create a ParticleFile object to store the output
    output_file = pset.ParticleFile(
        name=outFile,
        outputdt=timedelta(minutes=15),
        chunks=(1, 4*24*runtime),  # setting to write in chunks of x observations
    )

    # Now execute the kernels for D days, saving data every 15 minutes
    #run
    pset.execute(
        kernels,
        runtime=timedelta(hours=runtime),
        dt=timedelta(minutes=15),
        output_file=output_file
    )

    dsout=xr.open_zarr(outFile+'.zarr')
    
    lastvalid=np.where(~np.isnan(dsout['lat']))[-1][-1]
    allLats.extend(dsout['lat'][0][0:lastvalid].values)
    allLons.extend(dsout['lon'][0][0:lastvalid].values)
    motormask.extend([0]*lastvalid)
    allTimes.extend(dsout['time'][0][0:lastvalid].values)
    driftLayer.extend([siglay]*lastvalid)
    #Now, check the distance between the last valid coordinate and the target waypoint
    #If further than threshold, move to waypoint, and update the waypoint to the next one
    #if .... within threshold? What to do? update waypoint anyway? Yeah I guess? if within 100m check it off the list i guess
    frontlon=float(waypts[WP][0])
    frontlat=float(waypts[WP][1])
    
    d=haversine(frontlat,frontlon,dsout['lat'][0][lastvalid].values,dsout['lon'][0][lastvalid].values)
    dist.append(d)

    if (d>distThresh) & (siglay==depthlayer) & (not(Done)) & (checkcycle[t]==1):
        newlon=frontlon
        newlat=frontlat
        if (waypts[WP][0]!=corecoords[-1][0]) & (waypts[WP][1]!=corecoords[-1][1]) & (setFree==True):
            Done=True
        #then remove that coord so it can't get stuck in a loop?
        waypts[WP]=[0,0]
        reset_at_coords.append([dsout['lon'][0][lastvalid].values,dsout['lat'][0][lastvalid].values])
        reset_to_coords.append([newlon,newlat])
        timeA=dsout['time'][0][lastvalid].values + np.timedelta64(int(d*1000/transitspd),'s') #add the distance traveled times travel speed for deltaT
        timeB=timeA+np.timedelta64(48,'h') 
        # times=np.arange(dsout['time'][0][lastvalid].values,timeA,dtype='datetime64[15m]')

        #create a line between the two points, sample every 100m or so
        # times=np.arange(dsout['time'][0][lastvalid].values,timeA,dtype='datetime64[15m]')
        # npt=len(times) #number of points needed to have one every 15min

        # npt=d*10 #get distance in m, and pull number of points needed to get points every 100m: d*1000/100
        # pt1=Point(dsout['lat'][0][lastvalid].values,dsout['lon'][0][lastvalid].values)
        # pt2=Point(newlat,newlon)
        
        ##this here, replace with path finding
        #"zoom in" to the area of interest for higher resolution
        Dlatlim=[np.nanmin([float(np.nanmin(dsout['lat'].values)),newlat])-0.2,np.nanmax([float(np.nanmax(dsout['lat'].values)),newlat])+0.2]
        Dlonlim=[np.nanmax([float(np.nanmax(dsout['lon'].values)),newlon])+0.2,np.nanmin([float(np.nanmin(dsout['lon'].values)),newlon])-0.2]
        
        pathlat,pathlon=DijkstraPath(ds,dsout['lat'][0][lastvalid].values,dsout['lon'][0][lastvalid].values,newlat,newlon,dsout['time'][0][lastvalid].values,Dlatlim,Dlonlim)

        #flip the path output so it's starting at the start and ending at the end
        #is it flipped or not? I can't tell. 
        pathlat=pathlat[0::10]
        pathlon=pathlon[0::10]
        
        #now, get the times for going 0.5m/s between each pt?
        d2p=[haversine(dsout['lat'][0][lastvalid].values,dsout['lon'][0][lastvalid].values,pathlat[0],pathlon[0])*1000] #in m
        t2p=[d2p[0]*2] #in seconds
        
        for i in range(1,len(pathlat)):
            d2p.append(haversine(pathlat[i-1],pathlon[i-1],pathlat[i],pathlon[i]))
            t2p.append(d2p[i]*2) #in seconds
            
        tt2p=np.cumsum(t2p) 
        times=[]
        for i in range(0,len(tt2p)):
            times.append(dsout['time'][0][lastvalid].values+np.timedelta64(np.round(tt2p[i]).astype('int'),'s'))
        # line = LineString([pt1, pt2])
        # samplat=[]
        # samplon=[]
        # for div in np.arange(line.length/npt,line.length,line.length/npt):
        #     sampcoords=(line_interpolate_point(line,div).coords[:])[0]
        #     samplat.append(sampcoords[0])
        #     samplon.append(sampcoords[1])
            
        # times=np.arange(dsout['time'][0][lastvalid].values,timeA,dtype='datetime64[15m]')

        allLats.extend(pathlat)
        allLons.extend(pathlon)
        motormask.extend([1]*len(pathlon))
        driftLayer.extend([siglay]*len(pathlon))

        allTimes.extend(times)
        
        WP=WP+1
        
    elif d<0.1: #if within 100m just say it's checked
        WP=WP+1
        newlon=float(dsout['lon'][0][lastvalid].values)
        newlat=float(dsout['lat'][0][lastvalid].values)
        timeA=dsout['time'][0][lastvalid].values
        timeB=timeA+ np.timedelta64(12,'h')
        
    else:
         newlon=float(dsout['lon'][0][lastvalid].values)
         newlat=float(dsout['lat'][0][lastvalid].values)
         timeA=dsout['time'][0][lastvalid].values
         timeB=timeA+ np.timedelta64(12,'h')
         
    if WP==(len(waypts)-1):
        WP=1
    
    t=t+1
   

total_dist_moved=[]
for n in range(0,len(reset_at_coords)):
    dm=haversine(reset_to_coords[n][1],reset_to_coords[n][0],reset_at_coords[n][1],reset_at_coords[n][0])
    total_dist_moved.append(dm)

plt.figure()
# Initialize the Basemap

# m = Basemap(projection='merc', llcrnrlat=41.48, urcrnrlat=41.55, llcrnrlon=-70.7, urcrnrlon=-70.6, resolution='f')
m = Basemap(projection='merc', llcrnrlat=41, urcrnrlat=43, llcrnrlon=-71.5, urcrnrlon=-69.5, resolution='f')

m.drawcoastlines()
pathlon=[]
pathlat=[]
lastcoords=[]
for f in files[1:-1]:
    dsout= xr.open_zarr(f+'.zarr')
    
    lastvalid=np.where(~np.isnan(dsout['lat']))[-1][-1]
    if f != files[0]:
        lastx=x[-1]
        lasty=y[-1]

    for n in [0]:
        pathlon.extend(dsout.lon.values[n,0:lastvalid])
        pathlat.extend(dsout.lat.values[n,0:lastvalid])
        x,y=m(dsout.lon.values[n,0:lastvalid],dsout.lat.values[n,0:lastvalid])
        m.plot(x,y)
        if (f != files[0]):
            m.plot([lastx,x[0]],[lasty,y[0]],'k--')
        # m.plot(x[0],y[0],'g*')
        lastcoords.append([dsout.lat.values[n,lastvalid],dsout.lon.values[n,lastvalid]])
    if f==files[0]:
        m.plot(x[0],y[0],'g*')

m.plot(x[-1],y[-1],'r*')


plt.figure()
m = Basemap(projection='merc', llcrnrlat=42, urcrnrlat=42.6, llcrnrlon=-71, urcrnrlon=-70, resolution='f')
m.drawcoastlines()
for c in corecoords:
    x,y=m(c[0],c[1])
    m.plot(x,y,'.')
for l in range(0,len(allLats)):
    x,y=m(allLons[l],allLats[l])
    if motormask[l]==1:
        m.plot(x,y,'k.',markersize=2)
    else:
        m.plot(x,y,'b.',markersize=2)


##check the velocities at the end point

coords=[]
for c in range(0,len(ds.nele)):
    coords.append((ds.nele.latc.values[c],ds.nele.lonc.values[c]))
    
nodes=[]
for c in range(0,len(ds.node)):
    nodes.append((ds.node.lat.values[c],ds.node.lon.values[c]))
    
tree = KDTree(coords)
query_point = [allLons[-1], allLats[-1]]

# Find nearest neighbor
distance, index = tree.query(query_point)

    
nodetree = KDTree(nodes)
# Find nearest neighbor
distancenode, indexnode = nodetree.query(query_point)

ds_point=ds.isel(nele=index,node=indexnode)

plt.figure()
plt.plot(np.sqrt(ds_point.u[:,0]**2+ds_point.v[:,0]**2))

plt.figure()
plt.plot(np.sqrt(ds_point.u[:,-1]**2+ds_point.v[:,-1]**2))


############################################################################################
### here you can go and pull the "ambient" velocities from where the drifter is motoring ###
############################################################################################
dataFile=r'C:\Users\annis_rgr7tey\Documents\NECOFS_WaypointTransits\forecastRunSurfaceFull_extended.nc' #need to download for north of cc
data=xr.open_dataset(dataFile)

ambientU=[]
ambientV=[]
for n in range(0,len(allLats)):
    if motormask[n]==1:
        ambientU.extend([data.sel(latitude=allLats[n],longitude=allLons[n],time=allTimes[n],method='nearest').x_sea_water_velocity.values])
        ambientV.extend([data.sel(latitude=allLats[n],longitude=allLons[n],time=allTimes[n],method='nearest').y_sea_water_velocity.values])
    else:
        ambientU.extend([0])
        ambientV.extend([0])

cols=['Latitude','Longitude','Drift Layer','Motoring Y/N','Ambient U','Ambient V']
df=pd.DataFrame(np.vstack([allLats,allLons,driftLayer,motormask,ambientU,ambientV]).T,columns=cols,index=allTimes[0:len(ambientV)])
df.to_csv(dirPath+'\\logfile.csv')

df=pd.read_csv(dirPath+'\\logfile.csv')

allLats=df['Latitude']
allLons=df['Longitude']
allTimes=pd.to_datetime(df['Unnamed: 0'])
motormask=df['Motoring Y/N']
######################################
### make figure for each time step ###
############ and animate  ############
######################################
# X,Y=np.meshgrid(data.longitude.values,data.latitude.values)
  
# figpath=dirPath+'\\animfigs'

# plt.figure()
# # Initialize the Basemap

# # m = Basemap(projection='merc', llcrnrlat=41.48, urcrnrlat=41.55, llcrnrlon=-70.7, urcrnrlon=-70.6, resolution='f')
# m = Basemap(projection='merc', llcrnrlat=41, urcrnrlat=42, llcrnrlon=-71.5, urcrnrlon=-69.5, resolution='f')

# m.drawcoastlines()

# xm,ym=m(data.longitude.values,data.latitude.values)
# X,Y=np.meshgrid(xm,ym)
# allLons_m,allLats_m=m(allLons,allLats)

# for c in corecoords:
#     cx,cy=m(c[0],c[1])
#     m.plot(cx,cy,'k*')
    
# for n in range(0,len(allLats)):
#     dscut=data.sel(time=allTimes[n],method='nearest')
#     dscut_spd=np.sqrt(dscut.x_sea_water_velocity.values*dscut.x_sea_water_velocity.values+dscut.y_sea_water_velocity.values*dscut.y_sea_water_velocity.values)
#     # plt.figure()
#     # plt.pcolormesh(X,Y,dscut_spd)
#     c=m.pcolormesh(X,Y,dscut_spd,vmin=0,vmax=3)
#     m.fillcontinents('k')

#     if motormask[n]==1:
#         m.plot(allLons_m[n],allLats_m[n],'g.')
#     else:
#         m.plot(allLons_m[n],allLats_m[n],'r.')
#     if n==0:
#         plt.colorbar(c)
#     plt.savefig(figpath+'\\'+str(n)+'.png')
    
# import imageio
# images=[]
# # for file_name in sorted(os.listdir(figpath)):
# for f in range(0,len(os.listdir(figpath))):
#     file_path = os.path.join(figpath, str(f)+'.png')
#     images.append(imageio.imread(file_path))
# imageio.mimsave('AllPathPoints.gif', images)


###################################################################
# ok fine but need to do thiiiiissss somehow with accurate times? #
###################################################################
#loop over times in the model data
#then plot all points prior to or at that time?

figpath=dirPath+'\\animfigs2'
os.makedirs(figpath,exist_ok=True)

cb=False

plt.figure()
# Initialize the Basemap

# m = Basemap(projection='merc', llcrnrlat=41.48, urcrnrlat=41.55, llcrnrlon=-70.7, urcrnrlon=-70.6, resolution='f')
# m = Basemap(projection='merc', llcrnrlat=41, urcrnrlat=42, llcrnrlon=-71.5, urcrnrlon=-69.5, resolution='f')
m = Basemap(projection='merc', llcrnrlat=42, urcrnrlat=43, llcrnrlon=-71, urcrnrlon=-70, resolution='f')


m.drawcoastlines()


xm,ym=m(data.longitude.values,data.latitude.values)
X,Y=np.meshgrid(xm,ym)
allLons_m,allLats_m=m(allLons,allLats)

for c in corecoords:
    cx,cy=m(c[0],c[1])
    m.plot(cx,cy,'k*')
    
       
for n in range(0,len(data.time)):

    dscut=data.isel(time=n)
    dscut_spd=np.sqrt(dscut.x_sea_water_velocity.values*dscut.x_sea_water_velocity.values+dscut.y_sea_water_velocity.values*dscut.y_sea_water_velocity.values)
    # plt.figure()
    # plt.pcolormesh(X,Y,dscut_spd)

        
    ind=np.where(allTimes<=dscut.time.values)[0]
    if len(ind)>20:
        cb=True

        ind=ind[-20:-1]
        
        plt.figure()
        # Initialize the Basemap
        
        # m = Basemap(projection='merc', llcrnrlat=41.48, urcrnrlat=41.55, llcrnrlon=-70.7, urcrnrlon=-70.6, resolution='f')
        # m = Basemap(projection='merc', llcrnrlat=42, urcrnrlat=42.6, llcrnrlon=-71, urcrnrlon=-70, resolution='f')
        m = Basemap(projection='merc', llcrnrlat=42, urcrnrlat=43, llcrnrlon=-71, urcrnrlon=-70, resolution='f')

        m.drawcoastlines()
   
        for cc in corecoords:
            cx,cy=m(cc[0],cc[1])
            m.plot(cx,cy,'k*')
            
    c=m.pcolormesh(X,Y,dscut_spd,vmin=0,vmax=3)
    m.fillcontinents('k')
    if (n==0) or cb:
        plt.colorbar(c,label='[m/s]',orientation='vertical')        
    for i in ind:
        if motormask[i]==1:
            m.plot(allLons_m[i],allLats_m[i],'g--',markersize=1.5)
        else:
            m.plot(allLons_m[i],allLats_m[i],'r.',markersize=1.5)
    plt.title(dscut.time.values)   
    plt.savefig(figpath+'\\'+str(n)+'.png')
    
import imageio
images=[]
# for file_name in sorted(os.listdir(figpath)):
for f in range(0,len(os.listdir(figpath))):
    file_path = os.path.join(figpath, str(f)+'.png')
    images.append(imageio.imread(file_path))
imageio.mimsave('TimeSeries_slow.gif', images,duration=1000)
imageio.mimsave('TimeSeries.gif', images)


#######################################
### now, get the ambient velocities ###
## and get some kind of transit cost ##
#######################################
def uv2compass(u,v):
    import numpy as np
    u=np.array(u)
    v=np.array(v)
    spd=np.sqrt(u*u+v*v)
    drctn = np.degrees(np.arctan2(u, v)) % 360
    
    return spd,drctn

ambientspd,ambientdrctn=uv2compass(ambientU,ambientV)

ambientdrctn[ambientdrctn==0]=np.nan
#now get the bearing of transits
transitdrctn=[]
for n in range(0,len(allLats)-1):
    transitdrctn.append(bearing(allLats[n],allLons[n],allLats[n+1],allLons[n+1]))
transitdrctn.append(0)
    
plt.figure()
plt.plot(transitdrctn,ambientdrctn,'.')

plt.figure()
plt.plot(transitdrctn-ambientdrctn)

plt.figure()
plt.plot(ambientspd)

plt.figure()
plt.plot(ambientspd*(transitdrctn-ambientdrctn))

