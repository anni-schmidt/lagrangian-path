# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 06:02:53 2024
A toolbox of functions
@author: annis_rgr7tey
"""

#https://stackoverflow.com/questions/2361945/detecting-consecutive-integers-in-a-list
def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))



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

      lat1 = radians(lat1)
      lon1 = radians(lon1)
      lat2 = radians(lat2)
      lon2 = radians(lon2)
      y = sin(lon2 - lon1) * cos(lat2)
      x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(lon2 - lon1)
      intbearing = atan2(y, x)
      intbearing = degrees(intbearing)
      intbearing = (intbearing + 360) % 360
    
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

#thanks AI

def uv2spddrctn(u, v):
    import numpy as np
    
    # Calculate speed
    speed = np.sqrt(u**2 + v**2)
    
    # Calculate direction in degrees
    direction = np.degrees(np.arctan2(v, u))

    # Adjust direction to be in the range [0, 360]
    direction = (direction + 360) % 360
    
    return speed, direction
