import numpy as np
from osgeo import gdal, osr
import matplotlib.pylab as plt
import os
import TreePlanter_v2
import TreePlanter_v3
import TreePlanter_v4
import TreePlanter_v5
import TreePlanter_v2_backup
import time
import math
import glob
import datetime

import makevegdems
from wallalgorithms import findwalls
from shadowingfunction_wallheight_23 import shadowingfunction_wallheight_23
import Solweig_v2015_metdata_noload as metload
from misc import saveraster
import Solweig1D_2019a_calc as so
from clearnessindex_2013b import clearnessindex_2013b
from scipy.ndimage import label
import scipy as sp

import TreePlanterCalc
import TreePlanterOptInit

print(datetime.datetime.now().time())
start_time = time.time()

#### Paths

# Dummy
# infolder = 'C:/Users/xwanil/Desktop/Project_4/VegetationProject_Py3/inputdata/Dummy/'
# #insvffolder = infolder + 'SVF145/'
# outfolder = infolder + 'Out_v1/Test/'
# metfilepath = infolder + 'gothenburg_1983_173_midday.txt'

# Gustaf Adolfs Torg + Kronhusbodarna
# infolder = 'C:/Users/xwanil/Desktop/Project_4/VegetationProject_Py3/inputdata/GustafAdolfsTorg/'
# #insvffolder = infolder + 'SVF145/'
# outfolder = infolder + 'Out_v1/Test/'
# metfilepath = infolder + 'gothenburg_1983_173_midday.txt'

# Gustaf Adolfs Torg
infolder = 'C:/Users/xwanil/Desktop/Project_4/VegetationProject_Py3/inputdata/GustafAdolfSmall/'
#infolder = 'C:/Users/xwanil/Desktop/Project_4/VegetationProject_Py3/inputdata/GustafAdolfNSmall/'
#infolder = 'C:/Users/xwanil/Desktop/Project_4/VegetationProject_Py3/inputdata/GustafAdolfNorth/'
outfolder = infolder + 'Out_v1/Test/'
metfilepath = infolder + 'gothenburg_1983_173.txt'
#metfilepath = infolder + 'gothenburg_1983_173_midday.txt'

#infolder2 = 'C:/Users/xwanil/Desktop/Project_4/VegetationProject_Py3/inputdata/Out_v1/Test/1983/1983_173/'
infolder2 = infolder + 'SOLWEIG_RUN/'

## Searching for shadow and tmrt files in infolder
sh_fl = [f for f in glob.glob(infolder2 + 'shadow_*')]
tmrt_fl = [f for f in glob.glob(infolder2 + 'Tmrt_*')]

# Creating vector with hours from file names
h_fl = np.zeros((sh_fl.__len__(),1))
for iz in range(sh_fl.__len__()):
    h_fl[iz,0] = np.float(sh_fl[iz][108:110])

# Which hours to look between
h_range = 1
h_r1 = '1300'
h_r2 = '1400'
if h_range == 1:
    for ix in range(sh_fl.__len__()):
        if h_r1 in sh_fl[ix]:
            r1 = ix
            break

    for iy in range(sh_fl.__len__()):
        if h_r2 in sh_fl[iy]:
            r2 = iy
            break

else:
    r1 = 0
    r2 = sh_fl.__len__()

r_range = range(r1,r2)

# Reading first file in folder to get extent (rows and cols)
dataSet = gdal.Open(sh_fl[1])
temp = dataSet.ReadAsArray().astype(np.float)
rows = temp.shape[0]
cols = temp.shape[1]

# Loading all shadow and tmrt rasters and summing up all tmrt into one
shadow = np.zeros((rows,cols,r_range.__len__()))
tmrt = np.zeros((rows,cols,r_range.__len__()))
sum_tmrt = np.zeros((rows,cols))

i_c = 0
for i in r_range:
    dataSet1 = gdal.Open(sh_fl[i])
    shadow[:,:,i_c] = dataSet1.ReadAsArray().astype(np.float)
    dataSet2 = gdal.Open(tmrt_fl[i])
    tmrt[:, :, i_c] = dataSet2.ReadAsArray().astype(np.float)
    sum_tmrt = sum_tmrt + tmrt[:,:,i_c]
    i_c += 1

# import scipy as sp
# sigma_x = 3
# sigma_y = 3
# sigma = [sigma_x, sigma_y]
# tmrt_smooth = sp.ndimage.filters.gaussian_filter(tmrt[:,:,0], sigma)
# test = 0
# avg_tmrt = sum_tmrt/i_c
#
# t_out = 'C:/Users/xwanil/Desktop/Project_4/VegetationProject_Py3/inputdata/GustafAdolfNSmall/Out_v1'
# saveraster(dataSet, t_out + '/avg_tmrt_1000_1400.tif', avg_tmrt)

# Loading rasters
#dataSet = gdal.Open(infolder + 'DSM_KR_GAT_LARGER_12_00.tif')
dataSet = gdal.Open(infolder + 'DSM_GAT_12_00.tif')
#dataSet = gdal.Open(infolder + 'DummyDSM.tif')
dsm = dataSet.ReadAsArray().astype(np.float)
#dataSet = gdal.Open(infolder + 'DEM_KR_GAT_LARGER_12_00_RH2000.tif')
dataSet = gdal.Open(infolder + 'DEM_GAT_12_00.tif')
#dataSet = gdal.Open(infolder + 'DummyDEM.tif')
dem = dataSet.ReadAsArray().astype(np.float)

dataSet = gdal.Open(infolder + 'buildings.tif')
buildings = dataSet.ReadAsArray().astype(np.float)
#dataSet = gdal.Open(infolder + 'shadow.tif')
#shadow = dataSet.ReadAsArray().astype(np.float)

#dataSet = gdal.Open(infolder + 'tmrt.tif')
#Tmrt = dataSet.ReadAsArray().astype(np.float)

# find latlon etc.
old_cs = osr.SpatialReference()
# dsm_ref = dsmlayer.crs().toWkt()
dsm_ref = dataSet.GetProjection()
old_cs.ImportFromWkt(dsm_ref)

wgs84_wkt = """
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,
            AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.01745329251994328,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]]"""

new_cs = osr.SpatialReference()
new_cs.ImportFromWkt(wgs84_wkt)

transform = osr.CoordinateTransformation(old_cs, new_cs)

width1 = dataSet.RasterXSize
height1 = dataSet.RasterYSize
gt = dataSet.GetGeoTransform()
minx = gt[0]
miny = gt[3] + width1 * gt[4] + height1 * gt[5]
lonlat = transform.TransformPoint(minx, miny)
geotransform = dataSet.GetGeoTransform()
scale = 1 / geotransform[1]
lon = lonlat[0]
lat = lonlat[1]

# Misc
UTC = 1
met = np.loadtxt(metfilepath, skiprows=1, delimiter=' ')
alt = 10

#rows = Tmrt.shape[0]
#cols = Tmrt.shape[1]

onlyglobal = 1  # 1 if only global radiation exist
landcovercode = 0  # according to landcovrclasses_2018a_orig.txt
metfile = 1  # 1 if time series data is used
useveg = 1  # 1 if vegetation should be considered
ani = 1
cyl = 1
elvis = 0
alt = 3.0

sh = 1.  # 0 if shadowed by building
vegsh = 0.  # 0 if shadowed by tree
svf = 0.6
if useveg == 1:
    svfveg = 0.8
    svfaveg = 0.9
    trans = 0.03
else:
    svfveg = 1.
    svfaveg = 1.
    trans = 1.

absK = 0.7
absL = 0.98
PA = 'STAND'
sensorheight = 2.0

# program start
if PA == 'STAND':
    Fside = 0.22
    Fup = 0.06
    height = 1.1
    Fcyl = 0.28
else:
    Fside = 0.166666
    Fup = 0.166667
    height = 0.75
    Fcyl = 0.20

if metfile == 1:
    met = np.loadtxt(metfilepath, skiprows=1, delimiter=' ')
else:
    met = np.zeros((1, 24)) - 999.
    year = 2011
    month = 6
    day = 6
    hour = 12
    minu = 30

    if (year % 4) == 0:
        if (year % 100) == 0:
            if (year % 400) == 0:
                leapyear = 1
            else:
                leapyear = 0
        else:
            leapyear = 1
    else:
        leapyear = 0

    if leapyear == 1:
        dayspermonth = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        dayspermonth = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    doy = np.sum(dayspermonth[0:month - 1]) + day

    Ta = 25.
    RH = 50.
    radG = 880.
    radD = 150.
    radI = 950.

    met[0, 0] = year
    met[0, 1] = doy
    met[0, 2] = hour
    met[0, 3] = minu
    met[0, 11] = Ta
    met[0, 10] = RH
    met[0, 14] = radG
    met[0, 21] = radD
    met[0, 22] = radI

location = {'longitude': lon, 'latitude': lat, 'altitude': alt}
YYYY, altitude, azimuth, zen, jday, leafon, dectime, altmax = metload.Solweig_2015a_metdata_noload(met, location, UTC)

svfalfa = np.arcsin(np.exp((np.log((1.-svf))/2.)))

dectime_new = np.zeros((dectime.shape[0],1))
dec_doy = int(dectime[0])
for i in range(dectime.shape[0]):
    dectime_new[i,0] = dectime[i] - dec_doy

# Creating vectors from meteorological input
DOY = met[:, 1]
hours = met[:, 2]
minu = met[:, 3]
Ta = met[:, 11]
RH = met[:, 10]
radG = met[:, 14]
radD = met[:, 21]
radI = met[:, 22]
P = met[:, 12]
Ws = met[:, 9]
Twater = []
amaxvalue = dsm.max() - dsm.min()

# load landcover file
sitein = infolder + "landcoverclasses_2018a_orig.txt"
f = open(sitein)
lin = f.readlines()
lc_class = np.zeros((lin.__len__() - 1, 6))
for i in range(1, lin.__len__()):
    lines = lin[i].split()
    for j in np.arange(1, 7):
        lc_class[i - 1, j - 1] = float(lines[j])

# ground material parameters
ground_pos = np.where(lc_class[:, 0 ] == landcovercode)
albedo_g = lc_class[ground_pos, 1]
eground = lc_class[ground_pos, 2]
TgK = lc_class[ground_pos, 3]
Tstart = lc_class[ground_pos, 4]
TmaxLST = lc_class[ground_pos, 5]

# wall material parameters
wall_pos = np.where(lc_class[:, 0] == 99)
albedo_b = lc_class[wall_pos, 1]
ewall =  lc_class[wall_pos, 2]
TgK_wall = lc_class[wall_pos, 3]
Tstart_wall = lc_class[wall_pos, 4]
TmaxLST_wall = lc_class[wall_pos, 5]

# If metfile starts at night
CI = 1.

if ani == 1:
    skyvaultalt = np.atleast_2d([])
    skyvaultazi = np.atleast_2d([])
    skyvaultaltint = [6, 18, 30, 42, 54, 66, 78]
    skyvaultaziint = [12, 12, 15, 15, 20, 30, 60]
    for j in range(7):
        for k in range(1, int(360/skyvaultaziint[j]) + 1):
            skyvaultalt = np.append(skyvaultalt, skyvaultaltint[j])

    skyvaultalt = np.append(skyvaultalt, 90)

    diffsh = np.zeros((145))
    svfalfadeg = svfalfa / (np.pi / 180.)
    for k in range(0, 145):
        if skyvaultalt[k] > svfalfadeg:
            diffsh[k] = 1
else:
    diffsh = []

sh_tmrt = np.zeros((r_range.__len__(),2))
i_c = 0
for i in r_range:
    #print(i)
    # Daily water body temperature
    if (dectime[i] - np.floor(dectime[i])) == 0 or (i == 0):
        Twater = np.mean(Ta[jday[0] == np.floor(dectime[i])])

    # Nocturnal cloudfraction from Offerle et al. 2003
    if (dectime[i] - np.floor(dectime[i])) == 0:
        daylines = np.where(np.floor(dectime) == dectime[i])
        alt = altitude[0][daylines]
        alt2 = np.where(alt > 1)
        rise = alt2[0][0]
        [_, CI, _, _, _] = clearnessindex_2013b(zen[0, i + rise + 1], jday[0, i + rise + 1],
                                                Ta[i + rise + 1],
                                                RH[i + rise + 1] / 100., radG[i + rise + 1], location,
                                                P[i + rise + 1])
        if (CI > 1) or (CI == np.inf):
            CI = 1

    Tmrt, Kdown, Kup, Ldown, Lup, Tg, ea, esky, I0, CI, Keast, Ksouth, Kwest, Knorth, Least, Lsouth, Lwest, \
    Lnorth, KsideI, radIo, radDo, shadow1d = so.Solweig1D_2019a_calc(svf, svfveg, svfaveg, sh, vegsh,  albedo_b, absK, absL, ewall,
                                                           Fside, Fup, Fcyl,
                                                         altitude[0][i], azimuth[0][i], zen[0][i], jday[0][i],
                                                         onlyglobal, location, dectime[i], altmax[0][i], cyl, elvis,
                                                         Ta[i], RH[i], radG[i], radD[i], radI[i], P[i],
                                                         Twater, TgK, Tstart, albedo_g, eground, TgK_wall, Tstart_wall,
                                                         TmaxLST, TmaxLST_wall, svfalfa, CI, ani, diffsh, trans)

    sh_tmrt[i_c,0] = Tmrt
    sh_tmrt[i_c,1] = hours[i]
    i_c += 1

# Tree specs
#ttype = 1       # Conifer
ttype = 2       # Deciduous
trunk = 3       # Trunk height
height = 10     # Tree height
dia = 5         # Canopy diameter
# treex = 25      # Position lon
# treey = 25      # Position lat
sizex = cols
sizey = rows

tmrt_orig = Tmrt.copy()
bld_orig = buildings.copy()

sum_tmrt = sum_tmrt * buildings  # Remove all Tmrt values that are in shade or on top of buildings

rows = tmrt.shape[0]  # Y-extent of studied area
cols = tmrt.shape[1]  # X-extent for studied area

treey = math.ceil(rows / 2)  # Y-position of tree in empty setting. Y-position is in the middle of Y.
treex = math.ceil(cols / 2)  # X-position of tree in empty setting. X-position is in the middle of X.

cdsm_ = np.zeros((rows, cols))  # Empty cdsm
tdsm_ = np.zeros((rows, cols))  # Empty tdsm
dsm_empty = np.ones((rows, cols))  # Empty dsm raster
buildings_empty = np.ones((rows, cols))  # Empty building raster

rowcol = rows*cols

cdsm_, tdsm_ = makevegdems.vegunitsgeneration(buildings_empty, cdsm_, tdsm_, ttype, height, trunk, dia, treey, treex,
                                               cols, rows, scale)

# cdsm_1, tdsm_1 = makevegdems.vegunitsgeneration(buildings_empty, cdsm_, tdsm_, ttype, height, trunk, dia, 10, 84,
# #                                                cols, rows, scale)

treebush = np.zeros((rows, cols))  # Empty tree bush matrix

treewalls = np.zeros((rows, cols))  # Empty tree walls matrix
treewallsdir = np.zeros((rows, cols))  # Empty tree walls direction matrix

treesh_w = np.zeros((rows, cols))
treesh_h1 = np.zeros((rows, cols, r_range.__len__()))
treesh_h3 = np.zeros((rows, cols, r_range.__len__()))
treesh_sum = np.zeros((rows,cols))

# vegsh1, sh, _, wallsh, wallsun, wallshve, _, facesun = shadowingfunction_wallheight_23(dsm_empty, cdsm_1, tdsm_1,
#                                                                                           azimuth[0][i], altitude[0][i], scale,
#                                                                                           amaxvalue, treebush, treewalls,
#                                                                                           treewallsdir * np.pi / 180.)
#
# t_out = 'C:/Users/xwanil/Desktop/Project_4/VegetationProject_Py3/inputdata/GustafAdolfNSmall/Out_v1'
# saveraster(dataSet, t_out + '/treeshadow.tif', vegsh1)

# Create shadow for new tree
i_c = 0
for i in r_range:
    vegsh, sh, _, wallsh, wallsun, wallshve, _, facesun = shadowingfunction_wallheight_23(dem, cdsm_, tdsm_,
                                                                                          azimuth[0][i], altitude[0][i], scale,
                                                                                          amaxvalue, treebush, treewalls,
                                                                                          treewallsdir * np.pi / 180.)

    treesh_h1[:, :, i_c] = vegsh
    vegsh = (1 - vegsh)
    treesh_sum = treesh_sum + vegsh * i
    treesh_h3[:,:,i_c] = vegsh
    treesh_w[:, :] = treesh_w + vegsh * sh_tmrt[i_c,0]
    i_c += 1

# t_out = 'C:/Users/xwanil/Desktop/Project_4/VegetationProject_Py3/inputdata/GustafAdolfSmall/Out_v1'
# saveraster(dataSet, t_out + '/treeshadow_dem.tif', vegsh)
# saveraster(dataSet, t_out + '/cdsm_dem.tif', cdsm_)

t_r = range(0,r_range.__len__())
t_l = t_r.__len__()


## Regional groups for tree shadows
tree_u = np.unique(treesh_sum)                      # Unique values in summation matrix for tree shadows
tree_max = np.max(tree_u)                           # Maximum value of unique values
for i in range(1,tree_u.shape[0]):                  # Loop over all unique values
    treesh_b = (treesh_sum == tree_u[i])            # Boolean shadow for each timestep i
    tree_r = label(treesh_b)                        # Create regional groups
    tree_r_u = np.unique(tree_r[0])                 # Find out how many regional groups, i.e. unique values
    if np.sum(tree_r_u) > 1:                        # If more than there groups, i.e. 0, 1, 2, ... , continue
        for j in range(2,tree_r_u.shape[0]):        # Loop over the unique values and give all but 1 new values
            treesh_b2 = (tree_r[0] == tree_r_u[j])  # Boolean of shadow for each unique value
            tree_max += 1                           # Add +1 to the maximum value of unique values, continues (creates new unique values)
            treesh_sum[treesh_b2] = tree_max        # Add these to the building summation matrix

tree_u_u = np.unique(treesh_sum)                    # New unique values of regional groups
sh_vec_t = np.zeros((tree_u_u.shape[0],t_l+1))      # Empty array for storing which timesteps are found in each regional group
sh_vec_t[:,0] = tree_u_u                            # Adding the unique regional groups to the first column
for i in range(1,tree_u_u.shape[0]):                  # Loop over the unique values
    treesh_b = (treesh_sum == tree_u_u[i])          # Boolean of each regional group
    for j in t_r:                                   # Loop over each timestep
        #if i != j:
        treesh_b2 = (treesh_h3[:,:,j].copy() == 1)  # Boolean of shadow for each timestep
        treesh_b3 = (treesh_b) & (treesh_b2)        # Find out where they overlap, i.e. which timesteps are found in each regional group
        if np.sum(treesh_b3) > 0:                   # If they overlap, continue
            sh_vec_t[i,1+j] = 1                     # Add 1 to timestep column

## Regional groups for building shadows
# bldsh_sum = np.zeros((rows,cols))                   # Empty matrix for summation of building shadows
# for i in t_r:                                       # Loop for all timesteps
#     sh_t = (1 - abs(shadow[:,:,i]))                 # Boolean shadow for each timestep i
#     sh_t[sh_t < 0] = 0                              # Remove weird values
#     bldsh_sum = bldsh_sum + sh_t * i                # Add shadow to summation
#
# bld_u = np.unique(bldsh_sum)                        # Unique values in summation matrix
# bld_max = np.max(bld_u)                             # Maximum value of unique values
# for i in range(1,bld_u.shape[0]):                   # Loop over all unique values
#     bldsh_b = (bldsh_sum == bld_u[i])               # Boolean of shadow for each unique value
#     bld_r = label(bldsh_b)                          # Create regional groups
#     bld_r_u = np.unique(bld_r[0])                   # Find out how many regional groups, i.e. unique values
#     if np.sum(bld_r_u) > 1:                         # If more than there groups, i.e. 0, 1, 2, ... , continue
#         for j in range(2,bld_r_u.shape[0]):         # Loop over the unique values and give all but 1 new values
#             bldsh_b2 = (bld_r[0] == bld_r_u[j])     # Boolean of shadow for each unique value
#             bld_max += 1                            # Add +1 to the maximum value of unique values, continues (creates new unique values)
#             bldsh_sum[bldsh_b2] = bld_max           # Add these to the building summation matrix
#
# bld_u_u = np.unique(bldsh_sum)                      # New unique values of regional groups
# sh_vec_b = np.zeros((bld_u_u.shape[0],t_l+1))       # Empty array for storing which timesteps are found in each regional group
# sh_vec_b[:,0] = bld_u_u                             # Adding the unique regional groups to the first column
# for i in range(bld_u_u.shape[0]):                   # Loop over the unique values
#     bldsh_b = (bldsh_sum == bld_u_u[i])             # Boolean of each regional group
#     for j in t_r:                                   # Loop over each timestep
#         #if i != j:
#         bldsh_b2 = (shadow[:,:,j].copy() == 0)      # Boolean of shadow for each timestep
#         bldsh_b3 = (bldsh_b) & (bldsh_b2)           # Find out where they overlap, i.e. which timesteps are found in each regional group
#         if np.sum(bldsh_b3) > 0:                    # If they overlap, continue
#             sh_vec_b[i,1+j] = 1                     # Add 1 to timestep column

pos_ls, sum_tmrt_tsh_out, sum_tmrt_out, sun_vs_tsh, treeshadow_clip, treesh_sum_clip, tsh_bool, tpy, tpx, pos_m \
    = TreePlanterCalc.treeplanter(buildings, shadow, tmrt, sum_tmrt, treesh_w, treesh_h1, sh_tmrt, ttype, height, trunk, dia, scale, treesh_sum)

i_time = time.time() - start_time
#t_out = 'C:/Users/xwanil/Desktop/Project_4/VegetationProject_Py3/inputdata/GustafAdolfSmall/Smoothing/'
#saveraster(dataSet, t_out + '/tmrt_diff_around.tif', sun_vs_tsh_orig)
#saveraster(dataSet, t_out + '/tmrt_diff_around.tif', sun_vs_tsh)
# saveraster(dataSet, t_out + '/cdsm_dem.tif', cdsm_)

#print('--- %s seconds loading and preparing ---' % (time.time() - start_time))
#print(np.where(sun_vs_tsh == np.max(sun_vs_tsh)))

trees = 2

from scipy.signal import argrelextrema

ind_y, ind_x = argrelextrema(sun_vs_tsh, np.greater)

loc_max = np.zeros((ind_y.shape[0],1))
for i in range(ind_y.shape[0]):
    loc_max[i,0] = sun_vs_tsh[ind_y[i],ind_x[i]]

sort_ind = np.argsort(-loc_max[:,0])
ind_y_s = ind_y[sort_ind]
ind_x_s = ind_x[sort_ind]

ind_y_s = ind_y_s[0:trees]
ind_x_s = ind_x_s[0:trees]

test = 0

# plt.figure(num=1)
# plt.plot(sun_vs_tsh[25:98,36], 'k')
# plt.ylim(8500,15000)
# plt.title('Sun vs. Shade (10:00 - 14:00)')
# plt.ylabel(r'$\delta$T$_{mrt}$')
# plt.xlabel('Pixels')
# #frame = plt.gca()
# #frame.axes.get_xaxis().set_visible(False)
#
# plt.figure(num=2)
# plt.plot(sun_vs_tsh[31,13:140], 'k')
# plt.ylim(8500,15000)
# plt.ylabel(r'$\delta$T$_{mrt}$')
# plt.xlabel('Pixels')
# #frame = plt.gca()
# #frame.axes.get_xaxis().set_visible(False)

# plt.figure(num=1)
# plt.plot(np.diag(sun_vs_tsh, 36)[19:98], 'k')
# plt.ylim(10500,15000)
# plt.ylabel(r'$\delta$T$_{mrt}$')
# plt.xlabel('Pixels')
# plt.title(r'T$_{mrt}$ sun - T$_{mrt}$ shade (10:00 - 14:00)')
# #plt.text(8,14700, 'Global maximum', fontsize=10)
# plt.annotate('Global maximum', xy=(6,14800), xytext=(13,14700),
#              arrowprops=dict(arrowstyle='simple', facecolor='black'))
# plt.annotate('Local maximum', xy=(65,13850), xytext=(60,14300),
#              arrowprops=dict(arrowstyle='simple', facecolor='black'))
# plt.annotate('', xy=(77,13650), xytext=(75,14250),
#              arrowprops=dict(arrowstyle='simple', facecolor='black'))
# plt.annotate('', xy=(71.5,13600), xytext=(71.5,14250),
#              arrowprops=dict(arrowstyle='simple', facecolor='black'))
# # plt.text(42,13470,'x', fontsize=17)
# # plt.annotate('', xy=(58,13800), xytext=(44,13700),
# #              arrowprops=dict(arrowstyle='simple', facecolor='black'))
# plt.text(33,13490,'x', fontsize=17)
# plt.annotate('', xy=(20,13950), xytext=(34,13750),
#              arrowprops=dict(arrowstyle='simple', facecolor='black'))
# #ax = plt.gca()
# #ax.plot(50,13500,'o',ms=10,mec='b',mfc='b')
# #frame = plt.gca()
# #frame.axes.get_xaxis().set_visible(False)
#
# plt.savefig(outfolder + 'tmrt_global_local_max2.tif', dpi=150)

#plt.show()

k_s = 41 # Kernel size, k_s = 21 = 21x21
#i_rand = 0 # Random
i_rand = 1 # ILS with kernel search around local maximum
# loop = 1
n_trees = np.array([2])
n_r = np.array([10000])

#n_trees = np.array([4,5,6,7,8])
#n_trees = np.array([2,3,4,5,6,7,8,9,10])
#n_r = np.array([5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000])
#n_r = np.array([10000])

n_trees = np.array([2,3,4])
#n_trees = np.array([2,3])
n_r = np.array([100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,3000,4000,5000,6000,7000,8000,9000,10000])
#n_r = np.array([100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500])
#n_r = np.array([100,200,300,400,500])

loop = n_r.shape[0]
n_ir = np.array([0,1])
#n_ir = np.array([1])
for idz in n_ir:
    i_rand = idz
    for idy in n_trees:
        t_max = np.zeros((1, n_r.shape[0]))
        r_time = np.zeros((1, n_r.shape[0]))
        trees = idy

        for idx in range(loop):
            s_time = time.time()
            r_iters = n_r[idx]
            #trees = n_trees[idx]

            # Running tree planter
            t_y, t_x, unique_tmrt2, unique_tmrt2_counts, t_u_max = TreePlanterOptInit.treeoptinit(pos_ls, sum_tmrt_tsh_out, sum_tmrt_out, sun_vs_tsh, dia, treeshadow_clip,
                                                      tpy, tpx, \
                                                      sh_tmrt, buildings, treesh_sum_clip, sh_vec_t, trees, r_iters, pos_m, k_s, i_rand)

            treeshadow_tmrt = np.zeros((rows, cols, trees))
            cdsm_tmrt = np.zeros((rows, cols, trees))
            tdsm_tmrt = np.zeros((rows, cols, trees))

            treeshadow_new = np.zeros((rows,cols))
            cdsm_new = np.zeros((rows,cols))
            tdsm_new = np.zeros((rows,cols))

            for i in range(0,trees):
                cdsm_ = np.zeros((rows,cols))       # Empty cdsm
                tdsm_ = np.zeros((rows,cols))       # Empty tdsm

                cdsm_tmrt[:,:,i], tdsm_tmrt[:,:,i] = makevegdems.vegunitsgeneration(bld_orig, cdsm_, tdsm_, ttype, height, trunk, dia, t_y[i], t_x[i], cols, rows, scale)

                cdsm_new = cdsm_new + cdsm_tmrt[:, :, i]
                tdsm_new = tdsm_new + tdsm_tmrt[:, :, i]

                for j in r_range:
                    vegsh, sh, _, wallsh, wallsun, wallshve, _, facesun = shadowingfunction_wallheight_23(dsm, cdsm_tmrt[:,:,i], tdsm_tmrt[:,:,i],
                                                                                                  azimuth[0][j], altitude[0][j], scale,
                                                                                                  amaxvalue, treebush, treewalls, treewallsdir * np.pi / 180.)

                    treeshadow_tmrt[:,:,i] = vegsh

                    treeshadow_tmrt[:,:,i] = 1-treeshadow_tmrt[:,:,i]
                    treeshadow_new = treeshadow_new + treeshadow_tmrt[:,:,i]

            # Outputs
            #directory = outfolder + str(int(YYYY[0, i])) + '/' + str(int(YYYY[0, i])) + '_' + str(int(DOY[i]))
            directory = outfolder
            #print(directory)
            if not os.path.exists(directory):   # Create folder if not existing
                os.makedirs(directory)

            rnr = str(idx)
            t_p = str(trees) + '_trees'
            fm_str = 'tmrt_' + str(trees) + '_trees_' + str(r_iters) + '_' + rnr + '.tif'

            t_dir = directory + t_p + '/' + h_r1 + '_' + h_r2

            #print(t_dir)
            if not os.path.exists(t_dir):  # Create folder if not existing
                os.makedirs(t_dir)

            #saveraster(dataSet, t_dir + '/treeshadow_' + fm_str, treeshadow_new)
            #saveraster(dataSet, t_dir + '/cdsm_' + fm_str, cdsm_new)
            #saveraster(dataSet, t_dir + '/tdsm_' + fm_str, tdsm_new)
            #saveraster(dataSet, t_dir + '/sun_vs_tsh_' + fm_str, sun_vs_tsh)
            #saveraster(dataSet, t_dir + '/sum_tmrt.tif', sum_tmrt)
            #saveraster(dataSet, t_dir + '/sum_tmrt_sun_' + fm_str, sum_tmrt_s)
            #saveraster(dataSet, t_dir + '/sum_tmrt_sh_' + fm_str, sum_tmrt_tsh)

            if t_u_max[0].__len__() <= 1:
                print(unique_tmrt2[t_u_max[0][0],:])
            else:
                for ix in range(t_u_max[0].__len__()):
                    print(unique_tmrt2[t_u_max[0][ix],:])

            t_max[0,idx] = unique_tmrt2[t_u_max[0][0],trees]
            r_time[0,idx] = time.time() - s_time

            print('--- ' + str(time.time() - s_time) + ' seconds to finish ' + str(
                r_iters) + ' randomization runs with ' + str(trees) + ' trees ---')

        # Figure
        plt.scatter(n_r, t_max)
        plt.xlabel('Iterations')
        plt.ylabel(r'Decrease in T$_{mrt}$')
        f_name = 'C:/Users/xwanil/Desktop/Project_4/Figures/iters_vs_tmrt_' + str(trees) + '_trees_' + str(i_rand) + 'rand.tif'
        plt.savefig(f_name, dpi=150)
        plt.clf()

        # Table
        t_out = np.zeros((n_r.shape[0], 4))
        t_out[:, 0] = n_r
        t_out[:, 1] = t_max
        t_out[:, 2] = r_time
        t_out[:, 3] = i_time
        np.around(t_out, 1)
        t_head = 'Iterations Tmrt Time i_Time'
        txt_dir = 'C:/Users/xwanil/Desktop/Project_4/Figures/table_' + str(trees) + '_trees_' + str(i_rand) + 'rand.txt'
        np.savetxt(txt_dir, t_out, fmt='%1.2f', header=t_head)

print('--- ' + str(time.time() - start_time) + ' seconds to finish ---')

#print('--- ' + str(time.time() - start_time) + ' seconds to finish ' + str(loop) + ' loops of ' + str(r_iters) + ' randomization runs with ' + str(trees) + ' trees ---')

#print('y positions are = ' + str(t_y) + ' and x positions are = ' + str(t_x))
#print('x position of tree 1 is = ' + str(pos_ls[np.int_(unique_tmrt2[t_u_max[0],0][0]),1]))
#print('y position of tree 1 is = ' + str(pos_ls[np.int_(unique_tmrt2[t_u_max[0],0][0]),2]))
#print('x position of tree 2 is = ' + str(pos_ls[np.int_(unique_tmrt2[t_u_max[0],1][0]),1]))
#print('y position of tree 2 is = ' + str(pos_ls[np.int_(unique_tmrt2[t_u_max[0],1][0]),2]))
#print('Number of times trees end up in warmest position are = ' + str(unique_tmrt2_counts[t_u_max[0]][0]))
#print('Max unique count = ' + str(np.max(unique_tmrt2_counts)))

print(datetime.datetime.now().time())

test = 0