import numpy as np
from osgeo import gdal, osr
import matplotlib.pylab as plt
import os
import time
import math
import glob
import datetime
import scipy as sp
from scipy.ndimage import label

# Import UMEP tools
import TreeGenerator.makevegdems as makevegdems
from SOLWEIG.shadowingfunction_wallheight_23 import shadowingfunction_wallheight_23
import SOLWEIG.Solweig_v2015_metdata_noload as metload
import SOLWEIG.Solweig1D_2019a_calc as so
from SOLWEIG.clearnessindex_2013b import clearnessindex_2013b
from SOLWEIG.wallalgorithms import findwalls
from SOLWEIG.misc import saveraster

# Import functions and classes for Tree planter
import TreePlanterCalc
import TreePlanterOptInit
from treeplanterclasses import Inputdata
from treeplanterclasses import Treedata
from treeplanterclasses import Regional_groups
from SOLWEIG_1D import tmrt_1d_fun

print(datetime.datetime.now().time())
start_time = time.time()

#### Paths
# Gustaf Adolfs Torg
infolder = 'C:/Users/xwanil/Desktop/Project_4/VegetationProject_Py3/inputdata/GustafAdolfSmall/'
#infolder = 'C:/Users/xwanil/Desktop/Project_4/VegetationProject_Py3/inputdata/GustafAdolfNSmall/'
#infolder = 'C:/Users/xwanil/Desktop/Project_4/VegetationProject_Py3/inputdata/GustafAdolfNorth/'
outfolder = infolder + 'Out_v1/Testing/'
metfilepath = infolder + 'gothenburg_1983_173.txt'

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
tree_input = Inputdata(r_range, sh_fl, tmrt_fl, infolder, rows, cols)

# Loading rasters
dataSet = gdal.Open(infolder + 'DSM_GAT_12_00.tif')
tree_input.dsm = dataSet.ReadAsArray().astype(np.float)
dataSet = gdal.Open(infolder + 'DEM_GAT_12_00.tif')
tree_input.dem = dataSet.ReadAsArray().astype(np.float)

del dataSet

tmrt_1d, azimuth, altitude, amaxvalue = tmrt_1d_fun(metfilepath,tree_input.lon,tree_input.lat,tree_input.dsm,r_range)

# Tree specs
#ttype = 1       # Conifer
ttype = 2       # Deciduous
trunk = 3       # Trunk height
height = 10     # Tree height
dia = 5         # Canopy diameter
treey = math.ceil(rows / 2)  # Y-position of tree in empty setting. Y-position is in the middle of Y.
treex = math.ceil(cols / 2)  # X-position of tree in empty setting. X-position is in the middle of X.

treedata = Treedata(ttype, height, trunk, dia, treey, treex)

bld_orig = tree_input.buildings.copy()

tree_input.tmrt_s = tree_input.tmrt_s * tree_input.buildings  # Remove all Tmrt values that are in shade or on top of buildings

rows = tree_input.tmrt_s.shape[0]  # Y-extent of studied area
cols = tree_input.tmrt_s.shape[1]  # X-extent for studied area

cdsm_ = np.zeros((rows, cols))  # Empty cdsm
tdsm_ = np.zeros((rows, cols))  # Empty tdsm
dsm_empty = np.ones((rows, cols))  # Empty dsm raster
buildings_empty = np.ones((rows, cols))  # Empty building raster

rowcol = rows*cols

cdsm_, tdsm_ = makevegdems.vegunitsgeneration(buildings_empty, cdsm_, tdsm_, treedata.ttype, treedata.height, treedata.trunk, treedata.dia, treedata.treey, treedata.treex,
                                               cols, rows, tree_input.scale)

treebush = np.zeros((rows, cols))  # Empty tree bush matrix

treewalls = np.zeros((rows, cols))  # Empty tree walls matrix
treewallsdir = np.zeros((rows, cols))  # Empty tree walls direction matrix

treesh_ts1 = np.zeros((rows, cols, r_range.__len__()))      # Shade for each timestep, shade = 0
treesh_ts2 = np.zeros((rows, cols, r_range.__len__()))      # Shade for each timestep, shade = 1
treesh_sum_sh = np.zeros((rows,cols))                       # Sum of shade for all timesteps
treesh_sum_tmrt = np.zeros((rows, cols))                    # Sum of tmrt for all timesteps

# Create shadow for new tree
i_c = 0
for i in r_range:
    vegsh, sh, _, wallsh, wallsun, wallshve, _, facesun = shadowingfunction_wallheight_23(tree_input.dem, cdsm_, tdsm_,
                                                                                          azimuth[0][i], altitude[0][i], tree_input.scale,
                                                                                          amaxvalue, treebush, treewalls,
                                                                                          treewallsdir * np.pi / 180.)

    treesh_ts1[:, :, i_c] = vegsh
    treesh_ts2[:, :, i_c] = (1 - vegsh)
    treesh_sum_sh = treesh_sum_sh + treesh_ts2[:,:,i_c] * i
    treesh_sum_tmrt[:, :] = treesh_sum_tmrt + treesh_ts2[:,:,i_c] * tmrt_1d[i_c,0]
    i_c += 1

## Regional groups for tree shadows
shadow_rg = Regional_groups(r_range, treesh_sum_sh, treesh_ts2)

## Creating matrices with Tmrt for tree shadows at each possible position
treerasters, positions = TreePlanterCalc.treeplanter(tree_input, treedata, treesh_sum_tmrt, treesh_ts1, tmrt_1d , treesh_sum_sh)

trees = 2

k_s = 41 # Kernel size, k_s = 21 = 21x21
#i_rand = 0 # Random
i_rand = 1 # ILS with kernel search around local maximum

n_trees = np.array([2,3,4])
#n_r = np.array([100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,3000,4000,5000,6000,7000,8000,9000,10000])
#n_trees = np.array([3])
n_r = np.array([100,200,300,400,500,600,700,800,900,1000])

loop = n_r.shape[0]
n_ir = np.array([0,1])  # Random restart or ILS local maximum random restart
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

            # Running tree planter
            t_y, t_x, unique_tmrt, unique_tmrt_max = TreePlanterOptInit.treeoptinit(treerasters, positions, treedata.dia, tree_input.buildings,
                                                                                                  shadow_rg.timesteps, tmrt_1d, trees, r_iters, k_s, i_rand)

            treeshadow_tmrt = np.zeros((rows, cols, trees))
            cdsm_tmrt = np.zeros((rows, cols, trees))
            tdsm_tmrt = np.zeros((rows, cols, trees))

            treeshadow_new = np.zeros((rows,cols))
            cdsm_new = np.zeros((rows,cols))
            tdsm_new = np.zeros((rows,cols))

            for i in range(0,trees):
                cdsm_ = np.zeros((rows,cols))       # Empty cdsm
                tdsm_ = np.zeros((rows,cols))       # Empty tdsm

                cdsm_tmrt[:,:,i], tdsm_tmrt[:,:,i] = makevegdems.vegunitsgeneration(bld_orig, cdsm_, tdsm_, ttype, height, trunk, dia, t_y[i], t_x[i], cols, rows, tree_input.scale)

                cdsm_new = cdsm_new + cdsm_tmrt[:, :, i]
                tdsm_new = tdsm_new + tdsm_tmrt[:, :, i]

                for j in r_range:
                    vegsh, sh, _, wallsh, wallsun, wallshve, _, facesun = shadowingfunction_wallheight_23(tree_input.dsm, cdsm_tmrt[:,:,i], tdsm_tmrt[:,:,i],
                                                                                                  azimuth[0][j], altitude[0][j], tree_input.scale,
                                                                                                  amaxvalue, treebush, treewalls, treewallsdir * np.pi / 180.)

                    treeshadow_tmrt[:,:,i] = 1-vegsh
                    treeshadow_new = treeshadow_new + treeshadow_tmrt[:,:,i]

            # Outputs
            directory = outfolder

            if not os.path.exists(directory):   # Create folder if not existing
                os.makedirs(directory)

            rnr = str(idx)
            t_p = str(trees) + '_trees'
            fm_str = 'tmrt_' + str(trees) + '_trees_' + str(r_iters) + '_' + rnr + '.tif'

            t_dir = directory + t_p + '/' + h_r1 + '_' + h_r2

            if not os.path.exists(t_dir):  # Create folder if not existing
                os.makedirs(t_dir)

            saveraster(tree_input.dataSet, t_dir + '/treeshadow_' + fm_str, treeshadow_new)
            saveraster(tree_input.dataSet, t_dir + '/cdsm_' + fm_str, cdsm_new)
            #saveraster(dataSet, t_dir + '/tdsm_' + fm_str, tdsm_new)
            #saveraster(dataSet, t_dir + '/sun_vs_tsh_' + fm_str, sun_vs_tsh)
            #saveraster(dataSet, t_dir + '/sum_tmrt.tif', sum_tmrt)
            #saveraster(dataSet, t_dir + '/sum_tmrt_sun_' + fm_str, sum_tmrt_s)
            #saveraster(dataSet, t_dir + '/sum_tmrt_sh_' + fm_str, sum_tmrt_tsh)

            if unique_tmrt_max[0].__len__() <= 1:
                print(unique_tmrt[unique_tmrt_max[0][0],:])
            else:
                for ix in range(unique_tmrt_max[0].__len__()):
                    print(unique_tmrt[unique_tmrt_max[0][ix],:])

            t_max[0,idx] = unique_tmrt[unique_tmrt_max[0][0],trees]
            r_time[0,idx] = time.time() - s_time

            print(t_y)
            print(t_x)

            #print('--- ' + str(time.time() - s_time) + ' seconds to finish ' + str(
            #    r_iters) + ' randomization runs with ' + str(trees) + ' trees ---')

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
        t_out[:, 3] = 0#i_time
        np.around(t_out, 1)
        t_head = 'Iterations Tmrt Time i_Time'
        txt_dir = 'C:/Users/xwanil/Desktop/Project_4/Figures/table_' + str(trees) + '_trees_' + str(i_rand) + 'rand.txt'
        np.savetxt(txt_dir, t_out, fmt='%1.2f', header=t_head)

print('--- ' + str(time.time() - start_time) + ' seconds to finish ---')

print(datetime.datetime.now().time())