import numpy as np
import time
import TreeGenerator.makevegdems
from treeplanterclasses import Treerasters
from treeplanterclasses import Position

def treeplanter(treeinput,treedata,treesh_w,treesh_h1,tmrt_1d,treesh_sum):

    treeinput.tmrt_s = treeinput.tmrt_s * treeinput.buildings     # Remove all Tmrt values that are in shade or on top of buildings

    bld_copy = treeinput.buildings.copy()

    rows = treeinput.tmrt_s.shape[0]    # Y-extent of studied area
    cols = treeinput.tmrt_s.shape[1]    # X-extent for studied area

    cdsm_ = np.zeros((rows,cols))       # Empty cdsm
    tdsm_ = np.zeros((rows,cols))       # Empty tdsm
    buildings_empty = np.ones((rows, cols))  # Empty building raster

    # Calculate buffer to remove relaxation zone. Relaxation zone = r_zone percentage of rows and cols
    r_zone = 0.05   # Relaxation zone in percentage
    buffer = np.ones((rows, cols))  # Empty vector of ones
    rowb = np.around(rows*r_zone)   # r_zone percentage of padded rows
    colb = np.around(cols*r_zone)   # r_zone percentage of padded cols
    buffer[0:np.int_(rowb),:] = 0       # Setting buffer to 0
    buffer[np.int_(rows-rowb):,:] = 0   # Setting buffer to 0
    buffer[:,0:np.int_(colb)] = 0       # Setting buffer to 0
    buffer[:,np.int_(cols-colb):] = 0   # Setting buffer to 0

    # Create new tree
    cdsm_, tdsm_ = TreeGenerator.makevegdems.vegunitsgeneration(buildings_empty,cdsm_,tdsm_,treedata.ttype,treedata.height,treedata.trunk,treedata.dia,treedata.treey,treedata.treex,cols,rows,treeinput.scale)

    treerasters = Treerasters(treesh_w, treesh_sum, treesh_h1, cdsm_, treedata.height)

    # Creating boolean for where it is possible to plant a tree
    bd_b = np.int_(np.ceil(treedata.dia / 2))

    # Buffer on building raster so that trees can't be planted next to walls. Can be planted one radius from walls.
    for i1 in range(bd_b):
        walls = np.zeros((rows, cols))
        domain = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        for i in range(1, cols - 1):
            for j in range(1, rows - 1):
                dom = bld_copy[j - 1:j + 2, i - 1:i + 2]
                walls[j, i] = np.min(dom[np.where(domain == 1)])

        walls = bld_copy - walls
        bld_copy = 1 - bld_copy
        bld_copy = bld_copy + walls
        bld_copy = 1 - bld_copy

    if treeinput.aoi == 1:  # If area of interest, i.e. polygon to select a smaller area of full simulation, remove all other possible positions for trees
        bld_copy = bld_copy * treeinput.selected_area

    bld_copy = bld_copy * buffer    # Removing buffer from possible positions for trees

    # Calculating sum of Tmrt in shade for each possible position in the Tmrt matrix
    sum_tmrt_tsh = np.zeros((rows, cols))   # Empty matrix for sum of Tmrt in tree shadow
    sum_tmrt = np.zeros((rows, cols))  # Empty matrix for sum of Tmrt in sun under tree shadow

    res_y, res_x = np.where(bld_copy == 1)  # Coordinates for where it is possible to plant a tree (buildings, area of interest excluded)

    pos_ls = np.zeros((res_y.__len__(), 6))      # Length of vectors with y and x positions. Will have x and y positions, tmrt in shade and in sun and an id for each position

    index = 0

    # This loop pastes the tree shadow into all possible positions and calculates the sum of Tmrt under the shadow and
    # the sum Tmrt for the same area but sunlit
    # Padding shadow, building and tmrt rasters with width and length of shadows
    shadow_rasters = np.pad(treeinput.shadow[:,:,:], pad_width=((treerasters.tpy[0], treerasters.tpy[0]), (treerasters.tpx[0], treerasters.tpx[0]), (0,0)), mode='constant', constant_values=0)
    buildings_raster = np.pad(treeinput.buildings[:,:], pad_width=((treerasters.tpy[0], treerasters.tpy[0]), (treerasters.tpx[0], treerasters.tpx[0])), mode='constant', constant_values=0)
    tmrt_rasters = np.pad(treeinput.tmrt_ts[:,:,:], pad_width=((treerasters.tpy[0], treerasters.tpy[0]), (treerasters.tpx[0], treerasters.tpx[0]), (0,0)), mode='constant', constant_values=0)

    rows_b, cols_b = buildings_raster.shape # Rows and cols of padded rasters

    for i in range(res_y.__len__()):
        y1 = (res_y[i] + treerasters.tpy[0]) - treerasters.tpy[0]
        y2 = (res_y[i] + treerasters.tpy[0]) + treerasters.rows_s - treerasters.tpy[0]
        x1 = (res_x[i] + treerasters.tpx[0]) - treerasters.tpx[0]
        x2 = (res_x[i] + treerasters.tpx[0]) + treerasters.cols_s - treerasters.tpx[0]

        ts_temp1 = np.zeros((rows_b,cols_b))
        ts_temp1[y1:y2,x1:x2] = treerasters.treeshade

        # Klistra in skugga!! Fixa för TreePlanterTreeshade.py
        # Gör samma som i TreePlanterOptimizer (regional groups, etc)
        for j in range(tmrt_1d.__len__()):
            ts_temp2 = np.zeros((rows_b, cols_b))
            ts_temp2[y1:y2, x1:x2] = treerasters.treeshade_bool[:, :, j]
            sum_tmrt[res_y[i],res_x[i]] += np.sum(ts_temp2 * buildings_raster * shadow_rasters[:,:,j] * tmrt_rasters[:,:,j])
            sum_tmrt_tsh[res_y[i], res_x[i]] += np.sum(ts_temp2 * buildings_raster * shadow_rasters[:,:,j] * tmrt_1d[j,0])

        pos_ls[index, 1] = res_x[i]                         # X position of tree
        pos_ls[index, 2] = res_y[i]                         # Y position of tree
        pos_ls[index, 3] = sum_tmrt_tsh[res_y[i], res_x[i]] # Sum of Tmrt in tree shade - vector
        pos_ls[index, 4] = sum_tmrt[res_y[i], res_x[i]]     # Sum of Tmrt in same area as tree shade but sunlit - vector
        pos_ls[index, 5] = 1

        index += 1

    pos_bool = pos_ls[:,3] != 0
    pos_ls = pos_ls[pos_bool,:]
    # Gives a unique value ranging from 1 to length of pos_ls.shape[0]+1, for each position where it is possible to plant a tree
    pos_ls[:, 0] = np.arange(1,pos_ls.shape[0]+1)

    # Adding sum_tmrt and sum_tmrt_tsh to the Treerasters class as well as calculating the difference between sunlit and shaded
    treerasters.tmrt(sum_tmrt, sum_tmrt_tsh, 0)

    # Adding pos_ls and adding all unique values from pos_ls to their respective positions in a 2D matrix and returns a Positions class
    positions = Position(pos_ls,rows,cols)

    return treerasters, positions