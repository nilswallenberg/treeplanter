import numpy as np
import time
import TreeGenerator.makevegdems
from treeplanterclasses import Treerasters
from treeplanterclasses import Position

def treeplanter(treeinput,treedata,treesh_w,treesh_h1,tmrt_1d,treesh_sum):

    treeinput.tmrt_s = treeinput.tmrt_s * treeinput.buildings     # Remove all Tmrt values that are in shade or on top of buildings

    #bld_copy = treeinput.buildings.copy()

    rows = treeinput.tmrt_s.shape[0]    # Y-extent of studied area
    cols = treeinput.tmrt_s.shape[1]    # X-extent for studied area

    cdsm_ = np.zeros((rows,cols))       # Empty cdsm
    tdsm_ = np.zeros((rows,cols))       # Empty tdsm
    buildings_empty = np.ones((rows, cols))  # Empty building raster

    # Create new tree
    cdsm_, tdsm_ = TreeGenerator.makevegdems.vegunitsgeneration(buildings_empty,cdsm_,tdsm_,treedata.ttype,treedata.height,treedata.trunk,treedata.dia,treedata.treey,treedata.treex,cols,rows,treeinput.scale)

    treerasters = Treerasters(treesh_w, treesh_sum, treesh_h1, cdsm_, treedata.height)

    # Creating boolean for where it is possible to plant a tree
    bd_b = np.int_(treedata.dia / 2)

    # Buffer on building raster so that trees can't be planted next to walls. Can be planted one radius from walls.
    for i1 in range(bd_b):
        walls = np.zeros((rows, cols))
        domain = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        for i in range(1, cols - 1):
            for j in range(1, rows - 1):
                dom = treeinput.buildings[j - 1:j + 2, i - 1:i + 2]
                walls[j, i] = np.min(dom[np.where(domain == 1)])

        walls = treeinput.buildings - walls
        treeinput.buildings = 1 - treeinput.buildings
        treeinput.buildings = treeinput.buildings + walls
        treeinput.buildings = 1 - treeinput.buildings

    # Calculating sum of Tmrt in shade for each possible position in the Tmrt matrix
    sum_tmrt_tsh = np.zeros((rows, cols))   # Empty matrix for sum of Tmrt in tree shadow
    sum_tmrt = np.zeros((rows, cols))  # Empty matrix for sum of Tmrt in sun under tree shadow

    res_y, res_x = np.where(treeinput.buildings == 1)

    vec_ln = res_y.__len__()
    pos_ls = np.zeros((vec_ln, 6))

    index = 0

    ts_rows = treerasters.treeshade.shape[0]
    ts_cols = treerasters.treeshade.shape[1]

    # This loop pastes the tree shadow into all possible positions and calculates the sum of Tmrt under the shadow and
    # the sum Tmrt for the same area but sunlit

    for i in range(res_y.__len__()):
        tpy1 = ts_rows - treerasters.tpy[0]
        tpx1 = ts_cols - treerasters.tpx[0]

        tpy2 = rows - res_y[i]  # idy = 90 -> 101 - 90 = 11
        tpx2 = cols - res_x[i]

        tsh_pos = np.zeros((rows, cols))
        tsh_bool_temp = np.zeros((rows, cols, tmrt_1d.__len__()))

        rows_temp = np.zeros((1, 2))
        rows_temp1 = np.zeros((1, 2))
        cols_temp = np.zeros((1, 2))
        cols_temp1 = np.zeros((1, 2))

        # Calculating positions of how much of the shadow to copy depending on edges of the raster
        # and then where to paste it into the study area
        if (res_y[i] < treerasters.tpy[0]) & (tpy2 > tpy1):
            rows_temp = [int(abs(res_y[i] - treerasters.tpy[0])), int(ts_rows)]
            rows_temp1 = [0, int(res_y[i] + tpy1)]
        elif (res_y[i] < treerasters.tpy[0]) & (tpy2 <= tpy1):
            rows_temp = [int(abs(res_y[i] - treerasters.tpy[0])), int(ts_rows - (tpy1 - tpy2))]
            rows_temp1 = [int(rows - tpy2), rows]
        elif (res_y[i] >= treerasters.tpy[0]) & (tpy2 > tpy1):
            rows_temp = [0, int(ts_rows)]
            rows_temp1 = [int(res_y[i] - treerasters.tpy[0]), int(res_y[i] + tpy1)]
        elif (res_y[i] >= treerasters.tpy[0]) & (tpy2 <= tpy1):
            rows_temp = [0, int(ts_rows - (tpy1 - tpy2))]
            rows_temp1 = [int(res_y[i] - treerasters.tpy[0]), rows]

        if (res_x[i] < treerasters.tpx[0]) & (tpx2 > tpx1):
            cols_temp = [int(abs(res_x[i] - treerasters.tpx[0])), int(ts_cols)]
            cols_temp1 = [0, int(res_x[i] + tpx1)]
        elif (res_x[i] < treerasters.tpx[0]) & (tpx2 <= tpx1):
            cols_temp = [int(abs(res_x[i] - treerasters.tpx[0])), int(ts_cols - (tpx1 - tpx2))]
            cols_temp1 = [int(cols - tpx2), cols]
        elif (res_x[i] >= treerasters.tpx[0]) & (tpx2 > tpx1):
            cols_temp = [0, int(ts_cols)]
            cols_temp1 = [int(res_x[i] - treerasters.tpx[0]), int(res_x[i] + tpx1)]
        elif (res_x[i] >= treerasters.tpx[0]) & (tpx2 <= tpx1):
            cols_temp = [0, int(ts_cols - (tpx1 - tpx2))]
            cols_temp1 = [int(res_x[i] - treerasters.tpx[0]), cols]

        a = treerasters.treeshade[rows_temp[0]:rows_temp[1], cols_temp[0]:cols_temp[1]]
        tsh_pos[rows_temp1[0]:rows_temp1[1], cols_temp1[0]:cols_temp1[1]] = a[:, :]

        b = treerasters.treeshade_bool[rows_temp[0]:rows_temp[1], cols_temp[0]:cols_temp[1], :]
        tsh_bool_temp[rows_temp1[0]:rows_temp1[1], cols_temp1[0]:cols_temp1[1], :] = b[:, :, :]

        w_temp = tsh_pos[:, :].copy()

        t_temp = np.zeros((rows,cols))

        # Removing tmrt for each timestep depending on buildings and building shadows
        for iz in range(tmrt_1d.__len__()):
            sh_temp = treeinput.shadow[:, :, iz]  # Building shadows and tree shadows of existing buildings and trees
            if np.sum(sh_temp) > 0:
                #b_temp = bld_copy  # Buildings
                bool_temp = (treeinput.buildings[:, :] == 0) | (sh_temp[:, :] == 0)  # Boolean of buildings and shadows
                ts_temp = tsh_bool_temp[:, :, iz] * bool_temp * tmrt_1d[iz, 0]   # How much tmrt to remove from tree shadow depending on buildings and existing shadows
                w_temp = w_temp - ts_temp    # Removing tmrt
                t_temp = t_temp + (w_temp > 0) * treeinput.tmrt_ts[:,:,iz]  # Sum tmrt sunlit

        sum_tmrt_tsh[res_y[i], res_x[i]] = np.sum(w_temp)   # Sum of Tmrt in tree shade - matrix
        sum_tmrt[res_y[i], res_x[i]] = np.sum(t_temp)       # Sum of Tmrt in same area as tree shade but sunlit - matrix
        pos_ls[index, 1] = res_x[i]                         # X position of tree
        pos_ls[index, 2] = res_y[i]                         # Y position of tree
        pos_ls[index, 3] = sum_tmrt_tsh[res_y[i], res_x[i]] # Sum of Tmrt in tree shad - vector
        pos_ls[index, 4] = sum_tmrt[res_y[i], res_x[i]]     # Sum of Tmrt in same area as tree shade but sunlit - vector
        pos_ls[index, 5] = 1

        index += 1

    pos_bool = pos_ls[:,3] != 0
    pos_ls = pos_ls[pos_bool,:]
    # Gives a unique value ranging from 0 to length of pos_ls.shape[0], for each position where it is possible to plant a tree
    pos_ls[:, 0] = np.arange(pos_ls.shape[0])

    # Adding sum_tmrt and sum_tmrt_tsh to the Treerasters class as well as calculating the difference between sunlit and shaded
    treerasters.tmrt(sum_tmrt, sum_tmrt_tsh, 1)

    # Adding pos_ls and adding all unique values from pos_ls to their respective positions in a 2D matrix and returns a Positions class
    positions = Position(pos_ls,rows,cols)

    return treerasters, positions