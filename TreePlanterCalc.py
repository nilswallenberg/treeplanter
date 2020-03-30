import numpy as np
import scipy as sp
import time
import math
#from scipy.ndimage import label
import TreeGenerator.makevegdems
from treeplanterclasses import Treerasters

def treeplanter(treeinput,treesh_w,treesh_h1,sh_tmrt,treedata,treesh_sum):

    # treeinput.buildings
    # treeinput.shadow
    # treeinput.tmrt_ts
    # treeinput.tmrt_s
    # treeinput.scale

    # treedata.ttrype
    # treedata.height
    # treedata.trunk
    # treedata.dia

    treeinput.tmrt_s = treeinput.tmrt_s * treeinput.buildings     # Remove all Tmrt values that are in shade or on top of buildings

    bld_copy = treeinput.buildings.copy()

    rows = treeinput.tmrt_s.shape[0]    # Y-extent of studied area
    cols = treeinput.tmrt_s.shape[1]    # X-extent for studied area

    cdsm_ = np.zeros((rows,cols))       # Empty cdsm
    tdsm_ = np.zeros((rows,cols))       # Empty tdsm
    buildings_empty = np.ones((rows, cols))  # Empty building raster

    # Create new tree
    cdsm_, tdsm_ = TreeGenerator.makevegdems.vegunitsgeneration(buildings_empty,cdsm_,tdsm_,treedata.ttype,treedata.height,treedata.trunk,treedata.dia,treedata.treey,treedata.treex,cols,rows,treeinput.scale)

    # Find min and max rows and cols where there are shadows
    shy,shx = np.where(treesh_w > 0)
    shy_min = np.min(shy)
    shy_max = np.max(shy)+1
    shx_min = np.min(shx)
    shx_max = np.max(shx)+1

    # Cropping to only where there is a shadow
    treeshadow_clip = treesh_w[shy_min:shy_max, shx_min:shx_max]    # clipped shadow image
    treesh_sum_clip = treesh_sum[shy_min:shy_max, shx_min:shx_max]  # clipped shadow image
    tsh_bool = treesh_h1[shy_min:shy_max, shx_min:shx_max]
    tsh_bool = (1-tsh_bool)
    cdsm_clip = cdsm_[shy_min:shy_max, shx_min:shx_max]
    tpy,tpx = np.where(cdsm_clip == treedata.height) # Position of tree in clipped shadow image

    treerasters = Treerasters(treesh_w, treesh_sum, tsh_bool)
    treerasters.treexy(tpy,tpx)

    # Creating boolean for where it is possible to plant a tree
    #bd_b = math.ceil(dia/2)
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

    ts_rows = treeshadow_clip.shape[0]
    ts_cols = treeshadow_clip.shape[1]

    for i in range(res_y.__len__()):
        tpy1 = ts_rows - tpy[0]
        tpx1 = ts_cols - tpx[0]

        tpy2 = rows - res_y[i]  # idy = 90 -> 101 - 90 = 11
        tpx2 = cols - res_x[i]

        tsh_pos = np.zeros((rows, cols))
        tsh_bool_temp = np.zeros((rows, cols, sh_tmrt.__len__()))

        rows_temp = np.zeros((1, 2))
        rows_temp1 = np.zeros((1, 2))
        cols_temp = np.zeros((1, 2))
        cols_temp1 = np.zeros((1, 2))

        if (res_y[i] < tpy[0]) & (tpy2 > tpy1):
            rows_temp = [int(abs(res_y[i] - tpy[0])), int(ts_rows)]
            rows_temp1 = [0, int(res_y[i] + tpy1)]
        elif (res_y[i] < tpy[0]) & (tpy2 <= tpy1):
            rows_temp = [int(abs(res_y[i] - tpy[0])), int(ts_rows - (tpy1 - tpy2))]
            rows_temp1 = [int(rows - tpy2), rows]
        elif (res_y[i] >= tpy[0]) & (tpy2 > tpy1):
            rows_temp = [0, int(ts_rows)]
            rows_temp1 = [int(res_y[i] - tpy[0]), int(res_y[i] + tpy1)]
        elif (res_y[i] >= tpy[0]) & (tpy2 <= tpy1):
            rows_temp = [0, int(ts_rows - (tpy1 - tpy2))]
            rows_temp1 = [int(res_y[i] - tpy[0]), rows]

        if (res_x[i] < tpx[0]) & (tpx2 > tpx1):
            cols_temp = [int(abs(res_x[i] - tpx[0])), int(ts_cols)]
            cols_temp1 = [0, int(res_x[i] + tpx1)]
        elif (res_x[i] < tpx[0]) & (tpx2 <= tpx1):
            cols_temp = [int(abs(res_x[i] - tpx[0])), int(ts_cols - (tpx1 - tpx2))]
            cols_temp1 = [int(cols - tpx2), cols]
        elif (res_x[i] >= tpx[0]) & (tpx2 > tpx1):
            cols_temp = [0, int(ts_cols)]
            cols_temp1 = [int(res_x[i] - tpx[0]), int(res_x[i] + tpx1)]
        elif (res_x[i] >= tpx[0]) & (tpx2 <= tpx1):
            cols_temp = [0, int(ts_cols - (tpx1 - tpx2))]
            cols_temp1 = [int(res_x[i] - tpx[0]), cols]

        a = treeshadow_clip[rows_temp[0]:rows_temp[1], cols_temp[0]:cols_temp[1]]
        tsh_pos[rows_temp1[0]:rows_temp1[1], cols_temp1[0]:cols_temp1[1]] = a[:, :]

        b = tsh_bool[rows_temp[0]:rows_temp[1], cols_temp[0]:cols_temp[1], :]
        tsh_bool_temp[rows_temp1[0]:rows_temp1[1], cols_temp1[0]:cols_temp1[1], :] = b[:, :, :]

        w_temp = tsh_pos[:, :].copy()

        t_temp = np.zeros((rows,cols))

        # Removing tmrt for each timestep depending on buildings and building shadows
        for iz in range(sh_tmrt.__len__()):
            sh_temp = treeinput.shadow[:, :, iz]  # Building shadows and tree shadows of existing buildings and trees
            if np.sum(sh_temp) > 0:
                #b_temp = buildings[:, :]    # Buildings
                b_temp = bld_copy[:, :]  # Buildings
                bool_temp = (b_temp[:, :] == 0) | (sh_temp[:, :] == 0)  # Boolean of buildings and shadows
                ts_temp = tsh_bool_temp[:, :, iz] * bool_temp * sh_tmrt[iz, 0]   # How much tmrt to remove from tree shadow depending on buildings and existing shadows
                w_temp = w_temp - ts_temp    # Removing tmrt
                t_temp = t_temp + (w_temp > 0) * treeinput.tmrt_ts[:,:,iz]  # Sum tmrt sunlit

        sum_tmrt_tsh[res_y[i], res_x[i]] = np.sum(w_temp)
        sum_tmrt[res_y[i], res_x[i]] = np.sum(t_temp)
        pos_ls[index, 1] = res_x[i]
        pos_ls[index, 2] = res_y[i]
        pos_ls[index, 3] = sum_tmrt_tsh[res_y[i], res_x[i]]
        pos_ls[index, 4] = sum_tmrt[res_y[i], res_x[i]]
        pos_ls[index, 5] = 1

        index += 1

    pos_bool = pos_ls[:,3] != 0
    pos_ls = pos_ls[pos_bool,:]
    pos_ls[:, 0] = np.arange(pos_ls.shape[0])

    treerasters.tmrt(sum_tmrt, sum_tmrt_tsh)

    # sum_tmrt_tsh_copy = sum_tmrt_tsh.copy()
    # sum_tmrt_copy = sum_tmrt.copy()
    #
    # sun_vs_tsh = sum_tmrt_copy - sum_tmrt_tsh_copy

    # # Finding courtyards and small separate areas
    # sun_vs_tsh_filtered = sp.ndimage.label(sun_vs_tsh)
    #
    # sun_sh_d = sun_vs_tsh_filtered[0]
    # for i in range(1, sun_sh_d.max()+1):
    #     if np.sum(sun_sh_d[sun_sh_d == i]) < 2000:
    #         sun_vs_tsh[sun_sh_d == i] = 0

    #pos_m = np.zeros((rows,cols))
    #for idx in range(pos_ls.shape[0]):
    #    y = np.int_(pos_ls[idx, 2])
    #    x = np.int_(pos_ls[idx, 1])
    #    pos_m[y,x] = pos_ls[idx,0]

    treerasters.pos(pos_ls)
    test = 0

    # sun_vs_tsh_orig = sum_tmrt_copy - sum_tmrt_tsh_copy
    #
    # w1 = 0.80
    # w2 = (1.0-w1)/4
    # w3 = 0.0
    #
    # k = np.array([[w3, w2, w3], [w2, w1, w2], [w3, w2, w3]])
    # sum_tmrt_convol = sp.ndimage.convolve(sum_tmrt_copy, k, mode='constant', cval=0.0)
    #
    # print(np.sum(k))
    #
    # sum_tmrt_convol = sum_tmrt_convol * (sum_tmrt_copy > 0)
    #
    # #sun_vs_tsh = sum_tmrt_copy - sum_tmrt_tsh_copy
    # sun_vs_tsh = sum_tmrt_convol - sum_tmrt_tsh_copy
    #
    # # Finding courtyards and small separate areas
    # sun_vs_tsh_filtered = sp.ndimage.label(sun_vs_tsh)
    #
    # sun_sh_d = sun_vs_tsh_filtered[0]
    # for i in range(1, sun_sh_d.max()+1):
    #     if np.sum(sun_sh_d[sun_sh_d == i]) < 2000:
    #         sun_vs_tsh[sun_sh_d == i] = 0
    #
    # # Finding courtyards and small separate areas
    # sun_vs_tsh_filtered = sp.ndimage.label(sun_vs_tsh_orig)
    #
    # sun_sh_d = sun_vs_tsh_filtered[0]
    # for i in range(1, sun_sh_d.max() + 1):
    #     if np.sum(sun_sh_d[sun_sh_d == i]) < 2000:
    #         sun_vs_tsh_orig[sun_sh_d == i] = 0
    #
    # a_test = np.around(sun_vs_tsh)
    # b_test = np.around(sun_vs_tsh_orig)
    #
    # sun_vs_tsh_orig = b_test

    return treerasters
    #return pos_ls, sum_tmrt_tsh_copy, sum_tmrt_copy, sun_vs_tsh, treeshadow_clip, treesh_sum_clip, tsh_bool, tpy, tpx, pos_m