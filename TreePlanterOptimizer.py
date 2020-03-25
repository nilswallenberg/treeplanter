import numpy as np
import time
from TreePlanterTreeshade import tsh_gen

def topt(y,x,tmrt1,tmrth,sun_vs_sh,i,dia,treeshadow_clip,sh_bld,tpy,tpx,sh_tmrt,buildings,treesh_sum,sh_vec_t):

    # x = x positions of trees
    # y = y positions of trees
    # tmrt1 = sum of tmrt under each tree
    # tmrt2 = tmrt from SOLWEIG
    # pos = vector with positions of trees and the sum of the tmrt under the shadow
    # i = which tree to move
    # dia = diameter of tree canopy
    # rows = nr rows in dsm
    # cols = nr cols in dsm
    # dectime = dectime, i.e. decimals for tsh_hrs1-4

    t_y = y[i,0]    # y-position of tree to move
    t_x = x[i,0]    # x-position of tree to move

    # Array with movement one step south, one step north, one step west, one step east
    t_yx = np.array(([t_y, t_x], [t_y-1, t_x],[t_y+1, t_x],[t_y, t_x-1],[t_y, t_x+1]))

    # Where not to move, i.e. positions of all other trees
    not_t = ((x[:,0] != t_x) | (y[:,0] != t_y))

    y_n = y[not_t,0]    # y position of trees that are not moving
    x_n = x[not_t,0]    # x position of trees that are not moving

    # Check euclidean distance between trees, i.e. where possible to move
    eucl = np.zeros((y_n.shape[0],t_yx.shape[0]))
    e_bool = np.ones((t_yx.shape[0],1))
    for i in range(y_n.shape[0]):
        yx_temp = np.array((y_n[i], x_n[i]))
        for j in range(t_yx.shape[0]):
            eucl[i,j] = np.linalg.norm(t_yx[j,:] - yx_temp)
            if eucl[i,j] < dia:
                e_bool[j,0] = 0

    e_bool = np.bool_(e_bool)   # Boolean of where it's possible and not to move
    t_yx = t_yx[e_bool[:,0],:]  # Positions where it's possible to move to

    b_bool = np.zeros((t_yx.shape[0],1))
    for i in range(t_yx.shape[0]):
        if sh_bld[t_yx[i,0],t_yx[i,1]] == 1:
            b_bool[i,0] = 1

    b_bool = np.bool_(b_bool)
    t_yx = t_yx[b_bool[:,0],:]

    treesh_rg_n, treesh_bool_n = tsh_gen(y_n, x_n, tpy, tpx, treeshadow_clip,buildings,treesh_sum)

    tmrt_rg_n = np.zeros((y_n.shape[0],3))
    for idx in range(tmrt_rg_n.shape[0]):
        tmrt_rg_n[idx, 0] = y_n[idx]
        tmrt_rg_n[idx, 1] = x_n[idx]
        tmrt_rg_n[idx, 2] = sun_vs_sh[y_n[idx], x_n[idx]]


    tmrt_rg_n_s = np.argsort(-tmrt_rg_n[:,2], axis=0)

    treesh_rg_n_s = np.zeros((treesh_rg_n.shape[0], treesh_rg_n.shape[1], treesh_rg_n.shape[2]))
    treesh_bool_n_s = np.zeros((treesh_bool_n.shape[0], treesh_bool_n.shape[1], treesh_bool_n.shape[2]))
    for idx in range(tmrt_rg_n_s.shape[0]):
        treesh_rg_n_s[:, :, tmrt_rg_n_s[idx]] = treesh_rg_n[:, :, idx]
        treesh_bool_n_s[:, :, tmrt_rg_n_s[idx]] = treesh_bool_n[:, :, idx]

    tree_tmrt = np.zeros((t_yx.shape[0],5))          # y, x, tmrt shade, tmrt sun, tmrt diff
    for i in range(t_yx.shape[0]):
        y_t = np.array([t_yx[i,0]])
        x_t = np.array([t_yx[i,1]])
        treesh_rg, treesh_bool = tsh_gen(y_t, x_t, tpy, tpx, treeshadow_clip, buildings,treesh_sum)

        treesh_rg = treesh_rg[:,:,0]
        treesh_bool = treesh_bool[:,:,0]

        tree_tmrt[i, 0] = tmrt1[y_t, x_t]   # Potential decrease in Tmrt from shade due to other trees
        tree_tmrt[i, 1] = tmrth[y_t, x_t]   # Tmrt in sun for the tree shadow
        tree_tmrt[i, 2] = tmrth[y_t, x_t] - tmrt1[y_t, x_t]  # Difference in Tmrt between sunlit and shaded
        tree_tmrt[i, 3] = y_t               # y position of tree
        tree_tmrt[i, 4] = x_t               # x position of tree

        t3_t = 0

        for j in range(treesh_rg_n_s.shape[2]):
            tree_overlap = treesh_bool_n_s[:,:,j] * treesh_bool
            if np.sum(tree_overlap) > 0:
                treesh1 = treesh_rg * tree_overlap
                treesh2 = treesh_rg_n_s[:,:,j] * tree_overlap
                tree1_u, tree1_c = np.unique(treesh1, return_counts=True)
                tree2_u, tree2_c = np.unique(treesh2, return_counts=True)
                for ix in tree1_u:
                    if ix > 0:
                        t1 = sh_vec_t[sh_vec_t[:,0] == np.int_(ix), 1:]
                        for iy in tree2_u:
                            if iy > 0:
                                t2 = sh_vec_t[sh_vec_t[:,0] == np.int_(iy), 1:]
                                t3 = t1 * t2
                                if np.sum(t3) > 0:      # Kan vara större än 1 om det är fler som matchar...
                                    t1_t = treesh1 == ix
                                    t2_t = treesh2 == iy
                                    t3_t = t1_t * t2_t      # Overlapping timesteps
                                    if np.any(t3_t):        # If there are overlapping timesteps
                                        t3_s = np.sum(t3_t) # How many cells overlap
                                        t3_u = np.where(t3 == 1)    # What timestep is overlapping
                                        t4_s = t3_s * sh_tmrt[t3_u[1][0],0]
                                        t5 = np.zeros((2,1))
                                        t5[0, 0] = tmrt1[y_t, x_t]
                                        t5[1, 0] = tmrt1[y_n[j], x_n[j]]
                                        if t5[0,0] < t5[1,0]:
                                            tree_tmrt[i, 0] = tmrt1[y_t, x_t] - t4_s                # Potential decrease in Tmrt from shade due to other trees
                                            tree_tmrt[i, 1] = tmrth[y_t, x_t]                       # Tmrt in sun for the tree shadow
                                            tree_tmrt[i, 2] = tmrth[y_t, x_t] - tmrt1[y_t, x_t]     # Difference in Tmrt between sunlit and shaded

    nc = 0 # no change

    if np.sum(e_bool) > 0:
        t_max = np.where(tree_tmrt[:,2] == np.max(tree_tmrt[:,2]))          # Which position has highest tmrt
        t_out = tree_tmrt[t_max[0],:]    # Position of where tmrt is highest
        if (t_out[0,3] == t_y) & (t_out[0,4] == t_x):
            nc = 1
    else:
        t_out = np.zeros((1,5))                             # If no position has higher tmrt than current position don't move
        nc = 1

    return t_out, nc