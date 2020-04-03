import numpy as np
import itertools
from scipy.ndimage import label
import time
import datetime
import TreePlanterOptimizer
from TreePlanterTreeshade import tsh_gen
from TreePlanterTreeshade import tsh_gen_ts
from adjustments import treenudge

def combine(tup, t):
    return tuple(itertools.combinations(tup, t))

def treeoptinit(treerasters, treeinput, positions, treedata, shadow_rg, tmrt_1d, trees, r_iters):

    dia = treedata.dia  # Diameter of tree canopy

    counter = 0  # Counter for randomization runs

    i_tmrt = np.zeros((r_iters)) # Empty vector to be filled with Tmrt values for each tree
    i_y = np.zeros((r_iters, trees))    # Empty vector to be filled with corresponding y position of the above
    i_x = np.zeros((r_iters, trees))    # Empty vector to be filled with corresponding x position of the above

    tree_pos_all = np.zeros((r_iters,trees))

    pos_m_pad_t = np.pad(positions.pos_m, pad_width=((1, 1), (1, 1)), mode='constant',
                       constant_values=0)

    pos_temp = positions.pos[:, 0]

    break_loop = 0

    # Iterate for r_iters number of iterations. Will find optimal positions for trees and return the positions and decrease in tmrt
    for counter in range(r_iters):
    #while counter < r_iters:
        #start_time_full = time.time()
        print(counter)
        r_count = 0
        while r_count < 1:

            # Creating trees at random positions
            tree_pos = np.random.choice(pos_temp, trees)  # Random positions for trees
            tp_c = 0
            while np.any(np.where((tree_pos_all == np.sort(tree_pos)).all(axis=1))):
                tree_pos = np.random.choice(pos_temp, trees)
                tp_c += 1
                print('re-random')
                if tp_c == 100:
                    #counter = r_iters + 1
                    break_loop = r_iters + 1
                    break

            if tp_c == 100:
                break

            tree_pos_y = np.zeros((trees, 1), dtype=int)  # Random y-positions for trees
            tree_pos_x = np.zeros((trees, 1), dtype=int)  # Random x-positions for trees

            # Y and X positions of starting positions
            for i2 in range(tree_pos.__len__()):
                tree_pos_x[i2, 0] = positions.pos[positions.pos[:, 0] == tree_pos[i2], 1]
                tree_pos_y[i2, 0] = positions.pos[positions.pos[:, 0] == tree_pos[i2], 2]

            # Euclidean distance between random positions so that trees are not too close to each other
            it_comb = combine(tree_pos, 2)
            eucl_dist = np.zeros((it_comb.__len__(), 1))

            for i3 in range(it_comb.__len__()):
                ax = positions.pos[positions.pos[:, 0] == it_comb[i3][0], 1]
                ay = positions.pos[positions.pos[:, 0] == it_comb[i3][0], 2]
                a = np.array(([ay[0], ax[0]]))
                bx = positions.pos[positions.pos[:, 0] == it_comb[i3][1], 1]
                by = positions.pos[positions.pos[:, 0] == it_comb[i3][1], 2]
                b = np.array(([by[0], bx[0]]))
                eucl_dist[i3, 0] = np.linalg.norm(a - b)

            if np.min(eucl_dist[:, 0]) >= dia:
                r_count = 1
                # Lägg till alla positioner för eucl >= dia

            #r_count = 1
            tree_pos_all[counter] = np.sort(tree_pos)

        if break_loop == r_iters + 1:
            break

        tp_nc = np.zeros((trees, 1))    # 1 if tree is in the same position as in previous iteration, otherwise 0
        tp_nc_a = np.zeros((trees))     # 1 if tree is stuck but have been checked for better position and none was found, otherwise 0

        ti = itertools.cycle(range(trees)) # Iterator to move between trees moving around in the study area

        # Create matrices for tree paths and add starting positions
        tree_paths_temp = np.zeros((treerasters.rows, treerasters.cols, trees))
        for t_i in range(trees):
            tree_paths_temp[tree_pos_y[t_i],tree_pos_x[t_i], t_i] = 1

        a_c = 0

        # Moving trees, i.e. optimization
        while np.sum(tp_nc[:,0]) < trees:

            i = ti.__next__()

            # Running optimizer
            # t1 = best shading position
            start_time1 = time.time()
            t1, nc, y_out, x_out = TreePlanterOptimizer.topt(tree_pos_y, tree_pos_x, treerasters, treeinput, dia, shadow_rg, tmrt_1d, positions, i, pos_m_pad_t)
            #print('Running optimizer = ' + str(time.time() - start_time1))
            tp_nc[i, 0] = nc

            if tp_nc[i,0] == 0:
                tree_pos_y = y_out
                tree_pos_x = x_out

                tp_nc_a[i] = 0

            # Possibly moving trees that are stuck, where tree shadows intersect
            elif (tp_nc_a[i] == 0) & (tp_nc[i,0] == 1):
                start_time = time.time()
                y_out, x_out, tp_nc, tp_nc_a, t1 = treenudge(y_out, x_out, tp_nc, tp_nc_a, i, t1, counter, i_tmrt, treerasters, treeinput, positions,
                          tmrt_1d)
                if (tp_nc_a[i] == 0) & (tp_nc[i] == 0):
                    tree_pos_y = y_out
                    tree_pos_x = x_out
                #print('Tree nudger = ' + str(time.time() - start_time))

            if tp_nc[i,0] == 0: # Tree paths of current random run
                tree_paths_temp[tree_pos_y[i], tree_pos_x[i], i] = 1

            # Changing position of tree
            if t1[0, 2] > i_tmrt[counter]:
                i_tmrt[counter] = t1[0, 2]
                i_x[counter, :] = tree_pos_x[:, 0]
                i_y[counter, :] = tree_pos_y[:, 0]

        #print((time.time() - start_time) * 5000)
        if i_tmrt[counter] == np.max(i_tmrt):   # Path of trees with best positions from starting to ending
            tree_paths = tree_paths_temp.copy()

        #counter += 1

        #print('Counter = ' + str(counter))
        #print('Time = ' + str(time.time() - start_time))

        if tp_c == 100:
            break

        #print(time.time() - start_time_full)

    # Finding best position from all r_iters iteration, i.e. if r_iters = 1000 then best position out of 1000 runs
    t_max = np.max(i_tmrt)
    y = np.where(i_tmrt == t_max)

    # Calculate heat map for best 33 % positions

    # Returns all the unique positions found by the algorithm and the potential decrease in tmrt
    from misc import max_tmrt
    unique_tmrt, unique_tmrt_max = max_tmrt(i_y, i_x, i_tmrt, trees, positions.pos)

    # Optimal positions of trees
    i_y = i_y[y[0][0], :]
    i_x = i_x[y[0][0], :]

    print(datetime.datetime.now().time())

    return i_y, i_x, unique_tmrt, unique_tmrt_max, tree_paths

# tree_adjusted = 0
# if np.sum(tp_nc[:,0]) > 1:  # If more than one tree is standing still, check if shadows are next to each other
#     nc_r = np.where(tp_nc[:, 0] == 1)  # Rows of trees standing still
#     nc_y = np.squeeze(y_out[nc_r])  # Y-positions of trees
#     nc_x = np.squeeze(x_out[nc_r])  # X-positions of trees
#     tsh_rg_nc, tsh_bool_nc, _, __, ___ = tsh_gen(nc_y, nc_x, treerasters, treeinput)  # Shadows of trees
#     tsh_bool_all_nc = np.zeros((tsh_bool_nc.shape[0], tsh_bool_nc.shape[1]))  # Empty matrix to put shadows in
#     for iy in range(tsh_bool_nc.shape[2]):  # Loop over shadows
#         tsh_bool_all_nc += tsh_bool_nc[:, :, iy]  # Add each shadow
#     label_nc, num_feat_nc = label(tsh_bool_all_nc)  # Regional groups
#     if np.sum(tp_nc[:,0]) > num_feat_nc:  # If there are less regional groups than trees standing still, some must be side by side
#         nc_y_c = np.squeeze(y_out.copy())
#         nc_x_c = np.squeeze(x_out.copy())
#         # Find out label, i.e. regional group of shadows
#         nc_label = np.zeros((y_out.shape[0]))
#         for iy in range(nc_y.shape[0]):
#             nc_y_t = np.int_(y_out[iy] + treerasters.tpy[0])
#             nc_x_t = np.int_(x_out[iy] + treerasters.tpx[0])
#             nc_label[iy] = label_nc[nc_y_t, nc_x_t]
#
#         nc_i_y = np.int_(np.squeeze(y_out[i] + treerasters.tpy[0]))
#         nc_i_x = np.int_(np.squeeze(x_out[i] + treerasters.tpx[0]))
#         nc_i = label_nc[nc_i_y, nc_i_x]
#         # Only look at the regional group of the currently moving tree
#         nc_i_r = np.where(nc_label == nc_i)
#
#         nc_sum = np.sum(tp_nc[nc_i_r,0])    # Calculate the sum stuck trees in rg nc_i (should be all if adjustment should be made)
#
#         if (nc_i_r[0].__len__() > 1) & (nc_sum == nc_i_r[0].__len__()):
#             y_dir = np.array([1, -1, 0, 0])  # New temp y-position
#             x_dir = np.array([0, 0, 1, -1])  # New temp x-position
#
#             dir_bool = np.ones((y_dir.shape[0]))
#             # Check if it is possible to move in all directions, i.e. are these positions possible for trees
#             for iy in range(y_dir.shape[0]):
#                 y_dir_temp = np.squeeze(y_out[nc_i_r]) + y_dir[iy]
#                 x_dir_temp = np.squeeze(x_out[nc_i_r]) + x_dir[iy]
#                 for iy1 in range(y_dir_temp.shape[0]):
#                     if not np.any(positions.pos[(positions.pos[:, 1] == x_dir_temp[iy1]) & (positions.pos[:, 2] == y_dir_temp[iy1]), 0]):
#                         dir_bool[iy] = 0
#
#             dir_bool = np.bool_(dir_bool)
#             y_dir = y_dir[dir_bool] # Remove positions where it is not possible to put a tree
#             x_dir = x_dir[dir_bool] # Remove positions where it is not possible to put a tree
#
#             tmrt_nc = np.zeros((y_dir.shape[0], 3))  # New tmrt
#             if np.any(y_dir.shape[0]):  # Only go in to if if there are any possible locations for trees
#                 for iy in range(y_dir.shape[0]):  # Loop for shadows for new positions
#                     nc_y_c[nc_i_r] = np.squeeze(y_out[nc_i_r]) + y_dir[iy]  # New y-position
#                     nc_x_c[nc_i_r] = np.squeeze(x_out[nc_i_r]) + x_dir[iy]  # New x-position
#                     #tsh_rg_nc_t, tsh_bool_nc_t = tsh_gen(nc_y_c, nc_x_c, treerasters, treeinput)    # Shadows and rg for new position
#                     tsh_bool_nc_t, tsh_bool_nc_t_large, comp_ = tsh_gen_ts(nc_y_c, nc_x_c, treerasters,
#                                                          treeinput)  # Shadows and rg for new position
#                     # Calculate Tmrt for intersecting shadows
#                     for j in range(tmrt_1d.shape[0]):
#                         tsh_bool_nc_t[:, :, j] = tsh_bool_nc_t[:,:,j] * treeinput.shadows_pad[:,:,j] * treeinput.buildings_pad
#                         tmrt_nc[iy, 0] += np.sum(tsh_bool_nc_t[:,:,j] * tmrt_1d[j, 0])  # Tree shade
#                         tmrt_nc[iy, 1] += np.sum(tsh_bool_nc_t[:,:,j] * treeinput.tmrt_ts_pad[:, :, j])  # Sunlit
#                     tmrt_nc[iy, 2] = tmrt_nc[iy, 1] - tmrt_nc[iy, 0]    # New Tmrt (sunlit - tree shade)
#                 if np.around(np.max(tmrt_nc[:, 2]), decimals=1) > np.around(i_tmrt[counter], decimals=1):   # If any new Tmrt decrease is higher, continue
#                     nc_max_r = np.where(tmrt_nc[:, 2] == np.max(tmrt_nc[:, 2])) # Where is new Tmrt decrease highest
#                     if nc_max_r.__len__() > 1:      # If more than one positions has max tmrt, choose a random of the positions
#                         nc_max_r = np.random.choice(nc_max_r[0], 1)
#                     nc_y_out = np.squeeze(y_out[nc_i_r]) + y_dir[nc_max_r]  # New y-positions
#                     nc_x_out = np.squeeze(x_out[nc_i_r]) + x_dir[nc_max_r]  # New x-positions
#                     nc_tmrt_out = np.max(tmrt_nc[nc_max_r, 2]) # New Tmrt
#                     tp_nc[nc_i_r,0] = 0 # Reset tp_nc, i.e. trees have moved
#                     tp_nc_a[nc_i_r] = 0 # Reset tp_nc_a, i.e. trees have been adjusted
#                     t1[0, 2] = nc_tmrt_out  # New Tmrt
#                     y_out[nc_i_r,0] = nc_y_out  # New y-positions
#                     x_out[nc_i_r,0] = nc_x_out  # New x-positions
#                     tree_pos_y = y_out
#                     tree_pos_x = x_out
#                     tree_adjusted = 1
# if tree_adjusted == 0:
#     tp_nc_a[i] = 1

# # Adjusting tree with lowest Tmrt
# #start_time = time.time()
# a_counter = 0   # Adjustment counter
# while a_counter < trees:
#     tmrt_temp = np.zeros((trees))
#     for ix in range(trees):
#         tmrt_temp[ix] = treerasters.d_tmrt[np.int(i_y[counter,ix]),np.int(i_x[counter,ix])]
#
#     y_min = tmrt_temp[:] == np.min(tmrt_temp[:])
#     if y_min.shape[0] > 1:
#         y_min = y_min.cumsum(axis=0).cumsum(axis=0) == 1    # If more than one index with np.min, return only first
#     y_max = tmrt_temp[:] != np.min(tmrt_temp[:])
#     y_te = i_y[counter,:]
#     x_te = i_x[counter,:]
#     y_high = np.int_(y_te[y_max])
#     x_high = np.int_(x_te[y_max])
#     y_low = np.int_(y_te[y_min])
#     x_low = np.int_(x_te[y_min])
#     tsh_rg_temp, tsh_bool_temp, _, __, ___ = tsh_gen(y_high, x_high, treerasters, treeinput)
#     tsh_rg_min, tsh_bool_min, _, __, ___ = tsh_gen(y_low, x_low, treerasters, treeinput)
#     d_tmrt_pad = np.pad(treerasters.d_tmrt, pad_width=((treerasters.tpy[0], treerasters.tpy[0]), (treerasters.tpx[0], treerasters.tpx[0])), mode='constant',
#                               constant_values=0)
#
#     tsh_bool_all = np.zeros((treeinput.rows_pad,treeinput.cols_pad))
#     for ix in range(tsh_bool_temp.shape[2]):
#         tsh_bool_all[:,:] = tsh_bool_all[:,:] + tsh_bool_temp[:,:,ix]
#
#     tsh_bool_all = tsh_bool_all == 0
#
#     d_tmrt_pad = d_tmrt_pad * tsh_bool_all
#
#     tsh_bool_all = (1 - tsh_bool_all)
#
#     d_tmrt_vec = d_tmrt_pad.flatten()
#
#     d_tmrt_vec_s = -np.sort(-d_tmrt_vec)
#
#     d_bool = d_tmrt_vec_s > np.min(tmrt_temp[:])
#     d_tmrt_vec_s = d_tmrt_vec_s[d_bool]
#
#     d_tmrt_vec_s = np.unique(d_tmrt_vec_s)
#
#     #tmrt_adjust = np.max(d_tmrt_pad)  # Maximum temperature of another position
#     #if tmrt_adjust > np.min(tmrt_temp[:]):  # If tmrt_adjust is larger than the tree with lowest d_tmrt, proceed
#     a_nc = 0
#     for ix in range(d_tmrt_vec_s.shape[0]):
#         tmrt_adjust = d_tmrt_vec_s[ix]
#         y_adjust, x_adjust = np.where(d_tmrt_pad == tmrt_adjust)    # Find coordinates for tmrt_adjust
#         for iy in range(y_adjust.shape[0]):     # If there are more than one position with tmrt_adjust
#             tsh_bool_temp2 = tsh_bool_all * tsh_bool_min[:, :, 0]  # See if old trees overlap, i.e. the tree that is to be adjusted overlaps with the other trees
#             if np.sum(tsh_bool_temp2) == 0:  # If not, proceed. If they overlap, adjustment for overlap needs to be made, etc.
#                 y_temp = np.array([y_adjust[iy] - treerasters.tpy[0]])  # y-Position of tmrt_adjust
#                 x_temp = np.array([x_adjust[iy] - treerasters.tpx[0]])  # x-Position of tmrt_adjust
#                 tsh_rg_adjust, tsh_bool_adjust, _, __, ___ = tsh_gen(y_temp, x_temp, treerasters, treeinput)  # Create boolean shadow from y_temp, x_temp
#                 if np.sum(tsh_bool_all * tsh_bool_adjust[:, :, 0]) == 0:  # If the shadow of the new position does not overlap with the the other trees, proceed
#                     y_te[y_min] = y_temp
#                     x_te[y_min] = x_temp
#                     i_y[counter, :] = y_te
#                     i_x[counter, :] = x_te
#                     i_tmrt[counter] += (tmrt_adjust - np.min(tmrt_temp))
#                     a_nc = 1
#                     print('Adjusted')
#                     break
#                 else:
#                     break
#             else:
#                 a_counter = trees
#                 a_nc = 1
#                 break
#         if a_nc == 1:
#             break
#     a_counter += 1