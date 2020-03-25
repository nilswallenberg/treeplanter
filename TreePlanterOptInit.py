import numpy as np
import itertools
import time
import TreePlanterOptimizer
from TreePlanterTreeshade import tsh_gen
from scipy.signal import argrelextrema

def combine(tup, t):
    return tuple(itertools.combinations(tup, t))

def treeoptinit(pos_ls,sum_tmrt_tsh,sum_tmrt,sun_vs_tsh,dia,treeshadow_clip,tpy,tpx,sh_tmrt,buildings,treesh_sum_clip,sh_vec_t,trees,r_iters,pos_m,k_s,i_rand):

    rows,cols = buildings.shape

    ind_y, ind_x = argrelextrema(sun_vs_tsh, np.greater)

    loc_max = np.zeros((ind_y.shape[0], 1))
    for i in range(ind_y.shape[0]):
        loc_max[i, 0] = sun_vs_tsh[ind_y[i], ind_x[i]]

    sort_ind = np.argsort(-loc_max[:, 0])
    ind_y_s = ind_y[sort_ind]
    ind_x_s = ind_x[sort_ind]

    ind_y_s = ind_y_s[0:trees]
    ind_x_s = ind_x_s[0:trees]

    t_max_sh, t_max_sh_bool = tsh_gen(ind_y_s, ind_x_s, tpy, tpx, treeshadow_clip, buildings, treesh_sum_clip)

    t_max_overlap = np.zeros((rows,cols))
    for i in range(t_max_sh.shape[2]):
        t_max_overlap = t_max_overlap + t_max_sh_bool[:,:,i]

    sh_bld = buildings

    counter = 0  # Counter for randomization runs

    i_tmrt = np.zeros((r_iters, trees))
    i_y = np.zeros((r_iters, trees))
    i_x = np.zeros((r_iters, trees))

    # pos_ls_list = pos_ls[:,0].tolist()
    tree_pos_all = np.zeros((r_iters,trees))

    d_count = 0

    #i_rand = 0  # Random hill climbing
    #i_rand = 1  # Iterative Local Search
    #k_s = 21 # Kernel size 21x21

    while counter < r_iters:

        new_tree_pos = np.zeros((1, trees))
        ils_c = np.zeros((trees))

        r_count = 0
        while r_count < 1:
            # Creating trees at random positions
            # tree_pos = random.choices(pos_ls_list, k=trees)

            if i_rand == 0:
                tree_pos = np.random.randint(pos_ls.shape[0], size=trees)  # Random positions for trees
            elif i_rand == 1:
                if counter == 0:
                    tree_pos = np.random.randint(pos_ls.shape[0], size=trees)  # Random positions for trees
                else:
                    tree_pos = np.zeros((trees))
                    k_r = np.int_((k_s - 1) / 2)  # Kernel radian
                    pos_m_pad = np.pad(pos_m, pad_width=k_r, mode='constant', constant_values=0)
                    for i_t in range(tree_pos_y.shape[0]):
                        if ils_c[i_t] < 5:
                            y_t = tree_pos_y[i_t, 0]+k_r
                            x_t = tree_pos_x[i_t, 0]+k_r
                            u_p = np.unique(pos_m_pad[y_t - k_r:y_t + k_r + 1, x_t - k_r:x_t + k_r + 1])
                            u_p = u_p[u_p != 0.]
                            tree_pos[i_t] = np.random.choice(u_p, 1)
                        else:
                            print('Restart random')
                            tree_pos[i_t] = np.random.randint(pos_ls.shape[0], size=1)
                            ils_c[i_t] = 0

            for x1 in tree_pos_all:
                if all([np.allclose(x1, np.sort(tree_pos))]) == True:
                    #print('True')
                    print('Duplicate ' + str(d_count))
                    d_count += 1
                    break

            tree_pos_y = np.zeros((trees, 1), dtype=int)  # Random y-positions for trees
            tree_pos_x = np.zeros((trees, 1), dtype=int)  # Random x-positions for trees

            for i2 in range(tree_pos.__len__()):
                tree_pos_x[i2, 0] = pos_ls[pos_ls[:, 0] == tree_pos[i2], 1]
                tree_pos_y[i2, 0] = pos_ls[pos_ls[:, 0] == tree_pos[i2], 2]

            # Euclidean distance between random positions so that trees are not too close to each other
            it_comb = combine(tree_pos, 2)
            eucl_dist = np.zeros((it_comb.__len__(), 1))

            for i3 in range(it_comb.__len__()):
                ax = pos_ls[pos_ls[:, 0] == it_comb[i3][0], 1]
                ay = pos_ls[pos_ls[:, 0] == it_comb[i3][0], 2]
                a = np.array(([ay[0], ax[0]]))
                bx = pos_ls[pos_ls[:, 0] == it_comb[i3][1], 1]
                by = pos_ls[pos_ls[:, 0] == it_comb[i3][1], 2]
                b = np.array(([by[0], bx[0]]))
                eucl_dist[i3, 0] = np.linalg.norm(a - b)

            if np.min(eucl_dist[:, 0]) >= dia:
                r_count = 1
                # Lägg till alla positioner för eucl >= dia

            tree_pos_all[counter] = np.sort(tree_pos)

        tp_nc = np.zeros((trees, 1))  # 1 if tree's are in the same position as in previous iteration, otherwise 0

        ti = itertools.cycle(range(trees))

        # if counter < 25:
        #     m_c = 0

        # Moving trees, i.e. optimization
        while np.sum(tp_nc[:,0]) < trees:

            i = ti.__next__()
            # while tp_nc[i, 0] == 1:
            #     i = ti.__next__()
            #     if np.sum(tp_nc[:, 0] == trees):
            #         break
            if np.sum(tp_nc[:, 0] == trees):
                break

            # t1 = best shading position
            t1, nc = TreePlanterOptimizer.topt(tree_pos_y,tree_pos_x,sum_tmrt_tsh,sum_tmrt,sun_vs_tsh,i,dia,treeshadow_clip,sh_bld,
                              tpy,tpx,sh_tmrt,buildings,treesh_sum_clip,sh_vec_t)

            tp_nc[i, 0] = nc

            # Changing position of tree
            if t1[0, 2] > new_tree_pos[0, i]:
                new_tree_pos[0, i] = t1[0, 2]
                tree_pos_y[i, 0] = t1[0, 3]
                tree_pos_x[i, 0] = t1[0, 4]

                i_tmrt[counter, i] = new_tree_pos[0, i]
                i_x[counter, i] = tree_pos_x[i, 0]
                i_y[counter, i] = tree_pos_y[i, 0]

                # if counter < 25:
                #     cdsm_e = np.zeros((rows, cols))  # Empty cdsm
                #     tdsm_e = np.zeros((rows, cols))  # Empty tdsm
                #     cdsm_m = np.zeros((rows,cols))
                #     cdsm_t = np.zeros((rows, cols))
                #     fm_str = 'step_' + str(m_c) + '.tif'
                #     for ik in range(trees):
                #         cdsm_t, tdsm_m = makevegdems.vegunitsgeneration(bld_orig, cdsm_e, tdsm_e, ttype,
                #                                                               height, trunk, dia, tree_pos_y[ik,0],
                #                                                               tree_pos_x[ik,0], cols, rows, scale)
                #
                #         cdsm_m = cdsm_m + cdsm_t
                #
                #     c_out = str(trees) + '_trees' + '/' 'run_' + str(l_r) + '/' + 'random_' + str(counter) + '/'
                #     output = directory + c_out
                #     if not os.path.exists(output):  # Create folder if not existing
                #         os.makedirs(output)
                #     saveraster(dataSet, output + '/cdsm_' + fm_str, cdsm_m)
                #
                #     m_c += 1
            else:
                ils_c[i] += 1

        counter += 1

    tmrt_max = np.zeros((i_tmrt.shape[0], 1))
    for i in range(i_tmrt.shape[0]):
        tmrt_max[i, 0] = np.sum(i_tmrt[i, :])

    y = np.where(tmrt_max[:, 0] == np.max(tmrt_max[:, 0]))

    r_tmrt_runs = np.hstack((i_y, i_x, i_tmrt))

    unique_tmrt, unique_tmrt_counts = np.unique(r_tmrt_runs, return_counts=True, axis=0)    # How many unique rows are there, and how many replicates of each unique row are there

    u_tmrt = np.zeros((unique_tmrt.shape[0], trees+1))

    for idx in range(unique_tmrt.shape[0]):
        for idy in range(trees):
            u_tmrt[idx, idy] = pos_ls[(pos_ls[:, 1] == unique_tmrt[idx, idy+trees]) & (pos_ls[:, 2] == unique_tmrt[idx, idy]), 0]
        u_tmrt[idx, -1] = np.sum(unique_tmrt[idx, trees:])

    unique_tmrt1, unique_tmrt1_counts = np.unique(u_tmrt, return_counts=True, axis=0)

    u_sorted = np.sort(unique_tmrt1[:,:-1])
    u_new = np.concatenate([u_sorted, unique_tmrt1[:, -1:]], axis=1)

    unique_tmrt2, unique_tmrt2_counts = np.unique(u_new, return_counts=True, axis=0)

    t_u_max = np.where(unique_tmrt2[:,trees] == np.max(unique_tmrt2[:,trees]))

    i_y = i_y[y[0][0], :]
    i_x = i_x[y[0][0], :]

    return i_y, i_x, unique_tmrt2, unique_tmrt2_counts, t_u_max