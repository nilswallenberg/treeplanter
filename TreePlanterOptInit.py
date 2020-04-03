import numpy as np
import itertools
import time
import TreePlanterOptimizer

def combine(tup, t):
    return tuple(itertools.combinations(tup, t))

def treeoptinit(treerasters, positions, dia, buildings, shadow_rg, tmrt_1d, trees, r_iters, k_s, i_rand):

    counter = 0  # Counter for randomization runs

    i_tmrt = np.zeros((r_iters, trees)) # Empty vector to be filled with Tmrt values for each tree
    i_y = np.zeros((r_iters, trees))    # Empty vector to be filled with corresponding y position of the above
    i_x = np.zeros((r_iters, trees))    # Empty vector to be filled with corresponding x position of the above

    tree_pos_all = np.zeros((r_iters,trees))

    #d_count = 0

    # Iterate for r_iters number of iterations. Will find optimal positions for trees and return the positions and tmrt
    while counter < r_iters:

        new_tree_pos = np.zeros((1, trees))
        ils_c = np.zeros((trees))

        r_count = 0
        while r_count < 1:
            # Creating trees at random positions
            # tree_pos = random.choices(pos_ls_list, k=trees)

            # i_rand = 0 - random starting positions
            # i_rand = 1 - random starting positions. Next iteration will start from a new random position within the kernel k_s.
            # Will start from full random if same local maximum is found 5 times.
            if i_rand == 0:
                tree_pos = np.random.randint(positions.pos.shape[0], size=trees)  # Random positions for trees
            elif i_rand == 1:
                if counter == 0:
                    tree_pos = np.random.randint(positions.pos.shape[0], size=trees)  # Random positions for trees
                else:
                    tree_pos = np.zeros((trees))
                    k_r = np.int_((k_s - 1) / 2)  # Kernel radian
                    pos_m_pad = np.pad(positions.pos_m, pad_width=k_r, mode='constant', constant_values=0)
                    for i_t in range(tree_pos_y.shape[0]):
                        if ils_c[i_t] < 5:
                            y_t = tree_pos_y[i_t, 0]+k_r
                            x_t = tree_pos_x[i_t, 0]+k_r
                            u_p = np.unique(pos_m_pad[y_t - k_r:y_t + k_r + 1, x_t - k_r:x_t + k_r + 1])
                            u_p = u_p[u_p != 0.]
                            tree_pos[i_t] = np.random.choice(u_p, 1)
                        else:
                            print('Restart random')
                            tree_pos[i_t] = np.random.randint(treerasters.pos.shape[0], size=1)
                            ils_c[i_t] = 0

            # Looks for duplicate starting positions
            # for x1 in tree_pos_all:
            #     if all([np.allclose(x1, np.sort(tree_pos))]) == True:
            #         print('Duplicate ' + str(d_count))
            #         d_count += 1
            #         break

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

            tree_pos_all[counter] = np.sort(tree_pos)

        tp_nc = np.zeros((trees, 1))  # 1 if tree's are in the same position as in previous iteration, otherwise 0

        ti = itertools.cycle(range(trees)) # Iterator to move between trees moving around in the study area

        # Moving trees, i.e. optimization
        while np.sum(tp_nc[:,0]) < trees:

            i = ti.__next__()

            # Break if trees can't move, i.e. no better position can be found in optimizer
            if np.sum(tp_nc[:, 0] == trees):
                break

            # Running optimizer
            # t1 = best shading position
            t1, nc = TreePlanterOptimizer.topt(tree_pos_y, tree_pos_x, treerasters, dia, buildings, shadow_rg, tmrt_1d, i)
            tp_nc[i, 0] = nc

            # Changing position of tree
            if t1[0, 2] > new_tree_pos[0, i]:
                new_tree_pos[0, i] = t1[0, 2]
                tree_pos_y[i, 0] = t1[0, 3]
                tree_pos_x[i, 0] = t1[0, 4]

                i_tmrt[counter, i] = new_tree_pos[0, i]
                i_x[counter, i] = tree_pos_x[i, 0]
                i_y[counter, i] = tree_pos_y[i, 0]

            else:
                ils_c[i] += 1

        counter += 1

    # Finding best position from all r_iters iteration, i.e. if r_iters = 1000 then best position out of 1000 runs
    tmrt_max = np.zeros((i_tmrt.shape[0], 1))
    for i in range(i_tmrt.shape[0]):
        tmrt_max[i, 0] = np.sum(i_tmrt[i, :])
    t_max = np.max(tmrt_max[:,0])
    y = np.where(tmrt_max[:, 0] == t_max)

    # Returns all the unique positions found by the algorithm and the potential decrease in tmrt
    from misc import max_tmrt
    unique_tmrt, unique_tmrt_max = max_tmrt(i_y, i_x, i_tmrt, trees, positions.pos)

    # Optimal positions of trees
    i_y = i_y[y[0][0], :]
    i_x = i_x[y[0][0], :]

    return i_y, i_x, unique_tmrt, unique_tmrt_max