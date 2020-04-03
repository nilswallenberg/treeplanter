import numpy as np

def max_tmrt(y,x,tmrt,trees,pos):
# Functioning returning maximum decrease in Tmrt, i.e. optimal position for trees and their y,x positions and corresponding Tmrt decrease
    r_tmrt_runs = np.hstack((y, x, tmrt))

    # How many unique rows are there, and how many replicates of each unique row are there
    unique_tmrt = np.unique(r_tmrt_runs, axis=0)

    u_tmrt = np.zeros((unique_tmrt.shape[0], trees + 1))

    for idx in range(unique_tmrt.shape[0]):
        for idy in range(trees):
            u_tmrt[idx, idy] = pos[(pos[:, 1] == unique_tmrt[idx, idy + trees]) & (
                        pos[:, 2] == unique_tmrt[idx, idy]), 0]
        u_tmrt[idx, -1] = np.sum(unique_tmrt[idx, trees:])

    unique_tmrt1 = np.unique(u_tmrt, axis=0)

    u_sorted = np.sort(unique_tmrt1[:, :-1])
    u_new = np.concatenate([u_sorted, unique_tmrt1[:, -1:]], axis=1)

    unique_tmrt2 = np.unique(u_new, axis=0)

    t_u_max = np.where(unique_tmrt2[:, trees] == np.max(unique_tmrt2[:, trees]))

    return unique_tmrt2, t_u_max