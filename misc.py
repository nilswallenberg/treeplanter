import numpy as np

def max_tmrt(y,x,tmrt,trees,pos):
# Functioning returning maximum decrease in Tmrt, i.e. optimal position for trees and their y,x positions and corresponding Tmrt decrease
    r_tmrt_runs = np.hstack((y, x))
    r_tmrt_runs = np.column_stack((r_tmrt_runs,tmrt))
    # How many unique rows are there, and how many replicates of each unique row are there
    unique_tmrt = np.unique(r_tmrt_runs, axis=0)
    unique_tmrt = unique_tmrt[unique_tmrt[:,-1] != 0,:]
    u_tmrt = np.zeros((unique_tmrt.shape[0], trees + 1))

    for idx in range(unique_tmrt.shape[0]):
        for idy in range(trees):
            # print(pos[(pos[:, 1] == unique_tmrt[idx, idy + trees]) & (
            #            pos[:, 2] == unique_tmrt[idx, idy]), 0])
            # print(unique_tmrt[idx, idy + trees])
            # print(unique_tmrt[idx, idy])
            u_tmrt[idx, idy] = pos[(pos[:, 1] == unique_tmrt[idx, idy + trees]) & (pos[:, 2] == unique_tmrt[idx, idy]), 0]
        u_tmrt[idx, -1] = np.sum(unique_tmrt[idx, -1])

    u_sorted = np.sort(u_tmrt[:, :-1])
    u_new = np.concatenate([u_sorted, u_tmrt[:, -1:]], axis=1)

    unique_tmrt2 = np.unique(u_new, axis=0)

    t_u_max = np.where(np.around(unique_tmrt2[:, trees], decimals=1) == np.max(np.around(unique_tmrt2[:, trees], decimals=1)))

    return np.around(unique_tmrt2, decimals=1), t_u_max