import numpy as np

def tsh_gen(y,x,tpy,tpx,treeshadow,buildings,treesh_sum):
    # if (y == 10) & (x == 84):
    #     test = 0

    ts_rows = treeshadow.shape[0]   # Which rows have treeshadow (tree in middle of empty matrix)
    ts_cols = treeshadow.shape[1]   # Which cols have treeshadow (tree in middle of empty matrix)

    rows = buildings.shape[0]   # number of rows in study area
    cols = buildings.shape[1]   # numer of cols in study area

    tsh_pos = np.zeros((rows, cols, y.__len__()))
    tsh_pos_bool = np.zeros((rows, cols, y.__len__()))

    for i in range(y.__len__()):
        tpy1 = ts_rows - tpy[0]
        tpx1 = ts_cols - tpx[0]

        tpy2 = rows - y[i]  # idy = 90 -> 101 - 90 = 11
        tpx2 = cols - x[i]

        rows_temp = np.zeros((1, 2))    # empty temp file to calculate how much of the tree shadow that will fit in study area
        rows_temp1 = np.zeros((1, 2))   # empty temp file to calculate position of tree shadow in study area
        cols_temp = np.zeros((1, 2))    # empty temp file to calculate how much of the tree shadow that will fit in study area
        cols_temp1 = np.zeros((1, 2))   # empty temp file to calculate position of tree shadow in study area

        # Finding how much of the treeshadow that will fit in study area depending on tree position
        # and finding where the treeshadow fits into the study area. This is for rows.
        if (y[i] < tpy[0]) & (tpy2 > tpy1):
            rows_temp = [int(abs(y[i] - tpy[0])), int(ts_rows)]
            rows_temp1 = [0, int(y[i] + tpy1)]
        elif (y[i] < tpy[0]) & (tpy2 <= tpy1):
            rows_temp = [int(abs(y[i] - tpy[0])), int(ts_rows - (tpy1 - tpy2))]
            rows_temp1 = [int(rows - tpy2), rows]
        elif (y[i] >= tpy[0]) & (tpy2 > tpy1):
            rows_temp = [0, int(ts_rows)]
            rows_temp1 = [int(y[i] - tpy[0]), int(y[i] + tpy1)]
        elif (y[i] >= tpy[0]) & (tpy2 <= tpy1):
            rows_temp = [0, int(ts_rows - (tpy1 - tpy2))]
            rows_temp1 = [int(y[i] - tpy[0]), rows]

        # Finding how much of the treeshadow that will fit in study area depending on tree position
        # and finding where the treeshadow fits into the study area. This is for cols.
        if (x[i] < tpx[0]) & (tpx2 > tpx1):
            cols_temp = [int(abs(x[i] - tpx[0])), int(ts_cols)]
            cols_temp1 = [0, int(x[i] + tpx1)]
        elif (x[i] < tpx[0]) & (tpx2 <= tpx1):
            cols_temp = [int(abs(x[i] - tpx[0])), int(ts_cols - (tpx1 - tpx2))]
            cols_temp1 = [int(cols - tpx2), cols]
        elif (x[i] >= tpx[0]) & (tpx2 > tpx1):
            cols_temp = [0, int(ts_cols)]
            cols_temp1 = [int(x[i] - tpx[0]), int(x[i] + tpx1)]
        elif (x[i] >= tpx[0]) & (tpx2 <= tpx1):
            cols_temp = [0, int(ts_cols - (tpx1 - tpx2))]
            cols_temp1 = [int(x[i] - tpx[0]), cols]

        # Copying the treeshadow depending on how much of it will fit into the study area.
        a = treesh_sum[rows_temp[0]:rows_temp[1], cols_temp[0]:cols_temp[1]]
        # Pasting into new position in study area
        tsh_pos[rows_temp1[0]:rows_temp1[1], cols_temp1[0]:cols_temp1[1], i] = a[:, :]

        tsh_pos[:,:,i] = tsh_pos[:,:,i] * buildings

        tsh_pos_bool[:,:,i] = tsh_pos[:,:,i] > 0

    return tsh_pos, tsh_pos_bool