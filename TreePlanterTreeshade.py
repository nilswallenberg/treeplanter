import numpy as np

# This function returns a raster with a boolean shadow and the regional group shadow for the tree in position y,x
def tsh_gen(y,x,treerasters,buildings):

    ts_rows = treerasters.treeshade.shape[0]   # Which rows have treeshadow (tree in middle of empty matrix)
    ts_cols = treerasters.treeshade.shape[1]   # Which cols have treeshadow (tree in middle of empty matrix)

    tsh_pos = np.zeros((treerasters.rows, treerasters.cols, y.__len__()))
    tsh_pos_bool = np.zeros((treerasters.rows, treerasters.cols, y.__len__()))

    for i in range(y.__len__()):
        tpy1 = ts_rows - treerasters.tpy[0]
        tpx1 = ts_cols - treerasters.tpx[0]

        tpy2 = treerasters.rows - y[i]  # idy = 90 -> 101 - 90 = 11
        tpx2 = treerasters.cols - x[i]

        rows_temp = np.zeros((1, 2))    # empty temp file to calculate how much of the tree shadow that will fit in study area
        rows_temp1 = np.zeros((1, 2))   # empty temp file to calculate position of tree shadow in study area
        cols_temp = np.zeros((1, 2))    # empty temp file to calculate how much of the tree shadow that will fit in study area
        cols_temp1 = np.zeros((1, 2))   # empty temp file to calculate position of tree shadow in study area

        # Finding how much of the treeshadow that will fit in study area depending on tree position
        # and finding where the treeshadow fits into the study area. This is for rows.
        if (y[i] < treerasters.tpy[0]) & (tpy2 > tpy1):
            rows_temp = [int(abs(y[i] - treerasters.tpy[0])), int(ts_rows)]
            rows_temp1 = [0, int(y[i] + tpy1)]
        elif (y[i] < treerasters.tpy[0]) & (tpy2 <= tpy1):
            rows_temp = [int(abs(y[i] - treerasters.tpy[0])), int(ts_rows - (tpy1 - tpy2))]
            rows_temp1 = [int(treerasters.rows - tpy2), treerasters.rows]
        elif (y[i] >= treerasters.tpy[0]) & (tpy2 > tpy1):
            rows_temp = [0, int(ts_rows)]
            rows_temp1 = [int(y[i] - treerasters.tpy[0]), int(y[i] + tpy1)]
        elif (y[i] >= treerasters.tpy[0]) & (tpy2 <= tpy1):
            rows_temp = [0, int(ts_rows - (tpy1 - tpy2))]
            rows_temp1 = [int(y[i] - treerasters.tpy[0]), treerasters.rows]

        # Finding how much of the treeshadow that will fit in study area depending on tree position
        # and finding where the treeshadow fits into the study area. This is for cols.
        if (x[i] < treerasters.tpx[0]) & (tpx2 > tpx1):
            cols_temp = [int(abs(x[i] - treerasters.tpx[0])), int(ts_cols)]
            cols_temp1 = [0, int(x[i] + tpx1)]
        elif (x[i] < treerasters.tpx[0]) & (tpx2 <= tpx1):
            cols_temp = [int(abs(x[i] - treerasters.tpx[0])), int(ts_cols - (tpx1 - tpx2))]
            cols_temp1 = [int(treerasters.cols - tpx2), treerasters.cols]
        elif (x[i] >= treerasters.tpx[0]) & (tpx2 > tpx1):
            cols_temp = [0, int(ts_cols)]
            cols_temp1 = [int(x[i] - treerasters.tpx[0]), int(x[i] + tpx1)]
        elif (x[i] >= treerasters.tpx[0]) & (tpx2 <= tpx1):
            cols_temp = [0, int(ts_cols - (tpx1 - tpx2))]
            cols_temp1 = [int(x[i] - treerasters.tpx[0]), treerasters.cols]

        # Copying the treeshadow depending on how much of it will fit into the study area.
        a = treerasters.treeshade_rg[rows_temp[0]:rows_temp[1], cols_temp[0]:cols_temp[1]]
        # Pasting into new position in study area
        tsh_pos[rows_temp1[0]:rows_temp1[1], cols_temp1[0]:cols_temp1[1], i] = a[:, :]

        tsh_pos[:,:,i] = tsh_pos[:,:,i] * buildings

        tsh_pos_bool[:,:,i] = tsh_pos[:,:,i] > 0

    return tsh_pos, tsh_pos_bool