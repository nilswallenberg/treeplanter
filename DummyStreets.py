import numpy as np
from osgeo import gdal
from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt

from SOLWEIG.misc import saveraster

# Output
output = 'C:/Users/xwanil/Desktop/Project_4/Dummy/'

# East-West Street Canyon
#filepath = 'C:/Users/xwanil/Desktop/Project_4/Dummy/dummy_300x30.tif'
filepath = 'C:/Users/xwanil/Desktop/Project_4/Dummy/dummy_30x300.tif'
dataSet = gdal.Open(filepath)
square_dummy = dataSet.ReadAsArray().astype(np.float)
rows = square_dummy.shape[0]
cols = square_dummy.shape[1]

# DEM
dem_nw = np.zeros((rows,cols))
dem_nw[:,:] = 1.5

saveraster(dataSet, output + 'dem_nw.tif', dem_nw)        # East-West street canyon

# East - West street with H/W 1 (40 meters wide, 40 meter high buildings)
dsm_nw = np.zeros((rows,cols))
dsm_nw[:,0:5] = 10
dsm_nw[:,25:] = 10
dsm_nw = dsm_nw + dem_nw

saveraster(dataSet, output + 'dsm_nw_hw05.tif', dsm_nw)        # East-West street canyon

dsm_nw[:,0:5] += 10
dsm_nw[:,25:] += 10

saveraster(dataSet, output + 'dsm_nw_hw1.tif', dsm_nw)        # East-West street canyon

dsm_nw[:,0:5] += 20
dsm_nw[:,25:] += 20

saveraster(dataSet, output + 'dsm_nw_hw2.tif', dsm_nw)        # East-West street canyon

test = 0

# East-West Street Canyon
#filepath = 'C:/Users/xwanil/Desktop/Project_4/Dummy/dummy_300x30.tif'
filepath = 'C:/Users/xwanil/Desktop/Project_4/Dummy/dummy_diag.tif'
dataSet = gdal.Open(filepath)
square_dummy = dataSet.ReadAsArray().astype(np.float)
rows = square_dummy.shape[0]
cols = square_dummy.shape[1]

# DEM
dem_nesw = np.zeros((rows,cols))
dem_nesw[:,:] = 1.5

# East - West street with H/W 1 (40 meters wide, 40 meter high buildings)
dsm_nesw = np.zeros((rows,cols))
dsm_nesw[:,0:141] = 10
dsm_nesw[:,160:] = 10
dsm_nesw = dsm_nesw + dem_nesw

nesw_temp = rotate(dsm_nesw, angle=45)
nesw_temp = nesw_temp[115:-115,115:-115]
nesw_temp[nesw_temp == 0] = np.max(dsm_nesw)
nesw_temp[nesw_temp < 5] = 1.5
nesw_temp[nesw_temp > 5] = np.max(dsm_nesw)

filepath = 'C:/Users/xwanil/Desktop/Project_4/Dummy/dummy_194x194.tif'
dataSetDiag = gdal.Open(filepath)

saveraster(dataSetDiag, output + 'dsm_nesw_hw05.tif', nesw_temp)        # East-West street canyon
nesw_temp[nesw_temp > 5] += 10
saveraster(dataSetDiag, output + 'dsm_nesw_hw1.tif', nesw_temp)        # East-West street canyon
nesw_temp[nesw_temp > 5] += 20
saveraster(dataSetDiag, output + 'dsm_nesw_hw2.tif', nesw_temp)        # East-West street canyon
dem_nesw = dem_nesw[:194,:194]
saveraster(dataSetDiag, output + 'dem_nesw.tif', dem_nesw)        # East-West street canyon

nwse_temp = rotate(dsm_nesw, angle=135)
nwse_temp = nwse_temp[115:-115,115:-115]
nwse_temp[nwse_temp == 0] = np.max(dsm_nesw)
nwse_temp[nwse_temp < 5] = 1.5
nwse_temp[nwse_temp > 5] = np.max(dsm_nesw)

filepath = 'C:/Users/xwanil/Desktop/Project_4/Dummy/dummy_194x194.tif'
dataSetDiag = gdal.Open(filepath)

saveraster(dataSetDiag, output + 'dsm_nwse_hw05.tif', nwse_temp)        # East-West street canyon
nwse_temp[nwse_temp > 5] += 10
saveraster(dataSetDiag, output + 'dsm_nwse_hw1.tif', nwse_temp)        # East-West street canyon
nwse_temp[nwse_temp > 5] += 20
saveraster(dataSetDiag, output + 'dsm_nwse_hw2.tif', nwse_temp)        # East-West street canyon
dem_nwse = dem_nesw[:194,:194]
saveraster(dataSetDiag, output + 'dem_nwse.tif', dem_nwse)        # East-West street canyon

test = 0

# # North - South street with H/W 1 (40 meters wide, 40 meter high buildings)
# north_south = np.zeros((rows,cols))
# north_south[:,0:30] = 40
# north_south[:,70:] = 40
# north_south = north_south + dem
#
# # SW-NE street with H/W 1 (40 meters wide, 40 meter high buildings)
# se_nw_temp = rotate(east_west, angle=45)
# se_nw = np.zeros((rows,cols))
# se_nw[:,:] = se_nw_temp[20:120,20:120]
# se_nw[0:35,0:35] = 40
# se_nw[70:,70:] = 40
# se_nw[se_nw > 20] = 40
# se_nw[se_nw < 20] = 0
# se_nw = se_nw + dem
#
# # SE-NW street with H/W 1 (40 meters wide, 40 meter high buildings)
# ne_sw_temp = rotate(east_west, angle=135)
# ne_sw = np.zeros((rows,cols))
# ne_sw[:,:] = ne_sw_temp[20:120,20:120]
# ne_sw[0:35,70:] = 40
# ne_sw[70:,0:35] = 40
# ne_sw[ne_sw > 20] = 40
# ne_sw[ne_sw < 20] = 0
# ne_sw = ne_sw + dem
#
# #fig = plt.figure(figsize=(14, 7))
# fig = plt.figure()
# ax1 = plt.subplot(2, 2, 1)
# im1 = ax1.imshow(north_south, clim=[0, 40])
# plt.title('North-South')
# plt.colorbar(im1)
# ax2 = plt.subplot(2, 2, 2)
# im2 = ax2.imshow(east_west, clim=[0, 40])
# plt.title('East-West')
# plt.colorbar(im2)
# ax3 = plt.subplot(2, 2, 3)
# im3 = ax3.imshow(se_nw, clim=[0, 40])
# plt.title('SouthWest-NorthEast')
# plt.colorbar(im3)
# ax4 = plt.subplot(2, 2, 4)
# im4 = ax4.imshow(ne_sw, clim=[0, 40])
# plt.title('NorthWest-SouthEast')
# plt.colorbar(im4)
# plt.subplots_adjust(top=0.95, bottom=0.05, left=0.03, right=0.97, hspace=0.35,
#                     wspace=0.15)
# #plt.tight_layout()
# #plt.show()
#
# f_name = 'C:/Users/xwanil/Desktop/Project_4/Dummy/' + 'dummy_streets.tif'
# plt.savefig(f_name, dpi=150)
# plt.clf()
#
# # Saving rasters
# # saveraster(dataSet, output + 'north_south.tif', north_south)    # North-South street canyon
# # saveraster(dataSet, output + 'sw_ne.tif', ne_sw)    # SW-NE street canyon
# # saveraster(dataSet, output + 'se_nw.tif', se_nw)    # SE-NW street canyon
# # saveraster(dataSet, output + 'dem.tif', dem)    # SE-NW street canyon