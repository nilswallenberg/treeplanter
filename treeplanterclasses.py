from osgeo import gdal, osr
import numpy as np
from scipy.ndimage import label
from scipy import ndimage

class Inputdata():
# Class containing input data for Tree planter.
    #__slots__ = ('dataSet', 'buildings', 'selected_area', 'aoi', 'dem', 'dsm', 'shadow', 'tmrt_ts', 'tmrt_smooth_m', 'tmrt_smooth_mx', 'tmrt_smooth_s', 'tmrt_smooth_sm', 'tmrt_smooth_ms', 'tmrt_smooth_smx', 'tmrt_smooth_g', 'tmrt_smooth_gx', 'tmrt_smooth_mf', 'tmrt_ggm', 'tmrt_s', 'rows', 'cols', 'scale', 'lat', 'lon', 'shadows_pad', 'tmrt_ts_pad', 'buildings_pad', 'rows_pad', 'cols_pad')
    __slots__ = ('dataSet', 'buildings', 'selected_area', 'aoi', 'dem', 'dsm', 'cdsm', 'cdsm_b', 'shadow', 'tmrt_ts', 'tmrt_s', 'rows', 'cols', 'scale', 'lat', 'lon', 'shadows_pad', 'tmrt_ts_pad', 'buildings_pad', 'rows_pad', 'cols_pad')
    def __init__(self,r_range, sh_fl, tmrt_fl, infolder, infolder2, rows, cols, aoi):

        self.dataSet = gdal.Open(infolder2 + 'buildings.tif')            # GIS data
        self.buildings = self.dataSet.ReadAsArray().astype(np.float)    # Building raster
        self.buildings = self.buildings == 1.0
        self.dem = np.zeros((rows,cols))                                # Digital elevation model
        self.dsm = np.zeros((rows,cols))                                # Digital surface model
        self.cdsm = np.zeros((rows,cols))                               # Canopy digital surface model
        self.cdsm_b = np.zeros((rows, cols))  # Canopy digital surface model
        self.shadow = np.zeros((rows,cols, r_range.__len__()))          # Shadow rasters
        self.tmrt_ts = np.zeros((rows, cols, r_range.__len__()))        # Tmrt for each timestep
        #self.tmrt_smooth_m = np.zeros((rows, cols, r_range.__len__()))  # Tmrt for each timestep with mean smoothing
        #self.tmrt_smooth_mx = np.zeros((rows, cols, r_range.__len__()))  # Tmrt for each timestep with mean smoothing loop
        #self.tmrt_smooth_s = np.zeros((rows, cols, r_range.__len__()))  # Tmrt for each timestep with sink smoothing
        #self.tmrt_smooth_sm = np.zeros((rows,cols, r_range.__len__()))  # Tmrt for each timestep with sink smoothing then mean smoothing
        #self.tmrt_smooth_ms = np.zeros((rows, cols, r_range.__len__())) # Tmrt for each timestep with mean smoothing then sink smoothing
        #self.tmrt_smooth_smx = np.zeros((rows, cols, r_range.__len__()))# Tmrt for each timestep with sink x 10 and mean x 10
        #self.tmrt_smooth_g = np.zeros((rows,cols, r_range.__len__()))   # Tmrt with gaussian filter
        #self.tmrt_smooth_gx = np.zeros((rows, cols, r_range.__len__()))  # Tmrt with gaussian filter
        #self.tmrt_smooth_mf = np.zeros((rows,cols, r_range.__len__()))  # Tmrt with median filter
        #self.tmrt_ggm = np.zeros((rows,cols, r_range.__len__()))        # Tmrt Gaussian gradient magnitude filter
        self.tmrt_s = np.zeros((rows, cols))                            # Sum of tmrt for all timesteps
        self.rows = rows    # Rows of rasters
        self.cols = cols    # Cols of rasters
        self.aoi = aoi

        if self.aoi == 1:    # Area of interest
            self.dataSet = gdal.Open(infolder + 'selected_area.tif')
            self.selected_area = self.dataSet.ReadAsArray().astype(np.float)# Selected area
            #self.aoi = 1

        c = 0
        for iy in r_range:
            dataSet1 = gdal.Open(sh_fl[iy])
            self.shadow[:, :, c] = dataSet1.ReadAsArray().astype(np.float)
            dataSet2 = gdal.Open(tmrt_fl[iy])
            self.tmrt_ts[:, :, c] = np.around(dataSet2.ReadAsArray().astype(np.float), decimals=1)
            self.tmrt_s = self.tmrt_s + self.tmrt_ts[:, :, c]

            # # Gaussian gradient magnitude
            # self.tmrt_ggm[:,:,c] = ndimage.gaussian_gradient_magnitude(self.tmrt_ts[:,:,c], sigma=5)
            #
            # # Median filter
            # self.tmrt_smooth_mf[:,:,c] = ndimage.median_filter(self.tmrt_ts[:,:,c], size=3)
            #
            # # Gaussian filter
            # sigma_x = 1.0
            # sigma_y = sigma_x
            #
            # tmrt_temp = np.zeros((rows,cols))
            #
            # sigma = [sigma_y, sigma_x]
            # for ij in range(10):
            #     if ij == 0:
            #         tmrt_temp[:,:] = self.tmrt_ts[:,:,c]
            #     else:
            #         tmrt_temp[:,:] = self.tmrt_smooth_gx[:,:,c]
            #     self.tmrt_smooth_gx[:, :, c] = ndimage.gaussian_filter(tmrt_temp, sigma, mode='constant')
            #
            # self.tmrt_smooth_g[:,:,c] = ndimage.gaussian_filter(self.tmrt_ts[:,:,c], sigma, mode='constant')
            #
            # filterrange = 1.0
            # domain = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
            #
            # # sink loop and mean filter
            #
            # # mean filter
            # for j in np.arange(1,self.rows-1):
            #     for i in np.arange(1,self.cols-1):
            #         dom = self.tmrt_ts[j - 1:j + 2, i - 1:i + 2, c]
            #         if (dom.max() - dom.min()) < filterrange:
            #             self.tmrt_smooth_m[j, i, c] = 1 / 9 * np.sum(dom * domain)
            #         else:
            #             self.tmrt_smooth_m[j, i, c] = self.tmrt_ts[j, i, c]
            #
            # self.tmrt_smooth_mx[:,:,c] = self.tmrt_smooth_m[:,:,c]
            #
            # tmrt_temp = np.zeros((rows,cols))
            #
            # # mean filter
            # for ij in range(10):
            #     if ij == 0:
            #         tmrt_temp[:,:] = self.tmrt_ts[:,:,c]
            #     else:
            #         tmrt_temp[:,:] = self.tmrt_smooth_mx[:,:,c]
            #     for j in np.arange(1,self.rows-1):
            #         for i in np.arange(1,self.cols-1):
            #             dom = tmrt_temp[j - 1:j + 2, i - 1:i + 2]
            #             if (dom.max() - dom.min()) < filterrange:
            #                 self.tmrt_smooth_mx[j, i, c] = 1 / 9 * np.sum(dom * domain)
            #             else:
            #                 self.tmrt_smooth_mx[j, i, c] = tmrt_temp[j, i]
            #
            # # sink filter
            # for j in np.arange(1,self.rows-1):
            #     for i in np.arange(1,self.cols-1):
            #         dom = self.tmrt_ts[j - 1:j + 2, i - 1:i + 2, c]
            #         if self.tmrt_ts[j, i, c] == dom.max():
            #             if (dom.max() - dom.min()) < filterrange:
            #                 self.tmrt_smooth_s[j, i, c] = np.mean(dom)
            #             else:
            #                 self.tmrt_smooth_s[j, i, c] = self.tmrt_ts[j, i, c]
            #         else:
            #             self.tmrt_smooth_s[j, i, c] = self.tmrt_ts[j, i, c]
            #
            # # mean filter
            # for j in np.arange(1,self.rows-1):
            #     for i in np.arange(1,self.cols-1):
            #         dom = self.tmrt_smooth_s[j - 1:j + 2, i - 1:i + 2, c]
            #         if (dom.max() - dom.min()) < filterrange:
            #             self.tmrt_smooth_sm[j, i] = 1 / 9 * np.sum(dom * domain)
            #         else:
            #             self.tmrt_smooth_sm[j, i, c] = self.tmrt_smooth_s[j, i, c]
            #
            # # sink filter
            # for j in np.arange(1,self.rows-1):
            #     for i in np.arange(1,self.cols-1):
            #         dom = self.tmrt_smooth_m[j - 1:j + 2, i - 1:i + 2, c]
            #         if self.tmrt_smooth_m[j, i, c] == dom.max():
            #             if (dom.max() - dom.min()) < filterrange:
            #                 self.tmrt_smooth_ms[j, i, c] = np.mean(dom)
            #             else:
            #                 self.tmrt_smooth_ms[j, i, c] = self.tmrt_smooth_m[j, i, c]
            #         else:
            #             self.tmrt_smooth_ms[j, i, c] = self.tmrt_smooth_m[j, i, c]
            #
            # tmrt_temp = np.zeros((rows,cols))
            #
            # # sink filter
            # for ij in range(10):
            #     if ij == 0:
            #         tmrt_temp[:,:] = self.tmrt_ts[:,:,c]
            #     else:
            #         tmrt_temp[:,:] = self.tmrt_smooth_smx[:,:,c]
            #     for j in np.arange(1,self.rows-1):
            #         for i in np.arange(1,self.cols-1):
            #             dom = tmrt_temp[j - 1:j + 2, i - 1:i + 2]
            #             if tmrt_temp[j, i] < dom.max():
            #                 if (dom.max() - dom.min()) < filterrange:
            #                     self.tmrt_smooth_smx[j, i, c] = np.mean(dom)
            #                 else:
            #                     self.tmrt_smooth_smx[j, i, c] = tmrt_temp[j, i]
            #             else:
            #                 self.tmrt_smooth_smx[j, i, c] = tmrt_temp[j, i]
            #
            #     tmrt_temp = np.zeros((rows,cols))
            #
            # # mean filter
            # for ij in range(1):
            #     tmrt_temp[:,:] = self.tmrt_smooth_smx[:,:,c]
            #     for j in np.arange(1,self.rows-1):
            #         for i in np.arange(1,self.cols-1):
            #             dom = tmrt_temp[j - 1:j + 2, i - 1:i + 2]
            #             if (dom.max() - dom.min()) < filterrange:
            #                 self.tmrt_smooth_smx[j, i, c] = np.mean(dom * domain)
            #             else:
            #                 self.tmrt_smooth_smx[j, i, c] = tmrt_temp[j, i]
            #
            # self.tmrt_smooth_smx[:,:,c] = np.around(self.tmrt_smooth_smx[:,:,c], decimals=1)

            c += 1

        # find latlon etc.
        old_cs = osr.SpatialReference()
        # dsm_ref = dsmlayer.crs().toWkt()
        dsm_ref = self.dataSet.GetProjection()
        old_cs.ImportFromWkt(dsm_ref)

        wgs84_wkt = """
            GEOGCS["WGS 84",
                DATUM["WGS_1984",
                    SPHEROID["WGS 84",6378137,298.257223563,
                        AUTHORITY["EPSG","7030"]],
                    AUTHORITY["EPSG","6326"]],
                PRIMEM["Greenwich",0,
                    AUTHORITY["EPSG","8901"]],
                UNIT["degree",0.01745329251994328,
                    AUTHORITY["EPSG","9122"]],
                AUTHORITY["EPSG","4326"]]"""

        new_cs = osr.SpatialReference()
        new_cs.ImportFromWkt(wgs84_wkt)

        transform = osr.CoordinateTransformation(old_cs, new_cs)

        width1 = self.dataSet.RasterXSize
        height1 = self.dataSet.RasterYSize
        gt = self.dataSet.GetGeoTransform()
        minx = gt[0]
        miny = gt[3] + width1 * gt[4] + height1 * gt[5]
        lonlat = transform.TransformPoint(minx, miny)
        geotransform = self.dataSet.GetGeoTransform()
        self.scale = 1 / geotransform[1]
        self.lon = lonlat[0]
        self.lat = lonlat[1]

    # Padded rasters
    def padding(self, shadows, tmrt, buildings, tpy, tpx):
        self.shadows_pad = np.pad(shadows[:, :, :], pad_width=((tpy, tpy), (tpx, tpx), (0, 0)), mode='constant', constant_values=0)
        self.tmrt_ts_pad = np.pad(tmrt[:, :, :], pad_width=((tpy, tpy), (tpx, tpx), (0, 0)), mode='constant', constant_values=0)
        self.buildings_pad = np.pad(buildings, pad_width=((tpy, tpy), (tpx, tpx)), mode='constant', constant_values=0)
        self.rows_pad, self.cols_pad = self.buildings_pad.shape

class Treerasters():
# Class containing calculated shadows, regional grouping of shadows if many timesteps, tmrt in shade, tmrt sunlit, difference between shade and sunlit
    __slots__ = ('treeshade', 'treeshade_rg', 'treeshade_bool', 'tpy', 'tpx', 'rows', 'cols', 'rows_s', 'cols_s', 'euclidean', 'euclidean_d', 'tmrt_sun', 'tmrt_shade', 'd_tmrt')
    def __init__(self, treeshade, treeshade_rg, treeshade_bool, cdsm, height):
        # Find min and max rows and cols where there are shadows
        shy, shx = np.where((treeshade > 0) | (cdsm > 0))
        shy_min = np.min(shy); shy_max = np.max(shy) + 1
        shx_min = np.min(shx); shx_max = np.max(shx) + 1

        # Cropping to only where there is a shadow
        self.treeshade = treeshade[shy_min:shy_max, shx_min:shx_max]
        self.treeshade_rg = treeshade_rg[shy_min:shy_max, shx_min:shx_max]
        self.treeshade_bool = 1-treeshade_bool[shy_min:shy_max, shx_min:shx_max,:]
        cdsm_clip = cdsm[shy_min:shy_max, shx_min:shx_max]
        y, x = np.where(cdsm_clip == height)  # Position of tree in clipped shadow image
        self.tpy = y
        self.tpx = x
        self.rows = treeshade.shape[0]
        self.cols = treeshade.shape[1]
        self.rows_s = self.treeshade_rg.shape[0]
        self.cols_s = self.treeshade_rg.shape[1]

        a = np.array((self.tpy, self.tpx))
        b = np.zeros((4,2))
        b[0,:] = np.array((0, 0))   # Upper left corner
        b[1,:] = np.array((0, self.treeshade.shape[0] - 1)) # Lower left corner
        b[2,:] = np.array((self.treeshade.shape[1] - 1, 0)) # Upper right corner
        b[3,:] = np.array((self.treeshade.shape[0] - 1, self.treeshade.shape[1] - 1))   # Lower right corner
        eucl = np.zeros((b.shape[0],1))
        for i in range(b.shape[0]):
            eucl[i,0] = np.linalg.norm(a - b[i,:])
        eucl_d = np.array([eucl[0] + eucl[3], eucl[1] + eucl[2]])
        self.euclidean = np.max(eucl[:,0])
        self.euclidean_d = np.max(eucl_d)

    def tmrt(self, tmrt_sun, tmrt_shade, filter):
        self.tmrt_sun = tmrt_sun
        self.tmrt_shade = tmrt_shade
        self.d_tmrt = self.tmrt_sun - self.tmrt_shade
        if filter == 1:
            import scipy as sp
            # Finding courtyards and small separate areas
            sun_vs_tsh_filtered = sp.ndimage.label(self.d_tmrt)
            sun_sh_d = sun_vs_tsh_filtered[0]
            for i in range(1, sun_sh_d.max() + 1):
                if np.sum(sun_sh_d[sun_sh_d == i]) < 2000:
                    self.d_tmrt[sun_sh_d == i] = 0

        nr_dec = 1
        self.tmrt_sun = np.around(self.tmrt_sun, decimals=nr_dec)
        self.tmrt_shade = np.around(self.tmrt_shade, decimals=nr_dec)
        self.d_tmrt = np.around(self.d_tmrt, decimals=nr_dec)

class Position():
# Class containing y and x positions of trees and their corresponding sum of Tmrt in shade and sum of Tmrt in same area as shade but sunlit
# as well as a unique number for each position. Also a matrix with the unique number in each y,x position in the matrix.
    __slots__ = ('pos', 'pos_m')
    def __init__(self,vector,rows,cols):
        self.pos = vector

        self.pos_m = np.zeros((rows, cols))
        for idx in range(vector.shape[0]):
            y = np.int_(vector[idx, 2])
            x = np.int_(vector[idx, 1])
            self.pos_m[y, x] = vector[idx, 0]

class Treedata():
# Class containing data for the tree that is used in Tree planter, i.e. the tree that is being "planted" and studied
    __slots__ = ('ttype', 'height', 'trunk', 'dia', 'treey', 'treex')
    def __init__(self, ttype, height, trunk, dia, treey, treex):
        self.ttype = ttype
        self.height = height
        self.trunk = trunk
        self.dia = dia
        self.treey = treey
        self.treex = treex


class Regional_groups():
    # Class for creation of regional groups for shadows, i.e. which timesteps shade which pixels
    # Returns a matrix with regional groups and a vector with the corresponding timesteps for each regional group
    # range_ = between which timesteps to calculate regional groups
    # shadow_ = matrix with sum of shadows for all timesteps
    # shadow_ts = shadows for each timestep
    __slots__ = ('shadow', 'timesteps', 'tmrt_rg')
    def __init__(self, range_, shadow_, shadow_ts, tmrt):

        t_r = range(range_.__len__())
        t_l = t_r.__len__()

        shade_u = np.unique(shadow_)                            # Unique values in summation matrix for tree shadows
        shade_max = np.max(shade_u)                             # Maximum value of unique values

        for i in range(1, shade_u.shape[0]):                    # Loop over all unique values
            shade_b = (shadow_ == shade_u[i])                   # Boolean shadow for each timestep i
            shade_r = label(shade_b)                            # Create regional groups
            shade_r_u = np.unique(shade_r[0])                   # Find out how many regional groups, i.e. unique values
            if np.sum(shade_r_u) > 1:                           # If more than there groups, i.e. 0, 1, 2, ... , continue
                for j in range(2, shade_r_u.shape[0]):          # Loop over the unique values and give all but 1 new values
                    shade_b2 = (shade_r[0] == shade_r_u[j])     # Boolean of shadow for each unique value
                    shade_max += 1                              # Add +1 to the maximum value of unique values, continues (creates new unique values)
                    shadow_[shade_b2] = shade_max               # Add these to the building summation matrix

        shade_u_u = np.unique(shadow_)                          # New unique values of regional groups
        sh_vec_t = np.zeros((shade_u_u.shape[0], t_l + 1))      # Empty array for storing which timesteps are found in each regional group
        sh_vec_t[:, 0] = shade_u_u                              # Adding the unique regional groups to the first column
        tmrt_t = np.zeros((shade_u_u.shape[0], 2))
        tmrt_t[:,0] = shade_u_u

        for i in range(1, shade_u_u.shape[0]):                  # Loop over the unique values
            shade_b = (shadow_ == shade_u_u[i])                 # Boolean of each regional group
            for j in t_r:                                       # Loop over each timestep
                shade_b2 = (shadow_ts[:, :, j].copy() == 1)     # Boolean of shadow for each timestep
                shade_b3 = (shade_b) & (shade_b2)               # Find out where they overlap, i.e. which timesteps are found in each regional group
                if np.sum(shade_b3) > 0:                        # If they overlap, continue
                    sh_vec_t[i, 1 + j] = 1                      # Add 1 to timestep column
                    tmrt_t[i,1] += tmrt[j,0]

        self.shadow = shadow_
        self.timesteps = sh_vec_t
        self.tmrt_rg = tmrt_t
