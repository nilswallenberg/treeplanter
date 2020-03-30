class Inputdata():
    def __init__(self,r_range, sh_fl, tmrt_fl, infolder, rows, cols):
        from osgeo import gdal, osr
        import numpy as np

        self.dataSet = gdal.Open(infolder + 'buildings.tif')            # GIS data
        self.buildings = self.dataSet.ReadAsArray().astype(np.float)    # Building raster
        self.dem = np.zeros((rows,cols))                                # Digital elevation model
        self.dsm = np.zeros((rows,cols))                                # Digital surface model
        self.shadow = np.zeros((rows,cols, r_range.__len__()))          # Shadow rasters
        self.tmrt_ts = np.zeros((rows, cols, r_range.__len__()))        # Tmrt for each timestep
        self.tmrt_s = np.zeros((rows, cols))                            # Sum of tmrt for all timesteps
        self.rows = rows    # Rows of rasters
        self.cols = cols    # Cols of rasters

        c = 0
        for i in r_range:
            dataSet1 = gdal.Open(sh_fl[i])
            self.shadow[:, :, c] = dataSet1.ReadAsArray().astype(np.float)
            dataSet2 = gdal.Open(tmrt_fl[i])
            self.tmrt_ts[:, :, c] = dataSet2.ReadAsArray().astype(np.float)
            self.tmrt_s = self.tmrt_s + self.tmrt_ts[:, :, c]
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

class Treedata():
    def __init__(self, ttype, height, trunk, dia, treey, treex):
        self.ttype = ttype
        self.height = height
        self.trunk = trunk
        self.dia = dia
        self.treey = treey
        self.treex = treex
