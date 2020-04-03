import numpy as np
from osgeo import gdal, osr
from SOLWEIG.misc import saveraster

infolder1 = 'C:/Users/xwanil/Desktop/Project_4/StudyAreaJarntorget/'
dataSet = gdal.Open(infolder1 + 'CDSM_12_00.tif')
cdsm = dataSet.ReadAsArray().astype(np.float)

# 0900-1000
infolder2 = 'C:/Users/xwanil/Desktop/Project_4/StudyAreaJarntorget/TreePlanterOut/5_trees/0900_1000/20000/'
dataSet = gdal.Open(infolder2 + 'cdsm_tmrt_5_trees_20000_9.tif')
new_trees = dataSet.ReadAsArray().astype(np.float)
new_cdsm = cdsm + new_trees
saveraster(dataSet, infolder2 + str('cdsm_0900_1000.tif') , new_cdsm)

infolder2 = 'C:/Users/xwanil/Desktop/Project_4/StudyAreaJarntorget/TreePlanterOut/5_trees/1200_1300/20000/'
dataSet = gdal.Open(infolder2 + 'cdsm_tmrt_5_trees_20000_9.tif')
new_trees = dataSet.ReadAsArray().astype(np.float)
new_cdsm = cdsm + new_trees
saveraster(dataSet, infolder2 + str('cdsm_1200_1300.tif') , new_cdsm)

infolder2 = 'C:/Users/xwanil/Desktop/Project_4/StudyAreaJarntorget/TreePlanterOut/5_trees/1500_1600/20000/'
dataSet = gdal.Open(infolder2 + 'cdsm_tmrt_5_trees_20000_9.tif')
new_trees = dataSet.ReadAsArray().astype(np.float)
new_cdsm = cdsm + new_trees
saveraster(dataSet, infolder2 + str('cdsm_1500_1600.tif') , new_cdsm)

infolder2 = 'C:/Users/xwanil/Desktop/Project_4/StudyAreaJarntorget/TreePlanterOut/5_trees/0900_1600/20000/'
dataSet = gdal.Open(infolder2 + 'cdsm_tmrt_5_trees_20000_9.tif')
new_trees = dataSet.ReadAsArray().astype(np.float)
new_cdsm = cdsm + new_trees
saveraster(dataSet, infolder2 + str('cdsm_0900_1600.tif') , new_cdsm)