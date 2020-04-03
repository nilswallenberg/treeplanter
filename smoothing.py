import numpy as np
from osgeo import gdal
import matplotlib.pyplot as plt

filepath = 'C:/Users/xwanil/Desktop/Project_4/inputdata_treeplanter/GustafAdolfSmall/SOLWEIG_RUN/Tmrt_1983_173_1300D.tif'
gdal_cdsm = gdal.Open(filepath)
tmrt = gdal_cdsm.ReadAsArray().astype(np.float)
row = tmrt.shape[0]
col = tmrt.shape[1]
result = np.zeros((row,col))

filterrange = 1.0
domain = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

# sink and median filter
temp = np.zeros((row,col))
sink_med_loop = np.zeros((row,col))
loop = 5
# sink filter loop
for ij in range(loop):
    if ij == 0:
        temp[:,:] = tmrt[:,:]
    else:
        temp[:,:] = sink_med_loop[:,:]
    for j in np.arange(1, row - 1):
        for i in np.arange(1, col - 1):
            dom = temp[j - 1:j + 2, i - 1:i + 2]
            if tmrt[j, i] == dom.max():
                if (dom.max() - dom.min()) < filterrange:
                    sink_med_loop[j, i] = np.mean(dom)
                else:
                    sink_med_loop[j, i] = temp[j, i]
            else:
                sink_med_loop[j, i] = temp[j, i]

median_tmrt = np.zeros((row,col))
for j in np.arange(1, row - 1):
    for i in np.arange(1, col - 1):
        dom = sink_med_loop[j - 1:j + 2, i - 1:i + 2]
        if (dom.max() - dom.min()) < filterrange:
            median_tmrt[j, i] = np.median(dom * domain)
        else:
            median_tmrt[j, i] = sink_med_loop[j, i]

fig = plt.figure(figsize=(14, 7))
ax1 = plt.subplot(2, 2, 1)
im1 = ax1.imshow(tmrt, clim=[50, 60])
plt.title('Original')
plt.colorbar(im1)
ax2 = plt.subplot(2, 2, 2)
im2 = ax2.imshow(sink_med_loop, clim=[50, 60])
plt.title('Sink filter and median filter')
plt.colorbar(im2)
ax3 = plt.subplot(2, 2, 3)
im3 = ax3.imshow(sink_med_loop - tmrt, clim=[-1, 1])
plt.title('Sink filter vs. Tmrt')
plt.colorbar(im3)
plt.tight_layout()
plt.show()

# mean filter
mean_tmrt = np.zeros((row,col))
for j in np.arange(1, row - 1):
    for i in np.arange(1, col - 1):
        dom = tmrt[j - 1:j + 2, i - 1:i + 2]
        if (dom.max() - dom.min()) < filterrange:
            mean_tmrt[j, i] = np.mean(dom * domain)
        else:
            mean_tmrt[j, i] = tmrt[j, i]

mean_loop = mean_tmrt.copy()
for ij in range(10):
    result_temp = mean_loop
    for j in np.arange(1, row - 1):
        for i in np.arange(1, col - 1):
            dom = result_temp[j - 1:j + 2, i - 1:i + 2]
            if (dom.max() - dom.min()) < filterrange:
                mean_loop[j, i] = np.mean(dom * domain)
            else:
                mean_loop[j, i] = result_temp[j, i]

fig = plt.figure(figsize=(14, 7))
ax1 = plt.subplot(2, 2, 1)
im1 = ax1.imshow(tmrt, clim=[50, 60])
plt.colorbar(im1)
ax2 = plt.subplot(2, 2, 2)
im2 = ax2.imshow(mean_tmrt, clim=[50, 60])
plt.colorbar(im2)
ax3 = plt.subplot(2, 2, 3)
im3 = ax3.imshow(mean_loop, clim=[50,60])
plt.colorbar(im3)
plt.tight_layout()
plt.show()

# sink filter
sink_tmrt = np.zeros((row,col))
for j in np.arange(1, row - 1):
    for i in np.arange(1, col - 1):
        dom = tmrt[j - 1:j + 2, i - 1:i + 2]
        if tmrt[j, i] == dom.max():
            if (dom.max() - dom.min()) < filterrange:
                sink_tmrt[j, i] = np.mean(dom)
            else:
                sink_tmrt[j, i] = tmrt[j, i]
        else:
            sink_tmrt[j, i] = tmrt[j, i]

fig = plt.figure(figsize=(14, 7))
ax1 = plt.subplot(2, 2, 1)
im1 = ax1.imshow(tmrt, clim=[50, 60])
plt.colorbar(im1)
ax2 = plt.subplot(2, 2, 2)
im2 = ax2.imshow(sink_tmrt, clim=[50, 60])
plt.colorbar(im2)
ax3 = plt.subplot(2, 2, 3)
im3 = ax3.imshow(sink_tmrt - tmrt, clim=[-1, 1])
plt.colorbar(im3)
plt.tight_layout()
plt.show()

temp = np.zeros((row,col))
sink_loop = np.zeros((row,col))
loop = 5
# sink filter loop
for ij in range(loop):
    if ij == 0:
        temp[:,:] = tmrt[:,:]
    else:
        temp[:,:] = sink_loop[:,:]
    for j in np.arange(1, row - 1):
        for i in np.arange(1, col - 1):
            dom = temp[j - 1:j + 2, i - 1:i + 2]
            if tmrt[j, i] == dom.max():
                if (dom.max() - dom.min()) < filterrange:
                    sink_loop[j, i] = np.mean(dom)
                else:
                    sink_loop[j, i] = temp[j, i]
            else:
                sink_loop[j, i] = temp[j, i]

fig = plt.figure(figsize=(14, 7))
ax1 = plt.subplot(2, 2, 1)
im1 = ax1.imshow(tmrt, clim=[50, 60])
plt.colorbar(im1)
ax2 = plt.subplot(2, 2, 2)
im2 = ax2.imshow(sink_loop, clim=[50, 60])
plt.colorbar(im2)
ax3 = plt.subplot(2, 2, 3)
im3 = ax3.imshow(sink_loop - tmrt, clim=[-1, 1])
plt.colorbar(im3)
plt.tight_layout()
plt.show()

mean_sink = np.zeros((row,col))

# mean filter on sink
for j in np.arange(1, row - 1):
    for i in np.arange(1, col - 1):
        dom = sink_tmrt[j - 1:j + 2, i - 1:i + 2]
        if (dom.max() - dom.min()) < filterrange:
            mean_sink[j, i] = np.mean(dom * domain)
        else:
            mean_sink[j, i] = sink_tmrt[j, i]

fig = plt.figure(figsize=(14, 7))
ax1 = plt.subplot(2, 2, 1)
im1 = ax1.imshow(tmrt, clim=[50, 60])
plt.colorbar(im1)
ax2 = plt.subplot(2, 2, 2)
im2 = ax2.imshow(sink_loop, clim=[50, 60])
plt.colorbar(im2)
ax3 = plt.subplot(2, 2, 3)
im3 = ax3.imshow(mean_sink, clim=[50,60])
plt.colorbar(im3)
ax4 = plt.subplot(2, 2, 4)
im4 = ax4.imshow(mean_sink - tmrt, clim=[-1, 1])
plt.colorbar(im4)
plt.tight_layout()
plt.show()

mean_sink_loop = np.zeros((row,col))

# mean filter on sink
for j in np.arange(1, row - 1):
    for i in np.arange(1, col - 1):
        dom = sink_loop[j - 1:j + 2, i - 1:i + 2]
        if (dom.max() - dom.min()) < filterrange:
            mean_sink_loop[j, i] = np.mean(dom * domain)
        else:
            mean_sink_loop[j, i] = sink_loop[j, i]

fig = plt.figure(figsize=(14, 7))
ax1 = plt.subplot(2, 2, 1)
im1 = ax1.imshow(tmrt, clim=[50, 60])
plt.colorbar(im1)
ax2 = plt.subplot(2, 2, 2)
im2 = ax2.imshow(sink_loop, clim=[50, 60])
plt.colorbar(im2)
ax3 = plt.subplot(2, 2, 3)
im3 = ax3.imshow(mean_sink_loop, clim=[50,60])
plt.colorbar(im3)
ax4 = plt.subplot(2, 2, 4)
im4 = ax4.imshow(mean_sink_loop - tmrt, clim=[-1,1])
plt.colorbar(im4)
plt.tight_layout()
plt.show()

sigma_x = 1.0
sigma_y = sigma_x

tmrt_temp = np.zeros((row, col))

from scipy import ndimage

sigma = [sigma_y, sigma_x]

tmrt_gauss = ndimage.gaussian_filter(tmrt, sigma, mode='constant')

fig = plt.figure(figsize=(14, 7))
ax1 = plt.subplot(2, 2, 1)
im1 = ax1.imshow(tmrt, clim=[50, 60])
plt.colorbar(im1)
ax2 = plt.subplot(2, 2, 2)
im2 = ax2.imshow(tmrt_gauss, clim=[50, 60])
plt.colorbar(im2)
plt.tight_layout()
plt.show()

tmrt_mf = ndimage.median_filter(tmrt, size=3, mode='constant', cval=0)

fig = plt.figure(figsize=(14, 7))
ax1 = plt.subplot(2, 2, 1)
im1 = ax1.imshow(tmrt, clim=[50, 60])
plt.colorbar(im1)
ax2 = plt.subplot(2, 2, 2)
im2 = ax2.imshow(tmrt_mf, clim=[50, 60])
plt.colorbar(im2)
plt.tight_layout()
plt.title('Median filter')
plt.show()

tmrt_uf = ndimage.uniform_filter(tmrt, size=3, mode='constant', cval=0)

fig = plt.figure(figsize=(14, 7))
ax1 = plt.subplot(2, 2, 1)
im1 = ax1.imshow(tmrt, clim=[50, 60])
plt.colorbar(im1)
ax2 = plt.subplot(2, 2, 2)
im2 = ax2.imshow(tmrt_uf, clim=[50, 60])
plt.colorbar(im2)
plt.tight_layout()
plt.show()

from skimage.restoration import denoise_tv_chambolle

tmrt_tv = denoise_tv_chambolle(tmrt, weight=0.1)

fig = plt.figure(figsize=(14, 7))
ax1 = plt.subplot(2, 2, 1)
im1 = ax1.imshow(tmrt, clim=[50, 60])
plt.colorbar(im1)
ax2 = plt.subplot(2, 2, 2)
im2 = ax2.imshow(tmrt_tv, clim=[50, 60])
plt.colorbar(im2)
plt.tight_layout()
plt.show()

tmrt_maxf = ndimage.maximum_filter(tmrt, size=3, mode='constant', cval=0)
print(np.percentile(tmrt_maxf, 97))

fig = plt.figure(figsize=(14, 7))
ax1 = plt.subplot(2, 2, 1)
im1 = ax1.imshow(tmrt, clim=[50, 60])
plt.colorbar(im1)
ax2 = plt.subplot(2, 2, 2)
im2 = ax2.imshow(tmrt_maxf, clim=[50, 60])
plt.colorbar(im2)
plt.tight_layout()
plt.show()

tmrt_rank = ndimage.rank_filter(tmrt, rank=-1, size=3, mode='constant', cval=0)

fig = plt.figure(figsize=(14, 7))
ax1 = plt.subplot(2, 2, 1)
im1 = ax1.imshow(tmrt, clim=[50, 60])
plt.colorbar(im1)
ax2 = plt.subplot(2, 2, 2)
im2 = ax2.imshow(tmrt_rank, clim=[50, 60])
plt.colorbar(im2)
plt.tight_layout()
plt.show()

tmrt_ggm = ndimage.gaussian_gradient_magnitude(tmrt, sigma=10)

fig = plt.figure(figsize=(14, 7))
ax1 = plt.subplot(2, 2, 1)
im1 = ax1.imshow(tmrt, clim=[50, 60])
plt.colorbar(im1)
ax2 = plt.subplot(2, 2, 2)
im2 = ax2.imshow(tmrt_ggm, clim=[50, 60])
plt.colorbar(im2)
plt.tight_layout()
plt.show()

tmrt_genm = ndimage.generic_gradient_magnitude(tmrt, ndimage.prewitt)

fig = plt.figure(figsize=(14, 7))
ax1 = plt.subplot(2, 2, 1)
im1 = ax1.imshow(tmrt, clim=[50, 60])
plt.colorbar(im1)
ax2 = plt.subplot(2, 2, 2)
im2 = ax2.imshow(tmrt_genm, clim=[50, 60])
plt.colorbar(im2)
plt.tight_layout()
plt.show()

from scipy.signal import argrelextrema
lm_y_t, lm_x_t = argrelextrema(tmrt, np.greater)
lm_y_m, lm_x_m = argrelextrema(mean_tmrt, np.greater)
lm_y_ml, lm_x_ml = argrelextrema(mean_loop, np.greater)
lm_y_s, lm_x_s = argrelextrema(sink_tmrt, np.greater)
lm_y_sl, lm_x_sl = argrelextrema(sink_loop, np.greater)
lm_y_ms, lm_x_ms = argrelextrema(mean_sink, np.greater)
lm_y_msl, lm_x_msl = argrelextrema(mean_sink_loop, np.greater)
lm_y_ga, lm_x_ga = argrelextrema(tmrt_gauss, np.greater)
lm_y_mf, lm_x_mf = argrelextrema(tmrt_mf, np.greater)
lm_y_uf, lm_x_uf = argrelextrema(tmrt_uf, np.greater)
lm_y_tv, lm_x_tv = argrelextrema(tmrt_tv, np.greater)
lm_y_maxf, lm_x_maxf = argrelextrema(tmrt_maxf, np.greater)
lm_y_rank, lm_x_rank = argrelextrema(tmrt_rank, np.greater)
lm_y_ggm, lm_x_ggm = argrelextrema(tmrt_ggm, np.greater)
lm_y_genm, lm_x_genm = argrelextrema(tmrt_genm, np.greater)
lm_y_med, lm_x_med = argrelextrema(median_tmrt, np.greater)
lm_y_smed, lm_x_smed = argrelextrema(sink_med_loop, np.greater)

print('tmrt = ' + str(lm_y_t.shape[0]))
print('mean tmrt = ' + str(lm_y_m.shape[0]))
print('mean tmrt loop = ' + str(lm_y_ml.shape[0]))
print('sink tmrt = ' + str(lm_y_s.shape[0]))
print('sink loop = ' + str(lm_y_sl.shape[0]))
print('mean sink = ' + str(lm_y_ms.shape[0]))
print('mean sink loop = ' + str(lm_y_msl.shape[0]))
print('tmrt gaussian = ', str(lm_y_ga.shape[0]))
print('tmrt median = ', str(lm_y_mf.shape[0]))
print('tmrt uniform = ', str(lm_y_uf.shape[0]))
print('tmrt tv = ', str(lm_y_tv.shape[0]))
print('tmrt max filter = ', str(lm_y_maxf.shape[0]))
print('tmrt rank filter = ', str(lm_y_rank.shape[0]))
print('tmrt gaussian gradient magnitude filter = ', str(lm_y_ggm.shape[0]))
print('tmrt generic gradient magnitude filter = ', str(lm_y_genm.shape[0]))
print('tmrt custom median filter = ', str(lm_y_med.shape[0]))
print('tmrt sink filter and custom median filter = ', str(lm_y_smed.shape[0]))

test = 4

