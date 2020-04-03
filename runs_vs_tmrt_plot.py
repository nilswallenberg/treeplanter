import numpy as np
import matplotlib.pylab as plt
import glob

infolder = 'C:/Users/xwanil/Desktop/Project_4/Figures/Jarntorget/Tables/'

marker_style = np.array(['*','x','*','x'])
marker_color = np.array(['r','b','k','c'])
fn_vec = ['0900-1000','0900-1600','1200-1300','1500-1600']

fig = plt.figure()
ax = plt.subplot(111)

filename = [f for f in glob.glob(infolder + 'table_*')]
for i in range(filename.__len__()):
    A = np.loadtxt(filename[i], delimiter=' ', skiprows=1)
    A_n = np.zeros((A.shape[0]))
    for j in range(A_n.shape[0]):
        A_n[j] = (A[j,1] - np.min(A[:,1]))/(np.max(A[:,1]) - np.min(A[:,1]))
    ax.plot(np.arange(A[:, 0].shape[0]), A_n[:], c=marker_color[i], marker=marker_style[i], markersize=6, linestyle='none', mfc='none', label=str(fn_vec[i]))
ax.set_ylabel('Normalized potential decrease in T$_{mrt}$')
ax.set_xlabel('Iteration number')
#ax.set_ylim((1290,1310))
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, framealpha=1, edgecolor='k', ncol=2, labelspacing=0.2, columnspacing=0.5)
ax.legend(loc='best')
labels = ['10', '100', '500', '1000', '2000', '3000', '4000', '5000', '10000', '20000']
label_n = np.arange(0,A.shape[0])
plt.xticks(label_n,(labels))
#plt.show()

f_name = infolder + 'figure_runs_vs_tmrt.tif'
plt.savefig(f_name, dpi=150)
plt.clf()

test=0