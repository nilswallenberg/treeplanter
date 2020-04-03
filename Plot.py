import numpy as np
import glob
import matplotlib.pylab as plt
from matplotlib.pyplot import cm

## Loading data
# infolder = 'C:/Users/xwanil/Desktop/Project_4/Output/Figures/'
#
# filename = [f for f in glob.glob(infolder + 'table_*')]
# color=iter(cm.rainbow(np.linspace(0,1,7)))
# for i in range(0, filename.__len__(), 2):
#     A = np.loadtxt(filename[i], delimiter=' ', skiprows=1)
#     B = np.loadtxt(filename[i+1], delimiter=' ', skiprows=1)
#     C = np.concatenate((A[:,1],B[:,1]),axis=0)
#     n1 = A / np.linalg.norm(C)
#     n2 = B / np.linalg.norm(C)
#     col = next(color)
#     #plt.plot(A[:, 0], n1[:, 1], 'o', mfc='none')
#     #plt.plot(B[:, 0], n2[:, 1], '^', mfc='none')
#     plt.plot(np.arange(A[:, 0].shape[0]), n1[:, 1], c=col, marker='o', linestyle='none', mfc='none')
#     plt.plot(np.arange(B[:, 0].shape[0]), n2[:, 1], c=col, marker='^', linestyle='none', mfc='none')
#
# t = "First number = n trees, \n second number = random restart (0) or ILS (1)"
# plt.text(8,0.127,t,fontsize=10,wrap=True)
# plt.xlim(-1,28)
# plt.ylim(0.125,0.137)
# labels=np.array(['100','200','300','400','500','600','700','800','900','1000','1100','1200','1300','1400','1500','1600','1700','1800','1900','2000','3000','4000','5000','6000','7000','8000','9000','10000'])
# plt.xticks(np.arange(A[:,0].shape[0]), labels, rotation=90)
# #plt.yticks(np.arange(0.125,0.137,step=0.001))
# plt.ylabel(r'Normalized T$_{mrt}$')
# plt.xlabel('Number of model iterations')
# plt.gca().legend(('2-0','2-1','3-0','3-1','4-0','4-1','5-0','5-1','6-0','6-1','7-0','7-1','8-0','8-1'),ncol=7, labelspacing=0.2, columnspacing=0.5,loc=4)
# #plt.show()
#
# f_name = infolder + 'tmrt1.tif'
# #plt.savefig(f_name, dpi=150)
# plt.clf()
#
# color=iter(cm.rainbow(np.linspace(0,1,7)))
# for i in range(0, filename.__len__(), 2):
#     A = np.loadtxt(filename[i], delimiter=' ', skiprows=1)
#     B = np.loadtxt(filename[i+1], delimiter=' ', skiprows=1)
#     C = np.concatenate((A[:,2],B[:,2]),axis=0)
#     n1 = A / np.linalg.norm(C)
#     n2 = B / np.linalg.norm(C)
#     col = next(color)
#     plt.plot(np.arange(A[:, 0].shape[0]), n1[:, 2], c=col, marker='o', linestyle='none', mfc='none')
#     plt.plot(np.arange(B[:, 0].shape[0]), n2[:, 2], c=col, marker='^', linestyle='none', mfc='none')
#
# t = "First number = n trees, \n second number = random restart (0) or ILS (1)"
# plt.text(0.05,0.31,t,fontsize=10,wrap=True)
# plt.xlim(-1,28)
# #plt.ylim(0.125,0.137)
# labels=np.array(['100','200','300','400','500','600','700','800','900','1000','1100','1200','1300','1400','1500','1600','1700','1800','1900','2000','3000','4000','5000','6000','7000','8000','9000','10000'])
# plt.xticks(np.arange(A[:,0].shape[0]), labels, rotation=90)
# plt.ylabel('Normalized time (s)')
# plt.xlabel('Number of model iterations')
# plt.gca().legend(('2-0','2-1','3-0','3-1','4-0','4-1','5-0','5-1','6-0','6-1','7-0','7-1','8-0','8-1'),ncol=7, labelspacing=0.2, columnspacing=0.5,loc=2)
# #plt.show()
#
# f_name = infolder + 'tmrt2.tif'
# #plt.savefig(f_name, dpi=150)
# plt.clf()
#
# color=iter(cm.rainbow(np.linspace(0,1,7)))
# for i in range(0, filename.__len__(), 2):
#     A = np.loadtxt(filename[i], delimiter=' ', skiprows=1)
#     B = np.loadtxt(filename[i+1], delimiter=' ', skiprows=1)
#     C = A[:,2]-B[:,2]
#     col = next(color)
#     plt.plot(np.arange(A[:, 0].shape[0]), C, c=col, marker='o', linestyle='none', mfc='none')
#
# plt.xlim(-1,28)
# #plt.ylim(0.125,0.137)
# labels=np.array(['100','200','300','400','500','600','700','800','900','1000','1100','1200','1300','1400','1500','1600','1700','1800','1900','2000','3000','4000','5000','6000','7000','8000','9000','10000'])
# plt.xticks(np.arange(A[:,0].shape[0]), labels, rotation=90)
# plt.ylabel('Difference in time between random restart and ILS (s)')
# plt.xlabel('Number of model iterations')
# plt.gca().legend(('2','3','4','5','6','7','8'),ncol=7, labelspacing=0.2, columnspacing=0.5,loc=2)
# #plt.show()
#
# f_name = infolder + 'tmrt3.tif'
# #plt.savefig(f_name, dpi=150)
# plt.clf()

##

# infolder = 'C:/Users/xwanil/Desktop/Project_4/Output_dummy/Figures/'

# filename1 = [f for f in glob.glob(infolder + 'table_100_*')]
# color=iter(cm.rainbow(np.linspace(0,1,7)))
# A = np.loadtxt(filename1[0], delimiter=' ', skiprows=1)
# B = np.loadtxt(filename1[1], delimiter=' ', skiprows=1)
# plt.plot(np.arange(A[:, 0].shape[0]), A[:, 1], c='r', marker='o', linestyle='none', mfc='none')
# plt.plot(np.arange(B[:, 0].shape[0]), B[:, 1], c='b', marker='^', linestyle='none', mfc='none')
# plt.ylabel('Possible decrease in T$_{mrt}$')
# plt.xlabel('Iteration number')
# plt.title('100')
# plt.ylim((1200,1310))
# plt.gca().legend(('Random restart', 'Iterative Local Search'), labelspacing=0.2, columnspacing=0.5,loc=4)
# #plt.show()
#
# f_name = infolder + 'tmrt_100.tif'
# plt.savefig(f_name, dpi=150)
# plt.clf()
#
# filename1 = [f for f in glob.glob(infolder + 'table_500_*')]
# color=iter(cm.rainbow(np.linspace(0,1,7)))
# A = np.loadtxt(filename1[0], delimiter=' ', skiprows=1)
# B = np.loadtxt(filename1[1], delimiter=' ', skiprows=1)
# plt.plot(np.arange(A[:, 0].shape[0]), A[:, 1], c='r', marker='o', linestyle='none', mfc='none')
# plt.plot(np.arange(B[:, 0].shape[0]), B[:, 1], c='b', marker='^', linestyle='none', mfc='none')
# plt.ylabel('Possible decrease in T$_{mrt}$')
# plt.xlabel('Iteration number')
# plt.title('500')
# plt.ylim((1200,1310))
# plt.gca().legend(('Random restart', 'Iterative Local Search'), labelspacing=0.2, columnspacing=0.5,loc=4)
# #plt.show()
#
# f_name = infolder + 'tmrt_500.tif'
# plt.savefig(f_name, dpi=150)
# plt.clf()
#
# filename1 = [f for f in glob.glob(infolder + 'table_1000_*')]
# color=iter(cm.rainbow(np.linspace(0,1,7)))
# A = np.loadtxt(filename1[0], delimiter=' ', skiprows=1)
# B = np.loadtxt(filename1[1], delimiter=' ', skiprows=1)
# plt.plot(np.arange(A[:, 0].shape[0]), A[:, 1], c='r', marker='o', linestyle='none', mfc='none')
# plt.plot(np.arange(B[:, 0].shape[0]), B[:, 1], c='b', marker='^', linestyle='none', mfc='none')
# plt.ylabel('Possible decrease in T$_{mrt}$')
# plt.xlabel('Iteration number')
# plt.title('1000')
# plt.ylim((1200,1310))
# plt.gca().legend(('Random restart', 'Iterative Local Search'), labelspacing=0.2, columnspacing=0.5,loc=4)
# #plt.show()
#
# f_name = infolder + 'tmrt_1000.tif'
# plt.savefig(f_name, dpi=150)
# plt.clf()
#
# filename1 = [f for f in glob.glob(infolder + 'table_1500_*')]
# color=iter(cm.rainbow(np.linspace(0,1,7)))
# A = np.loadtxt(filename1[0], delimiter=' ', skiprows=1)
# B = np.loadtxt(filename1[1], delimiter=' ', skiprows=1)
# plt.plot(np.arange(A[:, 0].shape[0]), A[:, 1], c='r', marker='o', linestyle='none', mfc='none')
# plt.plot(np.arange(B[:, 0].shape[0]), B[:, 1], c='b', marker='^', linestyle='none', mfc='none')
# plt.ylabel('Possible decrease in T$_{mrt}$')
# plt.xlabel('Iteration number')
# plt.title('1500')
# plt.ylim((1200,1310))
# plt.gca().legend(('Random restart', 'Iterative Local Search'), labelspacing=0.2, columnspacing=0.5,loc=4)
# #plt.show()
#
# f_name = infolder + 'tmrt_1500.tif'
# plt.savefig(f_name, dpi=150)
# plt.clf()
#
# filename1 = [f for f in glob.glob(infolder + 'table_2000_*')]
# color=iter(cm.rainbow(np.linspace(0,1,7)))
# A = np.loadtxt(filename1[0], delimiter=' ', skiprows=1)
# B = np.loadtxt(filename1[1], delimiter=' ', skiprows=1)
# plt.plot(np.arange(A[:, 0].shape[0]), A[:, 1], c='r', marker='o', linestyle='none', mfc='none')
# plt.plot(np.arange(B[:, 0].shape[0]), B[:, 1], c='b', marker='^', linestyle='none', mfc='none')
# plt.ylabel('Possible decrease in T$_{mrt}$')
# plt.xlabel('Iteration number')
# plt.title('2000')
# plt.ylim((1200,1310))
# plt.gca().legend(('Random restart', 'Iterative Local Search'), labelspacing=0.2, columnspacing=0.5,loc=4)
# #plt.show()
#
# f_name = infolder + 'tmrt_2000.tif'
# plt.savefig(f_name, dpi=150)
# plt.clf()
#
# filename1 = [f for f in glob.glob(infolder + 'table_5000_*')]
# color=iter(cm.rainbow(np.linspace(0,1,7)))
# A = np.loadtxt(filename1[0], delimiter=' ', skiprows=1)
# B = np.loadtxt(filename1[1], delimiter=' ', skiprows=1)
# plt.plot(np.arange(A[:, 0].shape[0]), A[:, 1], c='r', marker='o', linestyle='none', mfc='none')
# plt.plot(np.arange(B[:, 0].shape[0]), B[:, 1], c='b', marker='^', linestyle='none', mfc='none')
# plt.ylabel('Possible decrease in T$_{mrt}$')
# plt.xlabel('Iteration number')
# plt.title('5000')
# plt.ylim((1200,1310))
# plt.gca().legend(('Random restart', 'Iterative Local Search'), labelspacing=0.2, columnspacing=0.5,loc=4)
# #plt.show()
#
# f_name = infolder + 'tmrt_5000.tif'
# plt.savefig(f_name, dpi=150)
# plt.clf()

infolder = 'C:/Users/xwanil/Desktop/Project_4/Output_dummy/Figures/'

fn_vec = np.array([100,500,1000,1500,2000,2500, 3000, 5000, 10000, 20000])
marker_style = np.array(['*','*','*','*','x','x','x','x','.','.'])
marker_color = np.array(['r','b','k','c','r','b','k', 'c', 'r','b'])

fn_vec = np.array([500,1000,2500,5000,10000,20000])
marker_style = np.array(['*','*','x','x','.','.'])
marker_color = np.array(['r','b','k','c','m','g'])

fig = plt.figure()
ax = plt.subplot(111)
for i in range(fn_vec.shape[0]):
    filename = [f for f in glob.glob(infolder + 'table_' + str(fn_vec[i]) + '_*')]
    A = np.loadtxt(filename[0], delimiter=' ', skiprows=1)
    ax.plot(np.arange(A[:, 0].shape[0]), A[:, 1], c=marker_color[i], marker=marker_style[i], markersize=4, linestyle='none', mfc='none', label=str(fn_vec[i]) + ' iterations')
ax.set_ylabel('Normalized possible decrease in T$_{mrt}$')
ax.set_xlabel('Iteration number')
#plt.title('5000')
ax.set_ylim((1290,1310))
#ax.legend(('100 iterations', '500 iterations', '1000 iterations', '1500 iterations', '2000 iterations', '5000 iterations'), labelspacing=0.2, columnspacing=0.5,loc=4)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fancybox=True, framealpha=1, edgecolor='k', ncol=2, labelspacing=0.2, columnspacing=0.5)

f_name = infolder + 'tmrt_all_.tif'
plt.savefig(f_name, dpi=150)
plt.clf()

fig = plt.figure()
ax = plt.subplot(111)

fn_vec = np.array([100,500,1000,1500,2000,2500, 3000, 5000, 10000, 20000])

labels = ['100', '500', '1000', '1500', '2000', '2500', '3000', '5000', '10000', '20000']
label_n = np.arange(1,fn_vec.shape[0]+1)

for i in range(fn_vec.shape[0]):
    filename = [f for f in glob.glob(infolder + 'table_' + str(fn_vec[i]) + '_*')]
    A = np.loadtxt(filename[0], delimiter=' ', skiprows=1)
    ytemp = A[:,4:7]
    xtemp = A[:,7:]
    ypos = np.zeros((ytemp.shape[0],ytemp.shape[1]))
    xpos = np.zeros((xtemp.shape[0], xtemp.shape[1]))
    for j in range(ytemp.shape[0]):
        ysort = np.argsort(ytemp[j,:])
        ypos[j,:] = ytemp[j,ysort]
        xpos[j,:] = xtemp[j,ysort]
    yxpos = np.concatenate((ypos,xpos), axis = 1)
    yxu, yxuc = np.unique(yxpos, return_counts=True, axis=0)
    bar_c = 0
    for j in range(yxuc.shape[0]):
        temp = (yxuc[j]/np.sum(yxuc))*100
        ax.bar(i+1,temp, bottom=bar_c, width=0.5, edgecolor='k')
        bar_c += temp
plt.xticks(label_n,(labels))
plt.ylabel('Unique positions and their percentage out of 100 model runs')
plt.xlabel('Number of iterations')
#plt.show()

f_name = infolder + 'positions.tif'
plt.savefig(f_name, dpi=150)
plt.clf()

test = 0