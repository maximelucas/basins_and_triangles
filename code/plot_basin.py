# In[]
import matplotlib
#matplotlib.use('pdf')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rc('text', usetex=True)

# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)

# In[]
basin = np.loadtxt('q=12_sigma=3.txt',delimiter=',')

fig = plt.figure()
ax = fig.add_subplot(111)

fig.set_size_inches((12,10))
plt.xticks(fontsize = 50)
plt.yticks(fontsize = 50)
plt.xlabel(r'$\alpha_1$',fontsize = 65)
plt.ylabel(r'$\alpha_2$',fontsize = 65)
ax.set_xticks([-np.pi,0,np.pi])
ax.set_yticks([-np.pi,0,np.pi])
ax.set_xticklabels([r'$-\pi$', '$0$', r'$\pi$'])
ax.set_yticklabels([r'$-\pi$', '$0$', r'$\pi$'])
#ax.set_axis_off()
#fig.add_axes(ax)

flag = 0
if 99 in basin:
    flag = 1
    no_sync = np.unique(basin)[-2] + 1
    basin[basin == 99] = no_sync

ub = np.max(basin)
lb = np.min(basin)

# keep the colorbar uniform across different plots
flag = 1
ub = 13
lb = -6

#cm = LinearSegmentedColormap.from_list('OrBu', tableau20, ub-lb+1)
cm = plt.get_cmap('RdBu',ub-lb+1)

plt.imshow(basin,cmap=cm,aspect='equal',vmin=lb-.5,vmax=ub+.5,alpha=.9,origin="lower",interpolation='none',extent=[-np.pi,np.pi,-np.pi,np.pi])
if flag == 0:
    cbar = plt.colorbar(ticks=np.arange(lb,ub+1))
else:
    cbar = plt.colorbar(ticks=np.arange(lb,ub))
cbar.ax.tick_params(labelsize=40)

#plt.axis('off')
fig.set_tight_layout(True)
plt.savefig('basin_q=12_sigma=3.png',dpi=100)



# In[]
basin = np.loadtxt('q=12_sigma=3.txt',delimiter=',')

fig = plt.figure()
ax = fig.add_subplot(111)

#fig.set_size_inches((10,10))
#ax.set_axis_off()

fig.set_size_inches((12,10))
plt.xticks(fontsize = 50)
plt.yticks(fontsize = 50)
plt.xlabel(r'$\alpha_1$',fontsize = 65)
plt.ylabel(r'$\alpha_2$',fontsize = 65)
ax.set_xticks([-np.pi,0,np.pi])
ax.set_yticks([-np.pi,0,np.pi])
ax.set_xticklabels([r'$-\pi$', '$0$', r'$\pi$'])
ax.set_yticklabels([r'$-\pi$', '$0$', r'$\pi$'])

flag = 0
if 99 in basin:
    flag = 1
    no_sync = np.unique(basin)[-2] + 1
    basin[basin == 99] = no_sync

ub = np.max(basin)
lb = np.min(basin)

# keep the colorbar uniform across different plots
flag = 1
ub = 13
lb = -6

cm = LinearSegmentedColormap.from_list('OrBu', tableau20, ub-lb+1)
#cm = plt.get_cmap('RdBu',ub-lb+1)

plt.imshow(basin,cmap=cm,aspect='equal',vmin=lb-.5,vmax=ub+.5,alpha=.9,origin="lower",interpolation='none',extent=[-np.pi,np.pi,-np.pi,np.pi])
if flag == 0:
    cbar = plt.colorbar(ticks=np.arange(lb,ub+1))
else:
    cbar = plt.colorbar(ticks=np.arange(lb,ub))
cbar.ax.tick_params(labelsize=40)

#plt.axis('off')
fig.set_tight_layout(True)
plt.savefig('basin_q=12_sigma=3_a.png',dpi=100)
