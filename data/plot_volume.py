# In[]
import matplotlib
#matplotlib.use('PDF')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
#import networkx as nx
plt.rc('text', usetex=True)

'''
This code plots the volume of the sync basin of twisted states
'''

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

sfont = {'fontname':'serif'}

# In[]
#winding_number = np.loadtxt('test.txt',delimiter=',')
#winding_number = np.loadtxt('winding_number.txt',delimiter=',')
winding_number = np.loadtxt('winding_numbers.txt',delimiter=',')
winding_number = np.reshape(winding_number, (10,-1)) 

# In[]

fig = plt.figure()
ax = fig.add_subplot(111)

for axis in ['bottom','left']:
  ax.spines[axis].set_linewidth(5)
for axis in ['top','right']:
  ax.spines[axis].set_visible(False)

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

fig.set_size_inches(23,12)
plt.xticks(fontsize = 60)
plt.yticks(fontsize = 60)
plt.xlabel(r'$q$',fontsize = 70)
plt.ylabel(r'$p$',fontsize = 70)

markers = ['o','s','p','h','d']
labels = [r'$\sigma_\Delta=0$',r'$\sigma_\Delta=1$',r'$\sigma_\Delta=2$',r'$\sigma_\Delta=3$',r'$\sigma_\Delta=4$']
for i in range(5):
  (unique, counts) = np.unique(winding_number[i,:], return_counts=True)
  frequencies = np.asarray((unique, counts)).T
  frequencies[:,1] = frequencies[:,1]/len(winding_number[i,:])
  #print(frequencies[:,0])
  #print(frequencies[:,1])
  plt.plot(frequencies[:,0], frequencies[:,1], linestyle='--', marker=markers[i], markersize=30, color=tableau20[2*i], alpha=.6, label=labels[i])
ax.set_yscale('log')
#ax.set_xscale('log')
#ax.set_xticks([4,16,64,128])

plt.legend(loc='upper right', frameon=False, prop={'size':45}, bbox_to_anchor=(1.05, 1.05), ncol=1)

plt.xlim([-6,6])
plt.ylim([2*1e-4,1e0])
plt.gca().tick_params(axis='y', pad=25, size=15, width=3)
plt.gca().tick_params(axis='x', pad=25, size=15, width=3)
fig.set_tight_layout(True)
plt.savefig('basin_size.pdf')


# In[]

freq = np.loadtxt('freq.txt',delimiter=' ')

#for i in range(10):
#  (unique, counts) = np.unique(winding_number[i,:], return_counts=True)
#  frequencies = np.asarray((unique, counts)).T
#  frequencies[:,1] = frequencies[:,1]/len(winding_number[i,:])
#  print(frequencies[:,0])
#  print(frequencies[:,1])

fig = plt.figure()
ax = fig.add_subplot(111)

for axis in ['bottom','left']:
  ax.spines[axis].set_linewidth(5)
for axis in ['top','right']:
  ax.spines[axis].set_visible(False)

ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

fig.set_size_inches(23,12)
plt.xticks(fontsize = 60)
plt.yticks(fontsize = 60)
plt.xlabel(r'$\sigma_\Delta$',fontsize = 70)
plt.ylabel(r'$p$',fontsize = 70)

markers = ['o','s','p','h','d','s','>','*']
labels = [r'$q=0$',r'$q=1$',r'$q=2$',r'$q=3$',r'$q=4$',r'$q=5$',r'$q=6$',r'non-twist']
Sigma = np.arange(0,5,0.5)
for i in [0,1,2,3,4,7]:
#for i in range(8):
  plt.plot(Sigma, freq[1:,i+6], linestyle='--', marker=markers[i], markersize=30, color=tableau20[2*i], alpha=.6, label=labels[i])
ax.set_yscale('log')
#ax.set_xscale('log')
#ax.set_xticks([4,16,64,128])

plt.legend(loc='upper right', frameon=False, prop={'size':45}, bbox_to_anchor=(1.05, .92), ncol=1)

#plt.xlim([-7,7])
plt.ylim([8*1e-4,1.2*1e0])
plt.gca().tick_params(axis='y', pad=25, size=15, width=3)
plt.gca().tick_params(axis='x', pad=25, size=15, width=3)
fig.set_tight_layout(True)
plt.savefig('basin_size_sigma.pdf')