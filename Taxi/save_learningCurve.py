import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import pandas as pd

WINDOW = 20
avg_mask = np.ones(WINDOW) / WINDOW

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

e=np.load('episodes_art.npy')
my_score=np.load('scores_art.npy')
my_score = np.convolve(my_score, avg_mask, 'same')
data = pd.read_csv('scores_yeison.csv')
his_score = data.values[:,1]
his_score = np.convolve(his_score, avg_mask, 'same')


# plt.plot(e,my_score,color=(0,0,1,0.5))
# plt.plot(e,his_score, color=(1,0,0,0.5))
# #plt.plot(e,his_score)
# plt.xlabel(r'\textit{Episode}',fontsize=11)
# plt.ylabel(r'\textit{Cummulative reward} ($\sum{r}$)',fontsize=11)
# plt.title(r"Learning curve for Taxi-v2 gym environment",
#           fontsize=12)
# # Make room for the ridiculously large title.
# plt.subplots_adjust(top=0.8)
# plt.savefig('learningCurve_taxi', format='eps')
# plt.show()

fig, ax = plt.subplots() # create a new figure with a default 111 subplot
ax.plot(e[10:20000], his_score[10:20000], color=(0,0,1,0.5), label='Features option 1')
ax.plot(e[10:20000], my_score[10:20000], color=(1,0,0,0.5), label='Features option 2')
plt.xlabel(r'\textit{Episode}',fontsize=11)
plt.ylabel(r'\textit{Cumulative reward} ($\sum{r}$)',fontsize=11)
plt.title(r"Learning curve for Taxi-v2 gym environment",
          fontsize=12)
ax.legend()
axins = zoomed_inset_axes(ax, 7.0, loc=7) # zoom-factor: 2.5, location: upper-left
axins.plot(e, my_score, color=(0,0,1,0.5))
axins.plot(e, his_score, color=(1,0,0,0.5))
x1, x2, y1, y2 = 18000, 20000, 0, 12 # specify the limits
axins.set_xlim(x1, x2) # apply the x-limits
axins.set_ylim(y1, y2) # apply the y-limits
plt.xticks(visible=False)
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

plt.savefig('./learningCurve_taxi2.eps', format='eps')
plt.show()



