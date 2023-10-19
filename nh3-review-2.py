import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import pandas as pd
import matplotlib.dates as mdates
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

'''Figure styles'''
mpl.rc('font',weight='bold',size=14, **{'family':'sans-serif','sans-serif':['Arial']})
mpl.rc('lines',linewidth=1.5,markersize=11)
mpl.rc('xtick',labelsize=14)
mpl.rc('ytick',labelsize=14)
mpl.rc('axes',labelsize=14, labelweight='bold', prop_cycle = cycler(color=[(0,0,0),(1,0,0),(0,0,1),(1,0,1),(0,0.5,0),(0,0,0.5),(0.5,0,1),(0.5,0,0.5),(0.5,0,0),(0.5,0.5,0)]), labelpad = 6.0)
mpl.rc('legend',fontsize=12,frameon=True)
mpl.rc('xtick.major', width = 1.5)
mpl.rc('ytick.major', width = 1.5)
mpl.rc('errorbar', capsize=4)
mpl.rc('scatter',marker='o')
color_map = plt.cm.get_cmap('coolwarm')
# pastels = ['#61a8ff','#ff6961','#61ffb8']
pastels = ['#00AEEF','#E0001B','#008D48']

def jitt(arr):
    stdev = .0005 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

colormap = {'Autoclave':pastels[0],'Compartment cell':pastels[1],'GDE cell':pastels[2],'Glass cell':pastels[0]}
# colormap = {'LiNTf2':pastels[0],'LiBF4':pastels[1],'LiClO4':pastels[2],'LiFSI':pastels[3],'LiOTf':pastels[4],'LiPF6':pastels[5]}
file_pre = 'N2_NH3_review'
df = pd.read_csv(file_pre+'.csv')
# df = df[8:]
df = df[df['Fig1']=='y']
# fig, ax = plt.subplots(figsize=(8.8, 8.8), layout="constrained",dpi=300)
# ax.set_aspect('equal')
# sc1 = sns.heatmap(df.corr(),annot=True,ax=ax)
# plt.savefig('LiNRR-heatmap.tif')
# plt.show()

dates = [datetime.strptime(d, "%Y-%m-%d") for d in df['Date Published']]
fig, ax = plt.subplots(figsize=(8.8, 4), layout="constrained",dpi=300)
divider = make_axes_locatable(ax)
ax2 = divider.new_horizontal(size="300%", pad=0.2)
ax3 = divider.new_horizontal(size="990%", pad=0.2)
fig.add_axes(ax2)
fig.add_axes(ax3)

df['color'] = df['Cell Type'].apply(lambda x: colormap[x])
df_author = df['Author'].unique()

# colors_author = {df_author[0]:'#6da7de',df_author[1]:'#00AEEF',
# df_author[2]:'#2c5043',df_author[3]:'#8e541f',df_author[4]:'#d3c913',df_author[5]:'#eb861e',
# df_author[6]:'#EE266D',df_author[7]:'#9e0059',
# df_author[8]:'#E0001B',df_author[9]:'#792C89',df_author[10]:'#d82222',df_author[11]:'#d82222',df_author[12]:'#d82222',df_author[13]:'#d82222',df_author[14]:'#d82222'} 
# df['color'] = df['Author'].apply(lambda x: colors_author[x])

# df_max = df.loc[df.groupby('Author')['FE%'].idxmax()]
# print(df_max['FE%'])
# datemax = [datetime.strptime(d, "%Y-%m-%d") for d in df_max['Date Published']]

sc1 = ax.scatter(jitt(dates), df['FE%'], marker="o",
        c=df['color'],edgecolor='k',s=(100*df['N2 Pressure (atm/bar)'])**0.6,linewidth=0.7, alpha = 0.8)  
sc2 = ax2.scatter(jitt(dates),df['FE%'], marker="o",
        c=df['color'],edgecolor='k',s=(100*df['N2 Pressure (atm/bar)'])**0.6,linewidth=0.7, alpha = 0.8)
sc3 = ax3.scatter(jitt(dates),df['FE%'], marker="o",
        c=df['color'],edgecolor='k',s=(100*df['N2 Pressure (atm/bar)'])**0.6,linewidth=0.7, alpha = 0.8)
# print(10*np.sqrt((df['Current Density (mA/cm2)'])*0.01*df['FE%']))
# plt.colorbar(s c,label='Log Current Density (mA/cm$\mathregular{^2}$)')
# format x-axis with 4-month intervals
# ax.legend(*sc.legend_elements("sizes", num=6))

# handles, labels = sc.legend_elements(prop="sizes", alpha=0.6,num=4) 
# labels = ['','','']
# legend = ax.legend(handles, labels, loc="upper left", title="Current Density (mA/cm$^2$)", ncols=6, borderpad=1., prop={'size':10})
# legend.get_title().set_fontsize('10')

ax.scatter(0, 105, s = (100*1)**0.6,facecolor='none',edgecolor='k',label='1')
ax.scatter(0, 105, s = (100*50)**0.6,facecolor='none',edgecolor='k',label='50')
ax.scatter(0, 105, s = (100*1000)**0.6,facecolor='none',edgecolor='k',label='1000 bar')
# ax.scatter(x4, 80, s = 576,facecolor='none',edgecolor='k')
# ax.scatter(x5, 80, s = 900,facecolor='none',edgecolor='k')
# ax2.annotate('1',(8690, 106),fontsize=10)
# ax2.annotate('100',(8730, 106),fontsize=10)
# ax2.annotate('1000 mA/cm$^2$',(8800, 106),fontsize=10)

# remove y-axis and spines
# ax.yaxis.set_visible(False)
ax.set_ylabel('NH$_3$ FE %')
ax.spines[["top", "right"]].set_visible(False)
ax.margins(y=0.1)
ax2.spines[['left',"right","top"]].set_visible(False)
ax3.spines[['left','right','top']].set_visible(False)

d = .02 # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((1-d-0.01,1+d+0.01), (-d,+d), **kwargs)
# ax.plot((1-d,1+d),(1-d,1+d), **kwargs)
kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
# ax2.plot((-d+0.02,+d-0.005), (1-d,1+d), **kwargs)
ax2.plot((-d+0.005,+d-0.01), (-d,+d), **kwargs)
ax2.tick_params(labelleft='off')
ax2.yaxis.set_tick_params(labelleft=False)
ax2.plot((1-d,1+d), (-d,+d), **kwargs)
# ax2.plot((1-d,1+d),(1-d,1+d), **kwargs)
kwargs.update(transform=ax3.transAxes)  # switch to the bottom axes
# ax3.plot((-d+0.015,+d-0.015), (1-d,1+d), **kwargs)
ax3.plot((-d+0.012,+d-0.015), (-d,+d), **kwargs)
ax3.tick_params(labelleft='off')
ax3.yaxis.set_tick_params(labelleft=False)
ax2.set_yticks([])
ax3.set_yticks([])
ax.set_xlim(-14300,-14200)
ax2.set_xlim(8300,8900)
ax3.set_xlim(17900,19500)

for a in [ax,ax2,ax3]:
    a.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
    a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    a.set_ylim(0,105)
    plt.setp(a.get_xticklabels(), rotation=30, ha="right",fontsize=12)

legend_elements = [
                   Line2D([0], [0], marker='o', color=pastels[0], label='Batch Cell',
                          markerfacecolor=pastels[0], markeredgecolor='k',markersize=10,linestyle='None'),
                          Line2D([0], [0], marker='o', color=pastels[1], label='Flow Cell - Parallel Plate',
                          markerfacecolor=pastels[1], markeredgecolor='k',markersize=10,linestyle='None'),
                          Line2D([0], [0], marker='o', color=pastels[2], label='Flow Cell - GDE',
                          markerfacecolor=pastels[2], markeredgecolor='k',markersize=10,linestyle='None')
                          ]
l = ax3.legend(handles=legend_elements,loc='upper left', bbox_to_anchor=(0.055, 1.23), ncol=3,fontsize=10,borderpad=1.2, labelspacing=0.01, handletextpad=0.01)
l.get_frame().set_edgecolor('k')

# legend_elements2 = [
#                    Line2D([0], [0], marker='o', label='1',
#                           markerfacecolor='none', markeredgecolor='k',markersize=2,linestyle='None'),
#                           Line2D([0], [0], marker='o', color='#ff6961', label='100',
#                           markerfacecolor='none', markeredgecolor='k',markersize=8,linestyle='None'),
#                           Line2D([0], [0], marker='o', color='#61ffb8', label='1000 mA/cm$^2$',
#                           markerfacecolor='none', markeredgecolor='k',markersize=15,linestyle='None')
#                           ]
l2 = ax.legend(loc='upper left', bbox_to_anchor=(-0.1, 1.23), ncol=3,fontsize=10,borderpad=1.2)
l2.get_frame().set_edgecolor('k')
# handles = l2.legendHandles
# hatches = ["o", "o", "o"]
# for i, handle in enumerate(handles):
#     handle.set_edgecolor("k") # set_edgecolors
#     handle.set_facecolor('none')
#     handle.set_hatch(hatches[i])
#     handle.set_markersize(6)

plt.savefig('Fig1-femax-pressure-all.tif')
plt.show()
plt.clf()
