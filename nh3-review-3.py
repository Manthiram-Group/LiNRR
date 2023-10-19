import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
import pandas as pd
import matplotlib.dates as mdates
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D


'''Figure styles'''
mpl.rc('font',weight='bold',size=24, **{'family':'sans-serif','sans-serif':['Arial']})
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
# pastels = ['#61a8ff','#ff6961','#61ffb8','#ffdfba','#baffc9','#ffffba']
# pastels = ['#eb861e','#943fa6','#63c5b5','#dee000','#56b2a1','#884299','#b76325']
pastels = ['#00AEEF','#E0001B','#008D48']

def jitt(arr):
    stdev = .005 * (max(arr) - min(arr))
    return arr + np.random.randn(len(arr)) * stdev

colormap = {'Autoclave':pastels[0],'Compartment cell':pastels[1],'GDE cell':pastels[2],'Glass cell':pastels[0]}
markers = {'LiNTf2':"s",'LiBF4':"o",'LiClO4':"^"}
# markers = {'Autoclave':"o",'Compartment cell':"s",'GDE cell':"^",'Glass cell':"o"}
# colormap_salts = {'LiNTf2':pastels[0],'LiBF4':pastels[1],'LiClO4':pastels[2]}

file_pre = 'N2_NH3_review' 
df = pd.read_csv(file_pre+'.csv')
df = df[8:]
# df = df[df['Fig1']=='y']
df = df[df['Electrolyte Salt'].isin(['LiClO4','LiBF4','LiNTf2'])].reset_index()
dates = [datetime.strptime(d, "%Y-%m-%d") for d in df['Date Published']]
fig, ax = plt.subplots(figsize=(7,4),layout='constrained',dpi=300)

'''Figure 2'''
df['color'] = df['Cell Type'].apply(lambda x: colormap[x])
df['marker']=df['Electrolyte Salt'].apply(lambda x: markers[x])
# df['color'] = df['Electrolyte Salt'].apply(lambda x: colormap_salts[x])
for i in range(8,len(df['Cell Type'])):
        sc1 = ax.scatter((df['Rate (nmol/s-cm2)'][i]), (df['FE%'][i]), marker=df['marker'][i],
        c=df['color'][i], s=(100*df['N2 Pressure (atm/bar)'][i])**0.6,edgecolor='k',linewidth=0.5)
        # sc1 = ax.scatter((df['Current Density (mA/cm2)'][i])*0.01*df['FE%'][i], (df['FE%'][i]), marker=df['marker'][i],
        # c=df['color'][i], s=np.sqrt(200*df['N2 Pressure (atm/bar)'][i]),edgecolor='k',linewidth=0.5)
# print(10*np.sqrt((df['Current Density (mA/cm2)'])*0.01*df['FE%']))
# plt.colorbar(s c,label='Log Current Density (mA/cm$\mathregular{^2}$)')
# format x-axis with 4-month intervals
# ax.legend(*sc.legend_elements("sizes", num=6))

# handles, labels = sc.legend_elements(prop="sizes", alpha=0.6,num=4) 
# labels = ['','','']
# legend = ax.legend(handles, labels, loc="upper left", title="Current Density (mA/cm$^2$)", ncols=6, borderpad=1., prop={'size':10})
# legend.get_title().set_fontsize('10')

ax.scatter(0, 100, s = 100**0.6,facecolor='none',edgecolor='k',label='1 bar, $\mathregular{LiBF_4}$',marker='o')
ax.scatter(0, 100, s = (100*10)**0.6,facecolor='none',edgecolor='k',label='10 bar',marker='o')
ax.scatter(0, 100, s = (100*50)**0.6,facecolor='none',edgecolor='k',label='50 bar',marker='o')
ax.scatter(0, 100, s = (100*1)**0.6,facecolor='none',edgecolor='k',label='1 bar, $\mathregular{LiNTf_2}$',marker='s')
ax.scatter(0, 100, s = (100*10)**0.6,facecolor='none',edgecolor='k',label='10 bar',marker='s')
ax.scatter(0, 100, s = (100*50)**0.6,facecolor='none',edgecolor='k',label='50 bar',marker='s')
ax.scatter(0, 100, s = (100*1)**0.6,facecolor='none',edgecolor='k',label='1 bar, $\mathregular{LiClO_4}$',marker='^')
ax.scatter(0, 100, s = (100*10)**0.6,facecolor='none',edgecolor='k',label='10 bar',marker='^')
ax.scatter(0, 100, s = (100*50)**0.6,facecolor='none',edgecolor='k',label='50 bar',marker='^')
# ax.scatter(x4, 80, s = 576,facecolor='none',edgecolor='k')
# ax.scatter(x5, 80, s = 900,facecolor='none',edgecolor='k')
# ax2.annotate('1',(8690, 106),fontsize=10)
# ax2.annotate('100',(8730, 106),fontsize=10)
# ax2.annotate('1000 mA/cm$^2$',(8800, 106),fontsize=10)

# remove y-axis and spines
# ax.yaxis.set_visible(False)
ax.set_ylabel('NH$_3$ FE%')

ax.set_xlabel('Rate (nmol/s-cm$^2$)')
# ax.set_xlabel('NH$_3$ Partial Current Density (mA/cm$^2$)')
ax.spines[["top", "right"]].set_visible(False)
ax.margins(y=0.1)
# ax.set_xlim(17900,19500)
ax.set_xscale('log')

for a in [ax]:
#     a.xaxis.set_major_locator(mdates.MonthLocator(interval=12))
#     a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    a.set_ylim(0,105)
#     plt.setp(a.get_xticklabels(), rotation=30, ha="right",fontsize=12)

# legend_elements = [Line2D([0], [0], label='LiNTf$_2$', marker = 's',
#                            markeredgecolor='k',color='None',linewidth=3),
#                           Line2D([0], [0],  label='LiBF$_4$', marker = 'o',
#                           markeredgecolor='k',color='None',linewidth=3),
#                           Line2D([0], [0],label='LiClO$_4$', marker = '^',
#                            markeredgecolor='k',color='None',linewidth=3)
#                           ]
# l = ax.legend(handles=legend_elements,loc='upper left', bbox_to_anchor=(0., 0.9), ncol=3,fontsize=8,borderpad=2.25)
# l.get_frame().set_edgecolor('k')

# legend_elements2 = [
#                    Line2D([0], [0], marker='o', markerfacecolor='none',label='Batch Cell',
#                            markeredgecolor='k',markersize=10),
#                           Line2D([0], [0], marker='s', markerfacecolor='none', label='Flow Cell - Parallel Plate',
#                         markeredgecolor='k',markersize=10),
#                           Line2D([0], [0], marker='^', markerfacecolor='none', label='Flow Cell - GDE',
#                            markeredgecolor='k',markersize=10)
#                           ]


# for i, marker in enumerate(markers):
#     for j, size in enumerate(sizes):
#       ax.scatter(0, 100, marker=marker, s=size, label=labels[j])

l = ax.legend(ncol=3, fontsize=10,borderpad=0.5, labelspacing=0.3, handletextpad=0.3,loc='upper left',bbox_to_anchor=(0., 1.25))
# l2= ax.legend(handles=legend_elements2,loc='upper left', bbox_to_anchor=(0., 1.), ncol=3,fontsize=8,borderpad=0.56)
l.get_frame().set_edgecolor('k')

# l3 = ax.legend(loc='upper left', bbox_to_anchor=(0., 0.9), ncol=3,fontsize=8,borderpad=0.56,handletextpad=0.1)
# l3.get_frame().set_edgecolor('k')

# plt.gca().add_artist(l)
# plt.gca().add_artist(l2)
plt.savefig('Fig3-salt-pressure.tif')
plt.show()
plt.clf()
