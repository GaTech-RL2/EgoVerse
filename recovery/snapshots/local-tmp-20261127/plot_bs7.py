from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/coc/flash7/scratch/cxin34/cart-drawer-nola-fix-bs-20261125/reports')
x=np.load(R/'bs-example7-worldaligned-20261125/trajectories.npz')
t=x['time'];t=t-t[0];p=761
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,2,figsize=(13,8),constrained_layout=True)
a=axs[0,0]
a.plot(t,x['error_cm'][:,0],label='Right hand',c='#D55E00');a.plot(t,x['error_cm'][:,1],label='Left hand',c='#0072B2')
a.axhline(2,c='gray',ls='--',label='Max-error gate: 2 cm');a.scatter(t[p],x['error_cm'][p,0],c='#D55E00');a.set(title='Stored hand-center displacement after BS',ylabel='Distance (cm)',xlabel='Time (s)');a.legend(loc='upper left')
a=axs[0,1]
a.plot(t,x['h_old'][:,0,2],label='Before BS',c='#0072B2');a.plot(t,x['h_new'][:,0,2],label='After BS',c='#D55E00');a.axvline(t[p],c='gray',ls=':');a.set(title='Right hand height (stored FK, before joint smoothing)',ylabel='World height (m)',xlabel='Time (s)');a.legend();a.annotate('+84.1 cm at 12.68 s',xy=(t[p],x['h_new'][p,0,2]),xytext=(5,1.7),arrowprops={'arrowstyle':'->'})
a=axs[1,0]
a.plot(t,np.rad2deg(x['old_q'][:,13]),label='Before BS',c='#0072B2');a.plot(t,np.rad2deg(x['new_q'][:,13]),label='After BS',c='#D55E00');a.axhline(1,c='gray',ls='--',label='Upper joint limit: +1 deg');a.set(title='Right elbow angle in saved joint trajectory',ylabel='Angle (deg)',xlabel='Time (s)');a.legend(loc='upper right')
a=axs[1,1];a.axis('off')
a.text(0,1,'Drawer example: manifest index 7',fontsize=14,weight='bold',va='top')
a.text(0,.85,'818 frames, ~13.6 s; same source recording\n\nPeak stored hand displacement: 88.3 cm\n99.9th percentile: 87.0 cm (gate: 1 cm)\n575 / 818 frames exceed 2 cm in either hand\n\nAt peak: right elbow ~-150 deg → +1 deg\nRight-hand height: 0.980 m → 1.821 m\n\nBS changes base motion, then re-solves torso/arms.\nThis rejected example shows a major arm-solution change.',va='top',linespacing=1.5)
fig.suptitle('Before / after base smoothing: rejected Drawer segment',fontsize=17,weight='bold')
fig.savefig(R/'bs-example7-comparison.png',dpi=170)
