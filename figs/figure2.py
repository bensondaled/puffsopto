"""
Figure 2: full-cue-period opto
"""
import matplotlib.pyplot as pl
from figure_panels import *

## Setup figure ##
fig_id = 'fig2'
figw,figh = 3.35,4
fig = pl.figure(fig_id, figsize=(figw,figh))

row_bottoms = [.82, .48, .08]
letter_ys = [.97, .74, .34]
letter_xs = [.01, .38, .01, .42, .01, .01, .43, .74]
letters = ['A','B','C', '', '', 'D', 'E','F']
letters = [l.upper() for l in letters]

let_kw = dict(fontsize=9, fontname='Arial', weight='bold')

# boxes: row_id, x, w, h (w/h in inches)
boxes       =       [   
                        [  0, # 0
                           0.01,
                          1.1,
                          .6 ],
                        
                        [  0, # 1
                           0.48,
                          1.6,
                          .83 ],

                        [  1, # 2
                          0.08,
                          .75,
                          1. ],
                        
                        [  1, # 3
                          0.45,
                          .6,
                          1. ],
                        
                        [  1, # 4
                          0.78,
                          .6,
                          1. ],
                        
                        [  2, # 5
                          0.15,
                          .74,
                          1. ],
                        
                        [  2, # 6
                          0.53,
                          .55,
                          1. ],
                        
                        [  2, # 7
                          0.85,
                          .45,
                          1. ],
                         
                        ]

# draw letters
for lx,letter,(row_id,*_) in zip(letter_xs, letters, boxes):
    fig.text(lx, letter_ys[row_id], letter, **let_kw)
# convert panel w/h to fractions
boxes = [[b[0], b[1], b[2]/figw, b[3]/figh] for b in boxes]
# convert row_ids to y positions
boxes = [[b[1], row_bottoms[b[0]], b[2], b[3]] for b in boxes]
# draw axes
axs = [fig.add_axes(box) for box in boxes]

## Draw panels
axs = ai27d_histology(axs, panel_id=0)
axs = ephys(axs, panel_id=1)
axs = psys(axs, panel_id=2, manips=[0,2], easy=True)
axs = psys(axs, panel_id=3, manips=[0,3,4])
axs = psys(axs, panel_id=4, manips=[0,2,3,4], easy=False, grp='ctl')
axs = fracs(axs, panel_id=5, manips=[2,3,4], labelmode=0, show_ctl=True)
axs = regs(axs, panel_id=6, manips=[0,2,3,4], ylab=True, ylim=(-.05,.3), xlab=True)
axs = latency(axs, panel_id=7)

prettify_axes(axs)

pl.savefig('/Users/ben/Desktop/{}.pdf'.format(fig_id), dpi=500)
pl.close('all')
