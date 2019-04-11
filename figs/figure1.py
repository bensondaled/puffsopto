"""
Figure 1: task & full-cue-period opto
"""
import matplotlib.pyplot as pl
from figure_panels import *

## Setup figure ##
fig_id = 'fig1'
figw,figh = 6.8,3.2
fig = pl.figure(fig_id, figsize=(figw,figh))

row_bottoms = [.65, .12]
letter_ys = [.96, .46]
letter_xs = [.01, .65, .01, 0.4, 0., .8]
letters = ['a','b','c', 'd', '', 'e']
#letters = [l.upper() for l in letters]

let_kw = dict(fontsize=9, fontname='Arial', weight='bold')

# boxes: row_id, x, w, h (w/h in inches)
boxes       =       [   
                        [  0, # 0
                           0.25,
                          2.2,
                          .8 ],
                        
                        [  0, # 0
                           0.7,
                          .9,
                          .9 ],
                        
                        [  1, # 1
                           0.1,
                          1.6,
                          1. ],

                        [  1, # 2
                          0.45,
                          .8,
                          1. ],
                        
                        [  1, # 3
                          0.65,
                          .8,
                          1. ],
                        
                        [  1, # 4
                          0.85,
                          .8,
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
axs = task_structure(axs, panel_id=0)
axs = heatmap(axs, panel_id=1)
axs = fracs(axs, panel_id=2, manips=[2,3,4], labelmode=0, show_ctl=True)
axs = psys(axs, panel_id=3, manips=[0,2], easy=False)
axs = psys(axs, panel_id=4, manips=[0,3,4])
axs = regs(axs, panel_id=5, manips=[0,2,3,4], ylab=True, ylim=(-.05,.3), xlab=True)

prettify_axes(axs)

pl.savefig('/Users/ben/Desktop/{}.pdf'.format(fig_id), dpi=500)
pl.close('all')
