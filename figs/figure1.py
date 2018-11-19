"""
Figure 1: task & full-cue-period opto
"""
import matplotlib.pyplot as pl
from figure_panels import *

## Setup figure ##
fig_id = 'fig1'
figw,figh = 3.35,3.
fig = pl.figure(fig_id, figsize=(figw,figh))

row_bottoms = [.60, .12]
letter_ys = [.96, .48]
letter_xs = [.01, .55, .01, 0, .69]
letters = ['a','b','c', '', 'd', '']
#letters = [l.upper() for l in letters]

let_kw = dict(fontsize=9, fontname='Arial', weight='bold')

# boxes: row_id, x, w, h (w/h in inches)
boxes       =       [   
                        [  0, # 0
                           0.07,
                          1.25,
                          .6 ],
                        
                        [  0, # 1
                           0.7,
                          .9,
                          1. ],

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
axs = fracs(axs, panel_id=1, manips=[2,3,4], labelmode=0, show_ctl=False)
axs = psys(axs, panel_id=2, manips=[0,2], easy=True)
axs = psys(axs, panel_id=3, manips=[0,3,4])
axs = regs(axs, panel_id=4, manips=[0,2,3,4], ylab=True, ylim=(-.05,.3), xlab=True)

prettify_axes(axs)

pl.savefig('/Users/ben/Desktop/{}.pdf'.format(fig_id), dpi=500)
pl.close('all')
