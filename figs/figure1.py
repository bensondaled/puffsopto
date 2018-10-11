"""
Figure 1: task
"""
import matplotlib.pyplot as pl
from figure_panels import *

## Setup figure ##
fig_id = 'fig1'
figw,figh = 3.35,2.2
fig = pl.figure(fig_id, figsize=(figw,figh))

row_bottoms = [.64, .11]
letter_ys = [.94, .47, .26]
letter_xs = [.01, .01, .33, .72]
letters = ['A','B','C', 'D', '','']
letters = [l.upper() for l in letters]

let_kw = dict(fontsize=9, fontname='Arial', weight='bold')

# boxes: row_id, x, w, h (w/h in inches)
boxes       =       [   
                        [  0, # 0
                            0.08,
                          2.8,
                          .6 ],

                        [  1, # 1
                          0.07,
                          .65,
                          .65],
                        
                        [  1, # 2
                          0.4,
                          .6,
                          .65 ],
                        
                        [  1, # 3
                          0.8,
                          .65,
                          .65 ],
                        
                         
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
axs = psy_bsl(axs, panel_id=1)
axs = heatmap(axs, panel_id=2)
axs = regr_bsl(axs, panel_id=3)

prettify_axes(axs)

pl.savefig('/Users/ben/Desktop/{}.pdf'.format(fig_id), dpi=500)
pl.close('all')
