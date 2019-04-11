"""
Figure S5: by-subj latencies
"""
import matplotlib.pyplot as pl
from figure_panels import *

## Setup figure ##
fig_id = 'sfig5'
figw,figh = 4.,1.3
fig = pl.figure(fig_id, figsize=(figw,figh))

row_bottoms = [.04]
letter_ys = [.8, .6, 0, 0, 0]
letter_xs = [.01, .21, .01, .01, .8,.01,.01,.01,.01]
letters = ['','','', '', '','','','','']
#letters = [l.upper() for l in letters]

let_kw = dict(fontsize=9, fontname='Arial', weight='bold')

# boxes: row_id, x, w, h (w/h in inches)
boxes       =       [   
                        [ 0, # 0
                          0.1,
                          3.3,
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
axs = latency_bysubj(axs, panel_id=0, manips=[0,234])

prettify_axes(axs)

pl.savefig('/Users/ben/Desktop/{}.pdf'.format(fig_id), dpi=500)
pl.close('all')
