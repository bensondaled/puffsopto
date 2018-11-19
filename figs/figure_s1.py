"""
Figure S1: baseline behav
"""
import matplotlib.pyplot as pl
from figure_panels import *

## Setup figure ##
fig_id = 'sfig1'
figw,figh = 6.,1.7
fig = pl.figure(fig_id, figsize=(figw,figh))

row_bottoms = [.2]
letter_ys = [.9, .63, 0, 0, 0]
letter_xs = [.06, .36, .76,]
letters = ['a','b','c',]
#letters = [l.upper() for l in letters]

let_kw = dict(fontsize=9, fontname='Arial', weight='bold')

# boxes: row_id, x, w, h (w/h in inches)
boxes       =       [   
                         
                        [ 0, # 0
                          0.08,
                          1. ,
                          1. ],

                        [ 0, # 1
                          0.38,
                          1. ,
                          1. ],
                        
                        [ 0, # 2
                          0.78,
                          1.,
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
axs = psy_bsl(axs, panel_id=0)
axs = heatmap(axs, panel_id=1)
axs = regr_bsl(axs, panel_id=2)

prettify_axes(axs)

pl.savefig('/Users/ben/Desktop/{}.pdf'.format(fig_id), dpi=500)
pl.close('all')
