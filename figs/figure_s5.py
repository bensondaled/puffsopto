"""
Figure S5: delay perturbation
"""
import matplotlib.pyplot as pl
from figure_panels import *

## Setup figure ##
fig_id = 'sfig5'
figw,figh = 1.5,1.5
fig = pl.figure(fig_id, figsize=(figw,figh))

row_bottoms = [.2,.1]
letter_ys = [.97, .63, 0, 0, 0]
letter_xs = [.01, .01, .01, .01, .01,.01,.01,.01,.01]
letters = ['','','', '', '','','','','']
letters = [l.upper() for l in letters]

let_kw = dict(fontsize=9, fontname='Arial', weight='bold')

# boxes: row_id, x, w, h (w/h in inches)
boxes       =       [   
                         
                       # [ 0, # 0
                       #   0.15,
                       #   .8 ,
                       #   1. ],
                        
                       # [ 1, # 2
                       #   0.15,
                       #   .8,
                       #   .3 ],

                       # [ 0, # 1
                       #   0.8,
                       #   .8 ,
                       #   1. ],
                        
                        [ 0, # 1
                          .25,
                          1. ,
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
#axs = fracs_simple(axs, panel_id=0, manips=[0,5,6,7])
#axs = light_delivery_schematic(axs, panel_id=1, manips=[5,6,7], exclude_phases=[2,3], labelmode=3)
axs = regs(axs, panel_id=0, manips=[0,8], ylab=True, shade=False, xlab=True, ylim=(.0,.3), annotate=True)
#axs = fracs_simple(axs, panel_id=3, manips=[0,8], dp_title=True, xlabs=True)

prettify_axes(axs)

pl.savefig('/Users/ben/Desktop/{}.pdf'.format(fig_id), dpi=500)
pl.close('all')
