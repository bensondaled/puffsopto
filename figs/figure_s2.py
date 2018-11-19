"""
Figure S2: histology & ephys
"""
import matplotlib.pyplot as pl
from figure_panels import *

## Setup figure ##
fig_id = 'sfig2'
figw,figh = 3.5,2.5,
fig = pl.figure(fig_id, figsize=(figw,figh))

row_bottoms = [.5, .08]
letter_ys = [.96, .45, 0, 0, 0]
letter_xs = [.01, .01, .01, .01, .01,.01,.01,.01,.01]
letters = ['a','b','', '', '','','','','']
#letters = [l.upper() for l in letters]

let_kw = dict(fontsize=9, fontname='Arial', weight='bold')

# boxes: row_id, x, w, h (w/h in inches)
boxes       =       [   
                         
                        [ 0, # 0
                          0.3,
                          1.4 ,
                          1.4 ],

                        [ 1, # 1
                          0.15,
                          2.5 ,
                          1.1 ],
                        
                       # [ 2, # 1
                       #   0.15,
                       #   2.5 ,
                       #   1.1 ],
                        
                        
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
axs = ephys(axs, panel_id=1, cell_type='pc')
#axs = ephys(axs, panel_id=2, cell_type='dcn')

prettify_axes(axs)

pl.savefig('/Users/ben/Desktop/{}.pdf'.format(fig_id), dpi=500)
pl.close('all')
