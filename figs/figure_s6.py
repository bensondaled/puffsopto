"""
Figure S6: simulation for ddm
"""
import matplotlib.pyplot as pl
from figure_panels import *

## Setup figure ##
fig_id = 'sfig6'
figw,figh = 6.,2.2
fig = pl.figure(fig_id, figsize=(figw,figh))

row_bottoms = [.18]
letter_ys = [.97, .63, 0, 0, 0]
letter_xs = [.01, .01, .01, .01, .01,.01,.01,.01,.01]
letters = ['','','', '', '','','','','']
letters = [l.upper() for l in letters]

let_kw = dict(fontsize=9, fontname='Arial', weight='bold')

# boxes: row_id, x, w, h (w/h in inches)
boxes       =       [   
                         
                        [ 0, # 0
                          0.2,
                          .8 ,
                          1.1 ],

                        [ 0, # 1
                          0.4,
                          .8 ,
                          1.1 ],
                        
                        [ 0, # 2
                          0.6,
                          .8 ,
                          1.1 ],
                        
                        [ 0, # 3
                          0.8,
                          .8 ,
                          1.1 ],
                        
                        
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
axs = ddm_params_julia_bootstrap(axs, panel_id=0, param_idx=0, manips=['0_sub6000flip25',0], yticklabels='text')
axs = ddm_params_julia_bootstrap(axs, panel_id=1, param_idx=1, manips=['0_sub6000flip25',0])
axs = ddm_params_julia_bootstrap(axs, panel_id=2, param_idx=2, manips=['0_sub6000flip25',0])
axs = ddm_params_julia_bootstrap(axs, panel_id=3, param_idx=4, manips=['0_sub6000flip25',0])

prettify_axes(axs)

pl.savefig('/Users/ben/Desktop/{}.pdf'.format(fig_id), dpi=500)
pl.close('all')
