"""
Figure 3: DDM
"""
import matplotlib.pyplot as pl
from figure_panels import *

## Setup figure ##
fig_id = 'fig3'
figw,figh = 6.,3.8 #4.2
fig = pl.figure(fig_id, figsize=(figw,figh))

row_bottoms = [.6, .04]
letter_ys = [.97, .37, 0, 0, 0]
letter_xs = [.04] * 10
letters = ['a','','', '', 'b','','','','']
#letters = [l.upper() for l in letters]

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
                        
                        [ 1, # 4
                          0.2,
                          3.5 ,
                          .95 ],
                        
                        #[ 1, # 5
                        #  0.5,
                        #  .6 ,
                        #  .9 ],
                        
                        #[ 1, # 6
                        #  0.8,
                        #  .6 ,
                        #  .9 ],
                        
                        
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

## Draw panels -- comments are for variations
axs = ddm_params_julia_bootstrap(axs, panel_id=0, param_idx=0, yticklabels=True, )
        #manips=['8_nobias','234_nobias','0_nobias'], reference='0_nobias')
axs = ddm_params_julia_bootstrap(axs, panel_id=1, param_idx=1,)
        #manips=['8_nobias','234_nobias','0_nobias'], reference='0_nobias')
axs = ddm_params_julia_bootstrap(axs, panel_id=2, param_idx=2,)
        #manips=['8_nobias','234_nobias','0_nobias'], reference='0_nobias')
axs = ddm_params_julia_bootstrap(axs, panel_id=3, param_idx=4,)
#        manips=['8_nobias','234_nobias','0_nobias'], reference='0_nobias')

#axs = ddm_params_julia_bootstrap(axs, panel_id=3, param_idx=3,
#        manips=['8_nobias','234_nobias','0_nobias'], reference='0_nobias')

axs = ddm_simulation(axs, panel_id=4)

#axs = likelihood_landscape_julia(axs, panel_id=4, param_idxs=[4,2], manip=234, ylab=True, xlab=True)
#axs = likelihood_landscape_julia(axs, panel_id=5, param_idxs=[4,2], manip=8, yticklabs=False, cbar=True, xlab=True)

prettify_axes(axs)

pl.savefig('/Users/ben/Desktop/{}.pdf'.format(fig_id), dpi=500)
pl.close('all')
