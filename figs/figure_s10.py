"""
Figure S10: whisker tracking
"""
import matplotlib.pyplot as pl
from figure_panels import *

## Setup figure ##
fig_id = 'sfig10'
figw,figh = 7.,7.
fig = pl.figure(fig_id, figsize=(figw,figh))

row_bottoms = [.8, .68, .56, .44, .32, .2, .08]
letter_ys = [.8, .6, 0, 0, 0, 0, 0, 0]
letter_xs = [.01, .21, .01, .01, .8,.01,.01,.01,.01]
letters = ['','','', '', '','','','','']
#letters = [l.upper() for l in letters]

let_kw = dict(fontsize=9, fontname='Arial', weight='bold')

# boxes: row_id, x, w, h (w/h in inches)
boxes       =       [   
                         
                        [ 0, # 0
                          0.09,
                          6. ,
                          .5 ],

                        [ 1, # 1
                          0.09,
                          6. ,
                          .5 ],
                        
                        [ 2, # 2
                          0.09,
                          6.,
                          .5 ],
                        
                        [ 3, # 3
                          0.09,
                          6.,
                          .5 ],
                        
                        [ 4, # 4
                          0.09,
                          6.,
                          .5 ],
                        
                        [ 5, # 5
                          0.09,
                            6.,
                          .5 ],
                        
                        [ 6, # 6
                          0.09,
                          6.,
                          .5 ],
                        
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
axs = whisker_tracking(axs, panel_id=0, manip=0)
axs = whisker_tracking(axs, panel_id=1, manip=2)
axs = whisker_tracking(axs, panel_id=2, manip=3)
axs = whisker_tracking(axs, panel_id=3, manip=4, ylab=True)
axs = whisker_tracking(axs, panel_id=4, manip=5)
axs = whisker_tracking(axs, panel_id=5, manip=6)
axs = whisker_tracking(axs, panel_id=6, manip=7, xticks=True)

prettify_axes(axs)

pl.savefig('/Users/ben/Desktop/{}.pdf'.format(fig_id), dpi=500)
pl.close('all')
