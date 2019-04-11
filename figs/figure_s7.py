"""
Figure S7: logistic regression model, by subj
"""
import matplotlib.pyplot as pl
from figure_panels import *

## Setup figure ##
fig_id = 'sfig7'
figw,figh = 7.2,5.
fig = pl.figure(fig_id, figsize=(figw,figh))

row_bottoms = [.85,.7,.55,.4,.25,.1]
letter_ys = [.8, .6, 0, 0, 0, 0, 0, 0]
letter_xs = [.01, .21, .01, .01, .8,.01,.01,.01,.01]
letters = ['','','', '', '','','','','']
#letters = [l.upper() for l in letters]

let_kw = dict(fontsize=9, fontname='Arial', weight='bold')

# boxes: row_id, x, w, h (w/h in inches)
boxes       =       [   
                         
                        [ 0, # 1
                          0.07,
                          6. ,
                          .5 ],
                        
                        [ 1, # 2
                          0.07,
                          6.,
                          .5 ],
                        
                        [ 2, # 3
                          0.07,
                          6.,
                          .5 ],
                        
                        [ 3, # 4
                          0.07,
                          6.,
                          .5 ],
                        
                        [ 4, # 5
                          0.07,
                            6.,
                          .5 ],
                        
                        [ 5, # 6
                          0.07,
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
#axs = regs_xval(axs, panel_id=0)
axs = reg_by_subj(axs, panel_id=0, manip=2, title=True)
axs = reg_by_subj(axs, panel_id=1, manip=3)
axs = reg_by_subj(axs, panel_id=2, manip=4)
axs = reg_by_subj(axs, panel_id=3, manip=5, ylab=True)
axs = reg_by_subj(axs, panel_id=4, manip=6)
axs = reg_by_subj(axs, panel_id=5, manip=7, xticklabs=True)

prettify_axes(axs)

pl.savefig('/Users/ben/Desktop/{}.pdf'.format(fig_id), dpi=500)
pl.close('all')
