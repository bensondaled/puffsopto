"""
Figure S4: opto behaviour extra details
"""
import matplotlib.pyplot as pl
from figure_panels import *

## Setup figure ##
fig_id = 'sfig4'
figw,figh = 3.35,3.8
fig = pl.figure(fig_id, figsize=(figw,figh))

row_bottoms = [.78, .45, .12]
letter_ys = [.98, .6, .3, 0, 0, 0, 0, 0]
letter_xs = [.01, .21, .01, .01, .8,.01,.01,.01,.01]
letters = ['a','','', 'b', '','','c','','']
#letters = [l.upper() for l in letters]

let_kw = dict(fontsize=9, fontname='Arial', weight='bold')

# boxes: row_id, x, w, h (w/h in inches)
boxes       =       [   
                         
                        [ 0, # 0
                          0.17,
                          .6,
                          .6 ],
                        
                        [ 0, # 1
                          0.44,
                          .6,
                          .6 ],
                        
                        [ 0, # 2
                          0.71,
                          .6,
                          .6 ],
                        
                        [ 1, # 3
                          0.17,
                          .6,
                          .6 ],
                        
                        [ 1, # 4
                          0.44,
                          .6,
                          .6 ],
                        
                        [ 1, # 5
                          0.71,
                          .6,
                          .6 ],
                        
                        [ 2, # 6
                          0.17,
                          .6,
                          .6 ],
                        
                        [ 2, # 7
                          0.44,
                          .6,
                          .6 ],
                        
                        [ 2, # 8
                          0.71,
                          .6,
                          .6 ],
                        
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
axs = fracs_scatter(axs, panel_id=0, manips=[0,2], xlab=False)
axs = fracs_scatter(axs, panel_id=1, manips=[0,3], ylab=False)
axs = fracs_scatter(axs, panel_id=2, manips=[0,4], ylab=False, xlab=False)
axs = heatmap(axs, panel_id=3, manip=2, cbar=False, xlab=False)
axs = heatmap(axs, panel_id=4, manip=3, cbar=False, ylab=False)
axs = heatmap(axs, panel_id=5, manip=4, cbar=True, ylab=False, xlab=False)
axs = regs_rl(axs, panel_id=6, manips=[2], ylab=True, annotations=True)
axs = regs_rl(axs, panel_id=7, manips=[3])
axs = regs_rl(axs, panel_id=8, manips=[4])

prettify_axes(axs)

pl.savefig('/Users/ben/Desktop/{}.pdf'.format(fig_id), dpi=500)
pl.close('all')
