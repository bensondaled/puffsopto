"""
Figure S3: no-opsin ctrls
"""
import matplotlib.pyplot as pl
from figure_panels import *

## Setup figure ##
fig_id = 'sfig3'
figw,figh = 6.,1.8
fig = pl.figure(fig_id, figsize=(figw,figh))

row_bottoms = [.2]
letter_ys = [.8, .6, 0, 0, 0]
letter_xs = [.02, .31, .01, .01, .01,.01,.01,.01,.01]
letters = ['a','b','', '', '','','','','']
#letters = [l.upper() for l in letters]

let_kw = dict(fontsize=9, fontname='Arial', weight='bold')

# boxes: row_id, x, w, h (w/h in inches)
boxes       =       [   
                         
                        [ 0, # 0
                          0.1,
                          .8 ,
                          1. ],

                        [ 0, # 1
                          0.35,
                          .8 ,
                          1. ],
                        
                        [ 0, # 2
                          0.6,
                          .8,
                          1. ],
                        
                        [ 0, # 3
                          0.85,
                          .8,
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
axs = fracs(axs, panel_id=0, manips=[2,3,4], labelmode=0, grp='ctl', show_ctl=False, show_signif=False)
axs = psys(axs, panel_id=1, manips=[0,2], easy=False, grp='ctl')
axs = psys(axs, panel_id=2, manips=[0,3], easy=False, grp='ctl')
axs = psys(axs, panel_id=3, manips=[0,4], easy=False, grp='ctl')

prettify_axes(axs)

pl.savefig('/Users/ben/Desktop/{}.pdf'.format(fig_id), dpi=500)
pl.close('all')
