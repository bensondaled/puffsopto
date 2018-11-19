"""
Figure S4: simulation for regressions
"""
import matplotlib.pyplot as pl
from figure_panels import *

## Setup figure ##
fig_id = 'sfig4'
figw,figh = 6.,6
fig = pl.figure(fig_id, figsize=(figw,figh))

row_bottoms = [.74, .52, .3, .08]
letter_ys = [.97, .63, 0, 0, 0]
letter_xs = [.01]*16
letters = ['']*16
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
                        
                        [ 1, # 0
                          0.2,
                          .8 ,
                          1.1 ],

                        [ 1, # 1
                          0.4,
                          .8 ,
                          1.1 ],
                        
                        [ 1, # 2
                          0.6,
                          .8 ,
                          1.1 ],
                        
                        [ 1, # 3
                          0.8,
                          .8 ,
                          1.1 ],
                        
                        [ 2, # 0
                          0.2,
                          .8 ,
                          1.1 ],

                        [ 2, # 1
                          0.4,
                          .8 ,
                          1.1 ],
                        
                        [ 2, # 2
                          0.6,
                          .8 ,
                          1.1 ],
                        
                        [ 2, # 3
                          0.8,
                          .8 ,
                          1.1 ],
                        
                        [ 3, # 0
                          0.2,
                          .8 ,
                          1.1 ],

                        [ 3, # 1
                          0.4,
                          .8 ,
                          1.1 ],
                        
                        [ 3, # 2
                          0.6,
                          .8 ,
                          1.1 ],
                        
                        [ 3, # 3
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
axs = impairment_simulation(axs, panel_id=0, agent=0, man=0)
axs = impairment_simulation(axs, panel_id=1, agent=0, man=5)
axs = impairment_simulation(axs, panel_id=2, agent=0, man=6)
axs = impairment_simulation(axs, panel_id=3, agent=0, man=7)
axs = impairment_simulation(axs, panel_id=4, agent=1, man=0)
axs = impairment_simulation(axs, panel_id=5, agent=1, man=5)
axs = impairment_simulation(axs, panel_id=6, agent=1, man=6)
axs = impairment_simulation(axs, panel_id=7, agent=1, man=7)
axs = impairment_simulation(axs, panel_id=8, agent=2, man=0)
axs = impairment_simulation(axs, panel_id=9, agent=2, man=5)
axs = impairment_simulation(axs, panel_id=10, agent=2, man=6)
axs = impairment_simulation(axs, panel_id=11, agent=2, man=7)
axs = impairment_simulation(axs, panel_id=12, agent=3, man=0)
axs = impairment_simulation(axs, panel_id=13, agent=3, man=5)
axs = impairment_simulation(axs, panel_id=14, agent=3, man=6)
axs = impairment_simulation(axs, panel_id=15, agent=3, man=7)

prettify_axes(axs)

pl.savefig('/Users/ben/Desktop/{}.pdf'.format(fig_id), dpi=500)
pl.close('all')
