# imports
import numpy as np
import pandas as pd
from routines import load_data, regress, psy, confidence, blend, agent, psy_fit, load_subject_key, frac_correct, df_to_h5, downsample_bins, light_triggered_regression, match_toi_tt, heatmap, reverse_correlation, response_latency, np_to_h5
from ddm import format_for_ddm, fit_ddm, get_ll_map, deverett_to_brody_format
from scipy.io import savemat

# global data params
cohort = [10,11,12,13]
dest_path = 'fig_data.h5'

# subject ids
subj_exp,subj_ctl = load_subject_key()

# analysis params
reg_kw = dict(nbins=3, only_dur=3.8, error='se')
psy_kw = dict(bins=3)
only_dur = 3.8
only_lev = None
usub = subj_exp
uman = [0,2,3,4,5,6,7,8]
mergers = {8:[8,10,18]}

# load data
trials,_ta,phases,density,_tt = load_data(cohort)

# define trials of interest mtrials
mtrials = trials.copy()
# merge fields
for m,m0 in mergers.items():
    mtrials.manipulation[mtrials.manipulation.isin(m0)] = m

# restrict sessions to those with manipulations being analyzed
uman = np.array(uman)
has_manip = mtrials.groupby('session').apply(lambda x: np.any(x.manipulation.isin(uman[uman!=0])))
manip_seshs = has_manip.index[has_manip.values]
mtrials = mtrials[mtrials.session.isin(manip_seshs)]

# restrict subjects
mtrials = mtrials[mtrials.subj.isin(usub)]

# restrict duration and level
if only_dur is not None:
    mtrials = mtrials[mtrials.dur==only_dur]
if only_lev is not None:
    mtrials = mtrials[mtrials.level==only_lev]

mtrials = mtrials.sort_values('uid')

# --- Part 1, basic behavioral data
tr = trials.copy()
tr = tr[tr.manipulation==0] 

# psychometrics & regs of subjs and meta-subj
for subj in tr.subj.unique():
    ti = tr[tr.subj==subj]
    densityi = density[density.uid.isin(ti.uid.values)].copy().sort_values('uid')
        
    # psy
    ps = psy(ti, bins=4)
     
    # reg
    rkw = reg_kw.copy()
    rkw['nbins'] = 3
    regr = regress(densityi, **rkw)

    subj = int(subj)
    with h5py.File(dest_path) as h:
        df_to_h5(h, data=ps, grp=f'psy_bsl_subj{subj}')
        df_to_h5(h, data=regr, grp=f'regr_bsl_subj{subj}')

ps = psy(tr)
popt,fit = psy_fit(ps.index, ps['mean'])
fit = pd.DataFrame(fit, columns=['mean'])
densityi = density[density.uid.isin(tr.uid.values)].copy().sort_values('uid')
rkw = reg_kw.copy()
rkw['error'] = '95ci'
rkw['nbins'] = 3
regr = regress(densityi, **rkw)
hm = heatmap(tr, at_least_ntrials=25)

density_shuf = densityi.copy()
ch = density_shuf.choice.values.copy()
rkw = reg_kw.copy()
rkw['error'] = '95ci'
shufs = []
for i in range(50):
    np.random.shuffle(ch)
    density_shuf.loc[:,'choice'] = ch
    regr_shuf = regress(density_shuf, **rkw)
    shufs.append(regr_shuf)
regr_shuf.loc[:,'weight'] = np.mean([s.weight.values for s in shufs], axis=0)
regr_shuf.loc[:,'yerr'] = np.mean([s.yerr.values for s in shufs], axis=0)

with h5py.File(dest_path) as h:
    df_to_h5(h, data=fit, grp='psy_bsl_meta')
    df_to_h5(h, data=regr, grp='regr_bsl_meta')
    df_to_h5(h, data=regr_shuf, grp='regr_bsl_shuf')
    if 'meta_heatmap' in h:
        del h['meta_heatmap']
    h.create_dataset('meta_heatmap', data=hm, compression='lzf')

# --- Part 2, send manipulation data to figure_data file for figure creation
assert reg_kw['nbins']==3
assert reg_kw['only_dur']==3.8
assert reg_kw['error']=='se'
assert np.array(uman).tolist()==[0,2,3,4,5,6,7,8]
assert only_dur==3.8

# psychometric, % corrects, regressions
if usub==subj_exp:
    pref = ''
elif usub==subj_ctl:
    pref = 'ctl_'

t9 = _ta[(_ta.level==9) & (_ta.outcome<2) & (_ta.subj.isin(usub))]
t9means = pd.DataFrame(columns=['manip', 'mean','n','subj'])
reqstrs = {(2,3,4):'_reqbil', (5,6,7,8):'_reqsub'}
for man in [0,(0,(2,3,4)),(0,(5,6,7,8)),2,3,4,5,6,7,8]:

    if isinstance(man, tuple):
        man,req = man
        reqstr = reqstrs[req]
    else:
        req = None
        reqstr = ''

    if req is None:
        mtr = mtrials.copy()
    else:
        req = np.array(req)
        has_manip = mtrials.groupby('session').apply(lambda x: np.any(x.manipulation.isin(req[req!=0])))
        manip_seshs = has_manip.index[has_manip.values]
        mtr = mtrials[mtrials.session.isin(manip_seshs)].copy()

    ti = mtr[mtr.manipulation==man].sort_values('uid')
    t9i = t9[t9.manipulation==man].copy()
    densityi = density[density.uid.isin(ti.uid.values)].copy().sort_values('uid')
    phases_ = phases[phases.session.isin(ti.session.unique())]
    phasesi = phases_[phases_.uid.isin(ti.uid.values)].copy().sort_values('uid')

    # psy
    ps = psy(ti, **psy_kw)
    with h5py.File(dest_path) as h:
        df_to_h5(h, data=ps, grp=f'{pref}psy_manip{man}{reqstr}')

    # fracs by subj
    fracs = pd.DataFrame(columns=['subj','frac','n'])
    for sub in sorted(ti.subj.unique()) + [None]:
        if sub is None:
            tis = ti.copy()
            sub = -1
        else:
            tis = ti[ti.subj==sub]
        fracs = fracs.append(dict(subj=sub, frac=tis.outcome.mean(), n=len(tis)), ignore_index=True)
    with h5py.File(dest_path) as h:
        df_to_h5(h, data=fracs, grp=f'{pref}fracs_manip{man}{reqstr}')

    if usub==subj_ctl:
        continue

    # regressions
    regr = regress(densityi, **reg_kw)
    with h5py.File(dest_path) as h:
        df_to_h5(h, data=regr, grp=f'regr_manip{man}{reqstr}')
    # regressions L and R separate
    regr_rl = regress(densityi, r_and_l=True, **reg_kw)
    if SAVE:
        with h5py.File(dest_path) as h:
            df_to_h5(h, data=regr_rl, grp=f'regrRL_manip{manstr}{reqstr}')

    # stats (regression 95% or 99% ci for significance)
    rkw = reg_kw.copy()
    rkw.update(error='99ci')
    regr = regress(densityi, **rkw)
    w = regr['weight'].values
    e = regr['yerr'].values
    print(f"Man{man}, req{req} {rkw['error']}:")
    for wi,ei in zip(w,e):
        print(f'\t{wi-ei:0.3f} – {wi+ei:0.3f}')
   
    # regr by sub 
    d0s = np.zeros([len(ti.subj.unique()), reg_kw['nbins']])
    for sidx,sub in enumerate(sorted(ti.subj.unique())):
        tis = ti[ti.subj==sub]
        dis = densityi[densityi.uid.isin(tis.uid.values)].copy().sort_values('uid')
        rx = regress(dis, **reg_kw)
        d0s[sidx,:] = rx.weight.values
    with h5py.File(dest_path) as h:
        if f'regr_subj_manip{man}{reqstr}' in h:
            del h[f'regr_subj_manip{man}{reqstr}']
        h.create_dataset(f'regr_subj_manip{man}{reqstr}', data=d0s)

    # ps9
    for sub in sorted(ti.subj.unique()):
        for side in [0,1]:
            tis = t9i[(t9i.subj==sub) & (t9i.side==side)]
            mean = tis.outcome.mean()
            n = len(tis)
            t9means = t9means.append(dict(mean=mean, n=n, manip=man, subj=sub, side=side), ignore_index=True)

    # latency
    resp_lat = response_latency(phasesi)
    with h5py.File(dest_path) as h:
        np_to_h5(h, data=resp_lat, grp='latency', dsname=f'{pref}manip{man}{reqstr}')

if usub == subj_exp:
    with h5py.File(dest_path) as h:
        df_to_h5(h, data=t9means, grp='easy_means')

    # light-triggered regression
    ltr = light_triggered_regression
    toi = mtrials.copy()
    toi = toi[toi.manipulation.isin([0,5,6,7])]
    dur = 3.8
    bins_per_third = 1
    ltr_kw = dict(density=density, dur=dur, bins_per_third=bins_per_third, subtract_baseline=False)
    time,full,(smean,serr) = ltr(toi, include_shuf=True, **ltr_kw)

    # error bars by bootstrap
    boot_err = np.std([ltr(toi, bootstrap=True, **ltr_kw)[1] for i in range(50)], axis=0, ddof=1)
    rdata = np.array([full, smean, serr, boot_err]).T
    res = pd.DataFrame(rdata, columns=['weights','shuffle_mean','shuffle_err','err_bootstrap'], index=time)
    
    with h5py.File(dest_path) as h:
        df_to_h5(h, data=res, grp='light_triggered_regression')


# --- Part 3, DDM - takes a while!
assert np.array(uman).tolist()==[0,2,3,4,5,6,7,8]
assert only_dur==None

# send to files for julia package
for man in [0,2,3,4,5,6,7,8,(2,3,4)]:
    print(man)
    if isinstance(man, (int,float)):
        man = (man,)
    tr_i = mtrials[mtrials.manipulation.isin(man)]
    output = deverett_to_brody_format(tr_i, _tt)
    output['perturbation_type'] = tr_i.manipulation.values
    mans = [str(i) for i in man]
    savemat(f"/Users/ben/Desktop/trials_man{''.join(mans)}.mat", output)

