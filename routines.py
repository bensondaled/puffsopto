import numpy as np
import pandas as pd
import os, warnings, json
from scipy.optimize import curve_fit
from statsmodels.stats.proportion import proportion_confint
from statsmodels.discrete.discrete_model import Logit
import matplotlib.pyplot as pl
import statsmodels.api as sm
from matplotlib.transforms import blended_transform_factory as blend

class HiddenValue(float):
    """A wrapper for python floats used to blind the experimenter from subject IDs
    """
    def __repr__(self):
        return 'No peeking!'
    def __str__(self):
        return 'No peeking!'

class HiddenList(list):
    """A wrapper for python lists used to blind the experimenter from subject IDs
    """
    def __init__(self, *args):
        super().__init__(*args)

        assert all([isinstance(i, (float,int)) for i in self])
        for i in range(len(self)):
            self[i] = HiddenValue(self[i])
    def __repr__(self):
        return 'No peeking!'
    def __str__(self):
        return 'No peeking!'

def load_subject_key():
    subj_key = pd.read_excel('subjects.xlsx').dropna(how='all')
    pos = (subj_key.ChR2.astype(int) & subj_key.Cre.astype(int)).values.astype(bool)
    neg = ~pos
    subj_experimental = subj_key['mouse #'][pos].values.tolist()
    subj_ctrl = subj_key['mouse #'][neg].values.tolist()
    return HiddenList(subj_experimental), HiddenList(subj_ctrl)

def load_data(cohort, levs=[6,7], root='/Users/ben/data/puffs', at_least_subj=73):
    """
    Loads behavioural data from `cohort`, restricting it to the levels in lev, outcome<2, and pre-running the regression caching.

    Parameters:
        cohort : cohort index or list thereof

    Returns:
        trials : filtered trials
        trials_all : all trials without any filtering
        phases : all phases
        density : stimulus timing in bins for regression
    """

    if not isinstance(cohort, (list,np.ndarray)):
        cohort = [cohort]
    
    fmt_str = '%Y%m%d%H%M%S'
    trials_all = []
    trials_timing = []
    phases = []
    for c in cohort:
        path = os.path.join(root, 'cohort_{}'.format(c), 'data.h5')
        assert os.path.exists(path), 'Path {} not found.'.format(path)
        print(path)

        with pd.HDFStore(path) as h:
            ti = h.trials
            pi = h.phases
            tti = h.trials_timing
        ti.loc[:,'cohort'] = c

        #print('Excluding img seshs.')
        for sesh in ti.session.unique():
            seshstr = pd.to_datetime(sesh).strftime(fmt_str)
            with pd.HDFStore(path) as h:
                params = json.loads(h['sessions/{}/params'.format(seshstr)][0])
                is_imging = params['imaging']
            if is_imging:
                ti = ti[~(ti.session==sesh)]

        trials_all.append(ti)
        trials_timing.append(tti)
        phases.append(pi)

    trials_all = pd.concat(trials_all)
    trials_timing = pd.concat(trials_timing)
    phases = pd.concat(phases)
    
    if at_least_subj is not None:
        trials_all = trials_all[trials_all.subj>=at_least_subj]

    ## Restrictions and labels

    t = trials_all.copy()
    t = add_trial_details(t)

    t = t[t.subj!=0]
    t = t[t.outcome<2]
    t = t[t.level.isin(levs)]

    # exclude bad trials
    t = t[t.n>0]
    t = t[(t.nL==t.nL_intended) & (t.nR==t.nR_intended)]

    density = cache_trial_timing_bins(t, trials_timing)
    density = add_trial_details(density, is_trials=False)

    # add durations,choices to trials_timing for trials in t (not all trials)
    fields = ['dur','choice']
    idict = dict(zip(t.uid.values, t[fields].values))
    vals = np.array([idict.get(uid,[np.nan]*len(fields)) for uid in trials_timing.uid.values])
    for fi,f in enumerate(fields):
        trials_timing.loc[:,f] = vals[:,fi]
    
    phases = phases[phases.session.isin(t.session.unique())]
    phases = add_uid(phases)

    return t,trials_all,phases,density,trials_timing

def add_uid(t):

    if 'uid' in t.columns:
        return t
    
    seshs = t.session.values
    if 'idx' in t.columns:
        trials = t.idx.values
    elif 'trial' in t.columns:
        trials = t.trial.values

    uids = []

    for idx,(sh,tr) in enumerate(zip(seshs,trials)):
        uid = '{}-{}'.format(sh,int(tr))
        uids.append(uid)

    t.loc[:,'uid'] = uids

    return t

def add_trial_details(t, is_trials=True):
    """For all rows of trials t, compute extra fields for convenience

    is_trials : indicates whether t is the true trials, or some other dataframe that only needs a subset of details
    """

    t.reset_index(inplace=True, drop=True)

    # choice
    if 'outcome' in t:
        t = add_trial_choice(t)

    # trialid
    t = add_uid(t)

    if is_trials:
        # difficulty
        t.loc[:,'rl'] = (t.nR-t.nL).values
        t.loc[:,'arl'] = np.abs(t.rl.values)
        t.loc[:,'n'] = (t.nR+t.nL-4).values

        # distractors
        t.loc[t.side==0,'distractors'] = t.nR[t.side==0].values-2
        t.loc[t.side==1,'distractors'] = t.nL[t.side==1].values-2

        # next and last choice, side, outcome
        seshs = []
        for sesh in t.session.unique():
            idxer = t.index[t.session==sesh]
            for thing in ['choice','side','outcome','manipulation']:
                last_str = 'last_'+thing
                next_str = 'next_'+thing
                first_things = t.loc[idxer].iloc[:-1][thing]
                last_things = t.loc[idxer].iloc[1:][thing]
                t.loc[idxer,last_str] = np.append(np.nan, first_things).astype(float)
                t.loc[idxer,next_str] = np.append(last_things, np.nan).astype(float)

                for i in range(2,3):
                    last_n_str = last_str+'_{}'.format(i)
                    next_n_str = next_str+'_{}'.format(i)
                    if len(t.loc[idxer])<=i:
                        t.loc[idxer,last_n_str] = np.nan
                        t.loc[idxer,next_n_str] = np.nan
                    else:
                        first_things = t.loc[idxer].iloc[:-i][thing]
                        last_things = t.loc[idxer].iloc[i:][thing]
                        t.loc[idxer,last_n_str] = np.append([np.nan]*i, first_things).astype(float)
                        t.loc[idxer,next_n_str] = np.append(last_things, [np.nan]*i).astype(float)
            seshs.append(t.loc[idxer])

        t = pd.concat(seshs)
    
    return t

def confidence(p,n):
    """Compute 95% confidence intervals

    Parameters
    ----------
    p : array-like
        probabilities (fractions) successes
    n : array-like
        trial counts
    """

    k = np.asarray(p*n) # number of successes
    n = np.asarray(n)
    
    # confidence interval bounds
    lo,hi = proportion_confint(k, n, method='jeffrey', alpha=0.05)

    # error values (difference from measured value)
    e0,e1 = p-lo,hi-p
    return np.array([e0,e1])[:,None]

def agent(trials, density, rule):
    """
    Given trials and density, construct an agent that makes a choice based on some rule

    Importantly, only the intersection of trials existing in `density` and `trials` is used.

    Also importantly, fills ambiguous trials with guesses, as an agent would.

    Rules:
        binX : use density bin, idx X
        or any column already existing in density (ex. first_stim)

    Returns the choice vector
    """

    density = density.copy()
    density.loc[:,'_idx'] = np.arange(len(density))

    include = trials.uid.isin(density.uid.values)
    exclude = trials[~include]
    trials = trials[include]

    # verify that the excluded trials are only those with a different duration
    assert np.all(trials.dur.values == trials.dur.values[0])
    if len(exclude) > 0:
        assert np.all(exclude.dur.values == exclude.dur.values[0])
        assert np.all(exclude.dur.values[0] != trials.dur.values[0])

    if rule.startswith('bin'):
        binidx = int(rule[3:])
        dcols = sorted([c for c in density.columns if isinstance(c,float)])
        coi = dcols[binidx]
        vals = density[coi].values
        ch = (vals>0).astype(float)
        ch[vals==0] = np.random.choice([0,1], size=np.sum(vals==0))
    elif rule in density.columns:
        ch = density[rule].values
    elif rule.startswith('damp'):
        binidx = rule[4:]
        binidx = binidx.split(',')
        binidx = [int(i) for i in binidx]
        dcols = sorted([c for c in density.columns if isinstance(c,float)])
        coi = [dcols[bi] for bi in binidx]
        dens = density.copy()
        dens[coi] = 0#dens[coi]/3
        dens_biased = dens[dcols].sum(axis=1)
        pr = sigmoid(dens_biased, .74, .14, .35, .17)
        ch = np.array([np.random.choice([0,1],p=[1-p,p]) for p in pr])
    
    dgb = density.groupby('uid')
    ch = trials.apply(lambda row: ch[int(dgb.get_group(row.uid)._idx)], axis=1)

    return ch

def add_trial_choice(trials):
    # if not present already, define choice based on outcome and side
    if 'choice' not in trials.columns:
        trials.loc[:,'choice'] = (trials.outcome == trials.side).astype(int)
    return trials

def psy(trials, bins=None, yvar='choice'):
    """Compute the psychometric curve for behavior in trials

    Parameters
    ----------
    trials : pd.DataFrame
        trials from data file
    bins : int
        number of bins for the nR-nL variable
    yvar : str
        the y variable to be computed as a function of RL

    Returns
    -------
    p : pd.Series
        index is nR-nL, and values are fraction of trials with R decision
    e : array
        associated error values for each fraction
    n : array
        associated trial counts for each fraction
    """

    # if not present already, define choice based on outcome and side
    trials = add_trial_choice(trials)

    # restrict trials to those with a correct/error outcome
    if 'outcome' in trials.columns:
        trials = trials[trials.outcome<2]
    
    # prepare bins, if applicable
    rl = trials.nR - trials.nL
    if bins is not None:
        maxx = np.max(np.abs(rl))
        bins = np.arange(-maxx,maxx+1,maxx/bins)
        grvar,binvals = pd.cut(rl, bins, retbins=True)
        bin_centers = (binvals[:-1]+binvals[1:])/2
    else:
        grvar = rl # grouping variable

    # compute psychometric
    grouping = trials.groupby(grvar)
    perf = grouping[yvar].agg([np.mean, len])
    if bins is not None:
        perf.index = bin_centers
    perf = perf.dropna()

    # add error measures
    yerr0,yerr1 = confidence(*perf.T.values).squeeze()
    perf.loc[:,'yerr0'] = yerr0
    perf.loc[:,'yerr1'] = yerr1

    return Psychometric(perf)

class Psychometric(pd.DataFrame):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def plot(self, **kwargs):

        ax = kwargs.pop('ax', pl.gca())
        
        kwargs['color'] = kwargs.get('color', 'k')
        kwargs['ecolor'] = kwargs.get('ecolor', kwargs['color'])

        ax.errorbar(    self.index, 
                        self['mean'].values, 
                        yerr=[self['yerr0'].values, self['yerr1'].values],
                        **kwargs)
        
        ax.set_ylim([-.02,1.02])

        # center guides
        #ax.plot([.5,.5], [0,1], lw=.5, color='dimgrey', linestyle=':', alpha=.5)
        #ax.plot(ax.get_xlim(), [.5, .5], lw=.5, color='dimgrey', linestyle=':', alpha=.5)

        ax.set_xlabel('#R-#L')
        ax.set_ylabel('Fraction rightward decisions')

        return ax

def sigmoid(x, A, x0, k, y0):
    return A / (1 + np.exp(-k*(x-x0))) + y0
    # slope is kA/4
def psy_fit(x, y=None, ax=None, lines=False, text=-1, **kwargs):
    """Fit sigmoid to a psychometric curve
    """
    if isinstance(x, pd.Series):
        y = x.values
        x = x.index
    try:
        popt,pcov = curve_fit(sigmoid, x, y, maxfev=10000)
    except:
        return [-1,-1,-1,-1],None
    xfit = np.linspace(x.min(), x.max(), 200)
    yfit = sigmoid(xfit, *popt)
    fit = pd.Series(yfit, index=xfit)

    return popt, fit

def downsample_bins(d, n, method=np.sum):
    # downsample density to n bins
    return np.array([method(i,axis=-1) for i in np.array_split(d, n, axis=-1)]).T

def regress(density, nbins=None, only_dur=None, error='95ci', fit_intercept=False, r_and_l=False):

    if only_dur is not None:
        density = density[density.dur==only_dur]
    else:
        only_dur = density.dur.values[0]

    assert np.all(density.dur.values == density.dur.values[0]), 'Cannot regress on different trial lengths at once.'

    dcols_L = np.array(sorted([c for c in density.columns if c.startswith('L_')]))
    dcols_R = np.array(sorted([c for c in density.columns if c.startswith('R_')]))
    dcols_L_num = np.array([float(i[2:]) for i in dcols_L])
    dcols_R_num = np.array([float(i[2:]) for i in dcols_L])
    
    # restrict to bins that exist for duration of interest
    is_indur_L = dcols_L_num <= only_dur
    is_indur_R = dcols_R_num <= only_dur
    dcols_L = dcols_L[is_indur_L]
    dcols_R = dcols_R[is_indur_R]
    dcols_L_num = dcols_L_num[is_indur_L]
    dcols_R_num = dcols_R_num[is_indur_R]

    assert np.all(dcols_L_num == dcols_R_num)
    dcols_num = dcols_L_num

    uchoice = density.choice.values
    udens_L = density[dcols_L].values
    udens_R = density[dcols_R].values
    udens = udens_R-udens_L

    if nbins is not None:
        udens = downsample_bins(udens, nbins)
        udens_L = downsample_bins(udens_L, nbins)
        udens_R = downsample_bins(udens_R, nbins)
        dcols_num = downsample_bins(dcols_num, nbins, method=np.mean)

    isnan = np.isnan(uchoice)
    if np.sum(isnan)>0:
        print('Excluding {:0.0f}% nans.'.format(np.mean(isnan)*100))
    uchoice = uchoice[~isnan]
    udens = udens[~isnan]
    udens_L = udens_L[~isnan]
    udens_R = udens_R[~isnan]

    if r_and_l: # use r and l as separate regressors
        udens = np.concatenate([udens_L, udens_R], axis=1)
    
    if fit_intercept:
        udens = sm.add_constant(udens)

    try:
        reg = Logit(uchoice, udens, missing='drop').fit(disp=False)
        reg_params = reg.params
        if error == '95ci':
            reg_err = np.abs(reg.conf_int(alpha=0.05).T - reg.params) # 95% CI
        elif error == '99ci':
            reg_err = np.abs(reg.conf_int(alpha=0.01).T - reg.params) # 99% CI
        elif error == 'se':
            reg_err = reg.bse # standard error
        elif error == 'bootstrap':
            boots = []
            for i in range(1000):
                samp = np.random.choice(np.arange(len(udens)), replace=True, size=len(udens))
                udb = udens[samp]
                uch = uchoice[samp]
                boots.append(Logit(uch, udb, missing='drop').fit(disp=False).params)
            reg_err = np.std(boots, axis=0) #/ np.sqrt(len(boots))
    except (np.linalg.LinAlgError, sm.tools.sm_exceptions.PerfectSeparationError):
        reg_params = np.nan * np.zeros(udens.shape[1])
        reg_err = np.nan * np.zeros([2, len(reg_params)])
    
    index = dcols_num
    if r_and_l:
        index = np.tile(index, 2)
    if fit_intercept:
        index = np.append(-1, index)

    res = pd.DataFrame(index=index)
    res.loc[:,'weight'] = reg_params

    if not np.any(np.isnan(reg_err)):
        if reg_err.ndim == 2:
            assert np.allclose(reg_err[0],reg_err[1]) # symmetrical errorbars
            res.loc[:,'yerr'] = reg_err[0] # half of confidence interval
        elif reg_err.ndim == 1:
            res.loc[:,'yerr'] = reg_err
    else:
        res.loc[:,'yerr'] = np.nan
    
    if r_and_l:
        resl = res.iloc[:len(res)//2]
        resl.columns = [c+'_L' for c in resl.columns]
        resr = res.iloc[len(res)//2:]
        resr.columns = [c+'_R' for c in resr.columns]
        res = resl.join(resr)

    return res

def OLD_regress(density):

    dcols = sorted([c for c in density.columns if isinstance(c,float)])

    uchoice = density.choice.values
    uvals = density[dcols].values

    isnan = np.isnan(uchoice)
    if np.sum(isnan)>0:
        print('Excluding {:0.0f}% nans.'.format(np.mean(isnan)*100))
    uchoice = uchoice[~isnan]
    uvals = uvals[~isnan]

    try:
        reg = Logit(uchoice,uvals).fit(disp=False)
        reg_params = reg.params
        reg_err = np.abs(reg.conf_int(alpha=0.05).T - reg.params)
    except (np.linalg.LinAlgError, sm.tools.sm_exceptions.PerfectSeparationError):
        reg_params = np.nan * np.zeros(uvals.shape[1])
        reg_err = np.nan * np.zeros([2, len(reg_params)])
        

    res = pd.DataFrame(index=dcols)
    res.loc[:,'weight'] = reg_params
    if not np.any(np.isnan(reg_err)):
        assert np.allclose(reg_err[0],reg_err[1]) # symmetrical errorbars
        res.loc[:,'yerr'] = reg_err[0] # half of confidence interval
    else:
        res.loc[:,'yerr'] = np.nan

    return res

def OLD_cache_trial_timing_bins(trials, trials_timing, nbins=3, dur=3.8, data_path='.'):
    """Ready for deletion
    """

    cpath = os.path.join(data_path, 'caches', 'density_{}_{}'.format(dur,nbins))
    tt = trials_timing
    ttgb = tt.groupby(['session','trial'])
    
    # load already cached trials and exclude from the current caching process all sessions that have been cached
    if os.path.exists(cpath):
        ctrials = pd.read_pickle(cpath)
        trials = trials[~trials.session.isin(ctrials.session.unique())]
    else:
        ctrials = None

    def make_bins(dur):
        bin_width = dur / nbins
        bins = np.arange(0, dur+1e-10, bin_width)
        bin_labs = (bins[1:]+bins[:-1])/2
        return bins,bin_labs

    def extract_trialtimings(t):
        sesh,tidx,dur = t.session,t.idx,t.dur
        tim = ttgb.get_group((sesh,tidx))

        # For cases where stereo first and last are present
        assert tim.iloc[0].time==tim.iloc[1].time and tim.iloc[-1].time==tim.iloc[-2].time
        tim = tim.iloc[2:-2]
        assert all(tim.time > 0)
        assert all(tim.time < dur)

        bins,bin_labs = make_bins(dur)
        binl,_ = np.histogram(tim[tim.side==0].time, bins=bins)
        binr,_ = np.histogram(tim[tim.side==1].time, bins=bins)
        assert len(tim) == np.sum(binl) + np.sum(binr)
        return binr-binl, tim.iloc[0].side, tim.iloc[-1].side

    # limit to the desired trial duration
    tud = trials[trials.dur==dur]
    
    # compute binned stim counts
    bins,bin_labs = make_bins(dur)
    density = np.empty([len(tud), len(bins)-1+2])
    for ti,(tid,t) in enumerate(tud.iterrows()):
        density[ti,:-2],density[ti,-2],density[ti,-1] = extract_trialtimings(t)
    density_df = pd.DataFrame(density, 
            columns=list(bin_labs)+['first_stim','last_stim'])

    # add trial info onto result
    df = tud[['session','idx','dur','side','outcome']].reset_index(drop=True)
    result = df.join(density_df)

    # combine cached and new results
    result = pd.concat([ctrials, result])
    result = result.reset_index(drop=True)
    pd.to_pickle(result, cpath)

    return result

def cache_trial_timing_bins(trials, trials_timing, bin_dur=.010, data_path='.'):
    """

    todo: reinclude a check for bilaterals?
    """

    cpath = os.path.join(data_path, 'caches', 'density')
    tt = trials_timing
    tt = add_uid(tt)
    ttgb = tt.groupby('uid')
    
    # load already cached trials and exclude from the current caching process all 
    # trials that have been cached already
    if os.path.exists(cpath):
        ctrials = pd.read_pickle(cpath)
        trials = trials[~trials.uid.isin(ctrials.uid.unique())]
    else:
        ctrials = None

    if len(trials) > 0:
        
        maxdur = trials.dur.max()
        bins = np.arange(0, maxdur+1e-10, bin_dur)
        bin_labs = (bins[1:]+bins[:-1])/2
        bin_labs_l = ['L_{:0.3f}'.format(i) for i in bin_labs]
        bin_labs_r = ['R_{:0.3f}'.format(i) for i in bin_labs]
        bin_labs = bin_labs_l + bin_labs_r

        density = np.empty([len(trials), 2, len(bins)-1]) # trials x LR x bins

        print('Caching new trials...')

        for ti,tuid in enumerate(trials.uid.values):
            if ti%250==0:   print(ti,'/',len(trials))
            tim = ttgb.get_group(tuid)

            assert all(tim.time >= 0)
            assert all(tim.time <= maxdur)

            binl,_ = np.histogram(tim[tim.side==0].time, bins=bins)
            binr,_ = np.histogram(tim[tim.side==1].time, bins=bins)
            assert len(tim) == np.sum(binl) + np.sum(binr)

            density[ti, 0, :] = binl
            density[ti, 1, :] = binr

        density = np.concatenate(density.transpose([1,0,2]), axis=1)
        
        density_df = pd.DataFrame(density, columns=list(bin_labs))

        # add trial info onto result
        df = trials[['session','idx','uid','dur','side','outcome']].reset_index(drop=True)
        result = df.join(density_df)

        # combine cached and new results
        result = pd.concat([ctrials, result])
        result = result.reset_index(drop=True)
        pd.to_pickle(result, cpath)

    else:
        result = ctrials

    return result

def licata_fit(t, max_rl=None):
    """Using symbols from Licata 2017
    Model from Busse 2011 JNeuro

    convention: -1 means left, 1 means right
    """
    if max_rl is None:
        max_rl = np.max(np.abs((t.nR-t.nL).values))

    r = np.abs(t.nR-t.nL).values / max_rl
    r[t.side==0] = -r[t.side==0]

    ch = t.choice.values.copy()
    #ch[ch==0] = -1

    h_success = t.last_outcome.values.astype(int)
    h_success[t.last_choice==0] *= -1
    # interpretation of this: 1 if previous trial was R-choice&correct, -1 if L-choice&correct
    # b/c R choices are coded as 1, fit weights on this regressor are interpreted as: higher positive weight means correct-and-stay, more negative weight means correct-and-switch

    h_fail = (t.last_outcome==0).astype(int).values
    h_fail[t.last_choice==0] *= -1
    # interpretation of this: 1 if previous trial was R-choice&error, -1 if L-choice&error
    # b/c R choices are coded as 1, fit weights on this regressor are interpreted as: higher positive weight means error-and-stay, more negative weight means error-and-switch

    b0 = np.ones_like(r)

    y = ch
    x = np.array([b0, r, h_success, h_fail]).T

    #print(x.min(axis=0), x.max(axis=0))

    # run GLM
    
    # version 1:
    """
    logit_link = sm.genmod.families.links.logit
    glm_binom = sm.GLM(
            y,
            x,
            family=sm.families.Binomial(link=logit_link))
    glm_result = glm_binom.fit(maxiter=1000, method='bfgs')
    """

    # version 2:
    glm_result = Logit(y,x).fit(maxiter=1000, method='powell', disp=False)

    params = glm_result.params  
    err = glm_result.bse

    return params,err

def load_session(sesh, data_path):
    sesh = os.path.split(sesh)[-1]
    sesh = os.path.splitext(sesh)[0]
    sesh = sesh[:sesh.index('_') if '_' in sesh else None]

    droot = os.path.split(data_path)[0]
    fmt_str = '%Y%m%d%H%M%S'

    # load in the desired fields
    with pd.HDFStore(data_path) as d:
        trials = d.trials[d.trials.session.dt.strftime(fmt_str)==sesh]
        dt_sesh = trials.iloc[0].session # session in datetime format

        sync = d['sessions/{}/sync'.format(sesh)]
        ar = d.select('analogreader', where='session={}'.format(repr(dt_sesh)))
        phases = d.select('phases', where='session={}'.format(repr(dt_sesh)))

    phases.ix[:,'onset'] = 0.0 #default for alignment to start of phase


    syncval = sync.session-sync.ar
    syncbase = 'index'
    ar = correct_ar_timestamps(ar, sync=syncval, syncbase=syncbase)
    ardiff = np.diff(ar.index)
    if not np.std(ardiff)<0.001:
        warnings.warn('check AR timestamps, consistency is suspect')
    ar.Ts = np.mean(ardiff)

    session_coords_time_clock_difference = trials.index[0] - float(trials.iloc[0].ts_global)
    
    return ar,trials,[sync,session_coords_time_clock_difference]
    
def correct_ar_timestamps(ar, fs=500, sync=None, syncbase=None):
    Ts = 1./fs
    assert len(ar)%10 == 0
    sub = np.arange(10,0,-1)*Ts
    sub = np.tile(sub, len(ar)//10)
    if syncbase == 'ts_global':
        base = ar.ts_global.values
    elif syncbase == 'index':
        base = np.asarray(ar.index)
    ar.set_index(base - sub, inplace=True, drop=True)
    if sync is not None:
        ar.index += sync
    return ar

def load_single_session_ar(sesh, data_path):
    """For a single session `sesh`, load from `data_path` the AR field
    """
    if not isinstance(data_path, (list,np.ndarray)):
        data_path = [data_path]
    fmt_str = '%Y%m%d%H%M%S'

    for dp in data_path:
        with pd.HDFStore(dp) as d:
            trials = d.trials[d.trials.session==sesh]
            if len(trials) > 0:
                break

    with pd.HDFStore(dp) as d:
        trials = d.trials[d.trials.session==sesh]
        dt_sesh = trials.iloc[0].session # session in datetime format
        ar = d.select('analogreader', where='session={}'.format(repr(dt_sesh)))
        sesh_str = trials.session.dt.strftime(fmt_str).values[0]
        sync = d['sessions/{}/sync'.format(sesh_str)]
    
    syncval = sync.session-sync.ar
    syncbase = 'index'
    ar = correct_ar_timestamps(ar, sync=syncval, syncbase=syncbase)
    ardiff = np.diff(ar.index)
    if not np.std(ardiff)<0.001:
        warnings.warn('check AR timestamps, consistency is suspect')
    ar.Ts = np.mean(ardiff)
    return ar

def compute_latency(p, phases=[2,4]):
    """
    p : rows from data.phases for a single trial
    returns latency from onset of phases[0] to onset of phases[1]
    """
    p = p[p.phase.isin(phases)]
    p0 = p[p.phase==phases[0]].start_time.values
    p1 = p[p.phase==phases[1]].start_time.values
    assert len(p0) == len(p1)
    res = p1-p0
    return res

def frac_correct(t):
    """Compute fraction correct and confidence interval of trials t
    """
    assert np.all(t.outcome.values<2)
    frac = t.outcome.mean()
    conf = confidence(frac, len(t))
    return frac, conf

def heatmap(trials, at_least_ntrials=1):
    trials = trials[trials.outcome<2]

    trials = trials.copy()
    trials.nR -= 2
    trials.nL -= 2
    n_grps = trials.groupby(['nL','nR'])
    keys = np.asarray(list(n_grps.groups))

    maxs,mins = keys.max(axis=0).astype(int),keys.min(axis=0).astype(int)
    if not np.all(mins==0):
        warnings.warn('Heat map ticks are shifted, b/c heatmap does not start at 0.')
    mat = np.zeros([maxs[0]-mins[0]+1, maxs[1]-mins[1]+1], dtype=float)
    mat[:] = np.nan
    for gr_id,gr in n_grps:
        if len(gr) < at_least_ntrials:
            continue
        gr_id = (np.asarray(gr_id) - mins).astype(int)
        mat[gr_id[0],gr_id[1]] = gr.choice.mean()
    return mat

def reverse_correlation(density, only_dur=3.8):
    tt = density[density.dur==only_dur]
    ch = tt.choice.values

    dcols_L = np.array(sorted([c for c in tt.columns if c.startswith('L_')]))
    dcols_R = np.array(sorted([c for c in tt.columns if c.startswith('R_')]))
    dcols_L_num = np.array([float(i[2:]) for i in dcols_L])
    dcols_R_num = np.array([float(i[2:]) for i in dcols_L])
    Ts = np.mean(np.diff(dcols_L_num))
    tt_L = tt[dcols_L].values
    tt_R = tt[dcols_R].values
    rl_byrow = (tt_R-tt_L).sum(axis=1)
    url = np.unique(rl_byrow)
    
    # ctrl agents
    #ch = ((tt_R-tt_L)[:,:100].sum(axis=1) > 0).astype(int)
    
    kern_dur = .700 # sec
    sigma = .3 # sec
    kernel = np.arange(0, kern_dur+Ts, Ts)
    kernel = np.sqrt(1/(2*np.pi*sigma)) * np.exp(-kernel**2/(2*sigma**2))

    ttl_filt = np.array([np.convolve(row, kernel, mode='valid') for row in tt_L])
    ttr_filt = np.array([np.convolve(row, kernel, mode='valid') for row in tt_R])
    rl = ttr_filt-ttl_filt
    
    # subtract medians:
    '''
    excess = np.zeros_like(rl)
    for rli in url:
        select = rl_byrow==rli
        rl_ = rl[select]
        median = np.median(rl_, axis=0)
        excess[select] = rl_-median
    '''
    # or dont:
    excess = rl.copy()

    rev_l = excess[ch==0].mean(axis=0)
    rev_r = excess[ch==1].mean(axis=0)

    rev = rev_r-rev_l

    #downsamp = 4
    #rev = rev.reshape([-1,downsamp]).mean(axis=1)
    return rev_l,rev_r

def confirm_cueperiod_predictability(trials, density):
    # sanity chk: confirm that cue period uniformly predicts correct side
    toi = trials[(trials.dur==3.8) & (trials.level.isin([6,7]))]
    tt = density[density.session.isin(toi.session.unique())]
    tt = tt[tt.uid.isin(toi.uid.unique())]
    dcols_L = np.array(sorted([c for c in tt.columns if c.startswith('L_')]))
    dcols_R = np.array(sorted([c for c in tt.columns if c.startswith('R_')]))
    tt_L = tt[dcols_L].values
    tt_R = tt[dcols_R].values
    rl = tt_R-tt_L
    side = (rl.sum(axis=1)>0).astype(int)
    ch = toi.choice.values
    N = 10
    Ts = 3.8/N
    idxs = np.array_split(np.arange(len(dcols_L)), N)
    inp = np.array([rl[:,i].sum(axis=1) for i in idxs])
    #inp = np.array([np.random.choice(i,replace=False,size=i.size) for i in inp])# ctrl
    logs = [Logit(ch, i).fit(disp=False) for i in inp]
    weights = np.array([l.params[0] for l in logs])
    werrs = np.array([l.bse[0] for l in logs])
    pl.fill_between(np.arange(len(weights))*Ts + Ts/2, weights+werrs, weights-werrs)

def remove_ending(t, cut=50):
    oc = t.outcome.values
    i_last = len(oc) - cut
    return t.iloc[:i_last]

def df_to_h5(handle, data, grp):
    grp = handle.require_group(grp)

    assert isinstance(data, pd.DataFrame)
    
    data = data.reset_index(drop=False)
    names = data.columns
    dats = data.values.T
    for dsn,dsdat in zip(names, dats):
        if dsn in grp:
            del grp[dsn]
        ds = grp.create_dataset(dsn, data=dsdat, compression='lzf')

def np_to_h5(handle, data, grp, dsname, attrs={}):
    grp = handle.require_group(grp)

    assert isinstance(data, np.ndarray)
    
    if dsname in grp:
        del grp[dsname]
    ds = grp.create_dataset(dsname, data=data, compression='lzf')

    for k,v in attrs.items():
        ds.attrs[k] = v

def light_triggered_regression(toi, density, dur=3.8, bins_per_third=2, include_shuf=False, subtract_baseline=True, bootstrap=False):
    """
    Given the trial-timing density matrix (one trial per row, one column per time bin)
    and the trials of interest toi,
    compute the regression analysis for manips 5,6,7: corresponding to light delivered in the first, second, and third thirds of the cue period respectively,
    then combine them such that you get a light-triggered regression

    bootstrap: instead of using trials, use a random sample with replacement of the same size from the trials
    include_shuf: in addition to computing the result, compute a bunch (200) versions of the result when manipulation labels are shuffled across trials
    """

    # restrict duration of trials to single duration
    toi = toi[toi.dur==dur].sort_values('uid')

    # extract the R-L values for all relevant trials
    tt = density.copy()
    tt = tt[tt.uid.isin(toi.uid.values)].sort_values('uid')
    dcols_L = np.array(sorted([c for c in tt.columns if c.startswith('L_')]))
    dcols_R = np.array(sorted([c for c in tt.columns if c.startswith('R_')]))
    dcols_L_num = np.array([float(i[2:]) for i in dcols_L])
    dcols_R_num = np.array([float(i[2:]) for i in dcols_L])
    tt_L = tt[dcols_L].values
    tt_R = tt[dcols_R].values
    last = np.argmin(np.abs(dcols_L_num-dur))
    rl = tt_R-tt_L
    rl = rl[:,:last]
    #rl = rl[:,1:last-1] # or this to try excluding bilaterals
    
    # prepare arrays of interest for regression
    mans = toi.manipulation.values
    mans_shuf = mans.copy()
    ch = toi.choice.values
    div = bins_per_third
    rl = downsample_bins(rl, div*3)
    Ts = dur/rl.shape[1]

    # regress
    def process_regs(regs):
        # given the regression results for manips 5,6,7, combine them using the knowledge that nothing precedes the first third (manip5), nothing follows the third third (manip7), etc.
        if subtract_baseline:
            ps = {man:val.params-regs[0].params for man,val in regs.items()}
        else:
            ps = {man:val.params for man,val in regs.items()} 
        pre = [np.append([np.nan]*(div), ps[6][:div]), ps[7][:2*div]]
        pre = np.nanmean(pre, axis=0)
        post = [ps[5][:], np.append(ps[6][div:],div*[np.nan]), np.append(ps[7][2*div:],2*div*[np.nan])]
        post = np.nanmean(post, axis=0)
        full = np.append(pre,post)
        time = (np.arange(len(full))-len(pre))*Ts
        return time,full

    regs = {}
    for man in [0,5,6,7]:
        ch_ = ch[mans==man]
        rl_ = rl[mans==man]
        if bootstrap:
            idx = np.random.choice(np.arange(len(ch_)), replace=True, size=len(ch_))
            ch_ = ch_[idx]
            rl_ = rl_[idx,:]
        regs[man] = Logit(ch_, rl_, missing='drop').fit(disp=False)

    time,full = process_regs(regs)

    if not include_shuf:
        return time,full

    # shuffle control
    shufs = []
    for i in range(200):
        reg_shufs = {}
        print(f'{i+1}/200')
        np.random.shuffle(mans_shuf)
        for man in [0,5,6,7]:
            reg_shufs[man] = Logit(ch[mans_shuf==man], rl[mans_shuf==man], missing='drop').fit(disp=False)
        shufs.append(process_regs(reg_shufs)[1])

    smean = np.mean(shufs, axis=0)
    serr = np.std(shufs, axis=0)

    return time,full,(smean,serr)

def match_toi_tt(toi, tt):
    toi = toi.sort_values('uid')
    tt = tt[tt.session.isin(toi.session.unique())]
    all_tt = []
    for sesh in toi.session.unique():
        tti = tt[tt.session==sesh]
        uuid = toi[toi.session==sesh].uid.unique()
        tti = tti[tti.uid.isin(uuid)].sort_values('uid')
        all_tt.append(tti)
    return toi,pd.concat(all_tt)

def response_latency(p):
    lats = []
    p = p[p.phase==3]
    lat = (p.end_time - p.start_time).values
    lat *= 1000 # to milliseconds
    return lat
