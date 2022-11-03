import multiprocessing
import math
import collections
from functools import partial
import random
import pickle
import copy
import itertools
from sys import stdout
from sys import exit
import os
import time # for sleep()
import datetime

import numpy as np

import scipy.stats as stats
import pandas as pd

nb_cores = multiprocessing.cpu_count()

def now(format_):
    """
        Example formats:
            '%d %b %Y %H:%M:%S'
            '%d/%m/%Y %H:%M:%S'
            '%Y%m%d'
    """
    timestamp = datetime.datetime.now()

    return timestamp.strftime(format_)

def create_dir(path):
    """
        create a directory if it does not already exist
    """
    if not os.path.exists(path):
        os.makedirs(path)

def find_nearest(array, value):
    """       
        Find nearest value in numpy array
        
        https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def load_data(fname):
    """
        load dataframe from pickle file
    """
    return pickle.load(open(fname, 'rb'))

def save_data(df, fname):
    """
        save dataframe to pickle file
    """
    pickle.dump(df, open(fname, 'wb'))

def is_iterable(x):
    from collections import Iterable
    return isinstance(x, Iterable)

def has_columns(df, cols):
    """
        returns a subset of the column names in `cols` which exist in `df`
    """
    cols_exist = [ col for col in cols if col in df.columns]
    return cols_exist

def get_unique(series):
    
    if series.empty:
        return np.nan
    
    # unique only works for immutable objects
    # https://stackoverflow.com/questions/50418645/unique-items-in-a-pandas-dataframe-with-a-list
    if isinstance(series.iloc[0], collections.Hashable):
        item = series.unique()
    else: # it might be a list
        item = series.transform(tuple).unique()
        
    assert(len(item) == 1)
        
    return item[0]

def query_all(df, query):
    """
        https://stackoverflow.com/questions/46822423/pandas-dataframe-query-expression-that-returns-all-rows-by-default
    """
    
    if query == 'all':
        return df
    
    # NOTE avoid "UndefinedVariableError: name 'x' is not defined" when using `query()`
    if len(df) == 0:
        return df

    return df.query(query)

def identity(x):
    """
        hack to get multiprocessing.Pool() to work. 
        see https://stackoverflow.com/questions/8804830/python-multiprocessing-picklingerror-cant-pickle-type-function
    """
    return x

def round_series(series):
    """
        round all floats to ints (user inputs in the app are restricted to ints)
    """
    
    return series.round(0).astype(int)

def to_z_score(conf_level):
    """
        https://stackoverflow.com/questions/20864847/probability-to-z-score-and-vice-versa
    """
    
    assert(conf_level < 1)
    assert(conf_level > 0)
    
    return stats.norm.ppf(1-(1-conf_level)/2)

def to_conf_level(z_score):
    """
        https://stackoverflow.com/questions/20864847/probability-to-z-score-and-vice-versa
    """
    
    return stats.norm.cdf(z_score)

def ecdf(series, threshold):
    """
        Query an empirical density function p(x <= threshold)
        
        https://stackoverflow.com/questions/36353997/empirical-distribution-function-in-numpy
    """
    series = series.sort_values()
    prob = np.searchsorted(series, threshold, side='right') / len(series)
    
    return prob

def conf_int(series, conf_lvl=0.95):
    """
        Confidence interval at `conf_lvl` confidence level
        
        https://kite.com/python/examples/702/scipy-compute-a-confidence-interval-from-a-dataset
        https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
    """
    
    count = len(series)
    mean = series.mean()
    std_err = stats.sem(series)
    h = std_err * stats.t.ppf((1 + conf_lvl) / 2, count - 1) # student’s t distribution
    
    return pd.Series(data={ 'mean':mean, 
                            'std_err':std_err, 
                            'error':h, 
                            'conf_lvl':conf_lvl, 
                            'count':count,
                            'lower_bound': mean-h, 
                            'upper_bound': mean+h })

def concat_data(data):
    """
        concatenate dictionaries of dataframes
    """
    
    for k in data.keys():
        if data[k] is not None and len(data[k]) > 0:
            data[k] = pd.concat(data[k], axis='index')
    
    return data

def concat_data_(data_list):
    """
        concatenate dictionaries of dataframes
    """
    
    data = { 'kept':None, 'dropped':None }
    for k in data.keys():
        u = [ d[k] for d in data_list if d[k] is not None and len(d[k]) > 0]
        if u is not None and len(u) > 0:
            data[k] = pd.concat(u, axis='index')
            
    return data

def set_df_params_ids(df_params):
    """
        set `'param_id'`
    """
    
    if 'param_id' in df_params.columns:
        return df_params
    
    df_params = df_params.reset_index(drop=True)
    df_params['id'] = range(len(df_params))
        
    return df_params

def unroll_df_params(df_params):
    """
        create a single row per iteration seed
        This makes it easier to parallelise runs
        duplicate each row with a different `iteration_id` and `rnd_seed`
    """
    
    if not isinstance(df_params, pd.DataFrame):
        df_params = params_to_df(copy.deepcopy(df_params))
    
    df = []
    for index, row in df_params.iterrows():
        rnd_seeds = row.get('rnd_seed')
        if is_iterable(rnd_seeds):
            nb_iterations = len(rnd_seeds)
            df_ = pd.concat([row.to_frame().T]*nb_iterations, axis='index')
            df_['rnd_seed'] = df_['rnd_seed'].iloc[0]
            df_['iteration_id'] = range(len(df_))
            df_['nb_iterations'] = nb_iterations
        else:
            df_ = row.to_frame().T
            df_['nb_iterations'] = 1

        df.append(df_)

    df = pd.concat(df, axis='index')
    df = df.reset_index(drop=True)
    
    assert(df['rnd_seed'].nunique() == len(df))
    
    return df

def process_params(params):
    """
        process parameters that can be either scalars or lists/arrays
    """
       
    params = copy.deepcopy(params)
    
    params['iSBP'] = np.atleast_1d(params['iSBP']) # ensures it is always a list
    assert((len(params['iSBP']) == 1) or (len(params['iSBP']) == params['nb_patients']))
    
    assert(params['controlled_threshold'] < params['treatment_threshold'])
    assert(params['treatment_threshold'] < params['inertia_threshold'])

    assert('id' in params), "params['id'] not set by default"
    assert('rnd_seed' in params), "params['rnd_seed'] not set by default"
    
    assert(params['start_inertia_at_cycle'] <= 1)
    
    params['is_processed'] = True
    
    return params

# mapping from parameter name to latex string
pname_to_latex = { 
    'meas_var': '\sigma_{{meas}}^2', 
    'meas_std': '\sigma_{{meas}}', 
    'drug_var': '\sigma_{{drug}}^2', 
    'drug_std': '\sigma_{{drug}}', 
    'drug_eff': '\mathrm{{drug}}_{{\mathrm{{eff}}}}', 
} 
pname_to_latex_df = pd.DataFrame(pname_to_latex, index=[0]).transpose().rename({0:'latex'}, axis='columns')

# template for default parameters (clinically relevant parameters)
# other than `'is_processed'`, all values are read-only
params_default = { 
    
    'is_processed':                False, 
    
    'iteration_id':                0,
    
    'nb_cycles':                   1,
    'max_nb_time_steps_per_cycle': None,
    
    # Patients
    'nb_patients':  1000, # population size (scalar)
    'iSBP':          150, # initial true systolic blood pressure (scalar of list)
    'age':           60, # scalar of list
    'is_male':       True, # scalar of list
    'ten_year_risk': 1, # scalar of list

    # Treatments
    'dose':      'FixedDose', 
    'drug_eff':  10, # only used when 'dose' is set to `FixedDose`
    'drug_std':  5, 
    'drug_dist': 'truncnorm', # drug distribution:
                              #     'norm', 'truncnorm', 'point_mass'
    
    # Measurements
    'measurement_strategy': 'single', # 'single': 
                                      # 'double_original': strategy as per Lorenzo's original paper
                                      # 'double_new':   new strategy for cycles
    'nb_initial_titration': 0, # 0: mono initiation strategy
                               # 1: dual initiation strategy
                               # 2: triple initiation strategy
    'meas_std':             10, # scalar of list
    'meas_dist':            'norm',       # measurement distribution:
                                          #     'norm', 'point_mass', 'skewnorm'
    
    # Inertia
    'inertia_fct': 'hard',
    'start_inertia_at_cycle': 0, 
    
    # White coat effect
    'meas_shift': 0, 
    'meas_skew': 0, 
    
    # Decisions
    'treatment_threshold':  140,
    'controlled_threshold': 120, 
    'inertia_threshold':    160
}

def params_to_df(params):
    """

    """
    
    # Numpy arrays are not allowed (use lists)
    if isinstance(params, pd.Series):
        values = params.values
    elif isinstance(params, dict):
        values = params.values()
    else:
        assert(False)
    for v in values:
        if is_iterable(v) and not isinstance(v, str):
            assert(isinstance(v, list))
    
    # wrap all lists into another list to keep pd.DataFrame() happy
    tmp = { k:([v] if isinstance(v, list) else v) for (k,v) in params.items() }
    return pd.DataFrame(tmp, index=[0])
    
params_default_df = params_to_df(params_default)

# NOTE do not use, as changing it would break everything
# default_index = ['iteration_id', 'cycle_step', 'time_step', 'patient_id']

def create_empty_treatment_step(params, patient_id=[]):
    """
        create the dataframe for the dataset 
    """

    columns = ['iteration_id',
               'cycle_step', 
               'time_step',
               'patient_id', 
               'iSBP', 
               
               'treatment_step', 
               'tSBP', 
               'drug_eff', 
               'drug_std', 
               
               'meas_step',
               'mSBP', 
               'meas_std',
               'param_id']
    
    # create the dataframe with the correct number of rows (filled with NaN)
    df = pd.DataFrame(columns=columns)
    
    df['patient_id'] = patient_id
    
    # NaN for integer columns
    df['treatment_step'] = df['treatment_step'].astype(pd.Int64Dtype())
    df['meas_step']      = df['meas_step'].astype(pd.Int64Dtype())
    
    return df

def generate_seed_treatment_step(params, 
                                 iteration_id=0, 
                                 cycle_step=0):
    """
        1. Creates an empty time step with `params['nb_patients']` number of patients
        
        2. Set:
            - the iteration_id from argument
            - the cycle step from argument
            - the time step to -1
            - the iSBP from params
            - the tSBP as iSBP
            - everything else to NaN
    """
    
    if not params['is_processed']:
        params = process_params(params)
    
    df_next = create_empty_treatment_step(params, patient_id=range(0, params['nb_patients']))

    # CAVEAT 'iteration_id', 'cycle_step', 'time_step' cannot be NaN
    # https://stackoverflow.com/questions/18429491/pandas-groupby-columns-with-nan-missing-values
    df_next['iteration_id']   = iteration_id
    df_next['cycle_step']     = cycle_step
    df_next['time_step']      = -1
    
    assert(not df_next['iteration_id'].isnull().values.any())
    assert(not df_next['cycle_step'].isnull().values.any())
    assert(not df_next['time_step'].isnull().values.any())
    
    iSBP = params['iSBP'][0] if len(params['iSBP']) == 1 else params['iSBP']
    df_next['iSBP']           = iSBP
    df_next['tSBP']           = iSBP
    
    df_next['param_id']       = params['id']
    
    return df_next

def drug_eff(params, df_next):
    """
        params: read only
        df_next: read only
        
        returns an array of drug effectiveness
    """
    
    if params['dose'] == 'FixedDose': 
        return params['drug_eff']
    
    elif params['dose'] == 'LawStandardDose':
        return 9.1+.1*(df_next['tSBP']-154)
    
    elif params['dose'] == 'LawHalfDose': 
        return (9.1+.1*(df_next['tSBP']-154))*0.78
    
    elif params['dose'] == 'rnd':
        return 10*np.random.rand()
    
    else:
        assert(False)

def treat(cycle_step, time_step):
    return False if cycle_step > 0 and time_step == 0 else True

def deliver_treatement(df_next, 
                       params):
    """
        Deliver a single treatment by:
            - overwriting `df_next['tSBP']`
            - seting `df_next['drug_eff']`
            - seting `df_next['drug_std']`
    """
    
    # set the drug effectiveness
    df_next['drug_eff'] = drug_eff(params, df_next)

    # set the drug standard deviation
    df_next['drug_std'] = params['drug_std']

    if params['drug_dist'] == 'point_mass':
        
        # point-mass distribution (no randomness)
        df_next['tSBP'] = \
            df_next['tSBP'] - df_next['drug_eff']
    
    # compute the drug effect on tSBP
    elif params['drug_dist'] == 'norm':
        
        # normal distribution
        df_next['tSBP'] = \
            df_next['tSBP'] \
            - df_next['drug_eff'] \
            + df_next['drug_std'] \
            * stats.norm.rvs(loc=0, scale=1, 
                size=len(df_next))

    elif params['drug_dist'] == 'truncnorm':
        
        # modes
        loc = df_next['tSBP'] - df_next['drug_eff']
        # clipping bounds
        a = (0 - loc) / df_next['drug_std'] # BP < 0 is not allowed
        b = (df_next['tSBP'] - loc) / df_next['drug_std']
        
        # truncated normal distribution
        vtruncnorm = np.vectorize(stats.truncnorm)
        df_next['tSBP'] = \
            [ truncnorm.rvs(1).item() for truncnorm in vtruncnorm(
                       loc=loc, 
                       scale=df_next['drug_std'], 
                       a=a, 
                       b=b)]
    
    elif params['drug_dist'] == 'lognorm':
        assert(False), 'TODO maybe?'
        # lognormal distribution
        loc = df_next['tSBP'] - df_next['drug_eff']
        s = 1 # FIXME df_next['drug_std']
        df_next['tSBP'] = stats.lognorm(loc=loc, s=s).rvs(size=len(df_next))
        
    else:
        assert(False), 'Drug distribution not implemented: {}'.format(params['drug_dist'])
    
    
    # NOTE prior to any treatment, `'treatment_step'` is NaN
    df_next.loc[~df_next['treatment_step'].isnull(), 'treatment_step'] += 1
    df_next.loc[df_next['treatment_step'].isnull(),  'treatment_step']  = 0
    
    return df_next

def nb_measurements_at(cycle_step, 
                       time_step, 
                       params):
    """
        return the required number of measurements
    """
    
    if params['measurement_strategy'] == 'none':
        
        return 0
    
    elif params['measurement_strategy'] == 'rnd':
        if cycle_step == 0:
            if time_step < params['nb_initial_titration']: # <- FIXME replace time_step by treatement_step
                return 0
            else:
                return 2
        else:
#             return np.random.randint(1, 3) # random in [1, 2]
            return 2
    
    elif params['measurement_strategy'] == 'single':
        
        if cycle_step == 0:
            if time_step < params['nb_initial_titration']: # <- FIXME replace time_step by treatement_step
                return 0
            else:
                return 1
        else:
            return 1
        
    elif params['measurement_strategy'] == 'double_original' or \
         params['measurement_strategy'] == 'double_new':
    
        if cycle_step == 0:
            if time_step < params['nb_initial_titration']:  # <- FIXME replace time_step by treatement_step
                return 0
            else:
                return 1
        else:
            return 2

    elif params['measurement_strategy'] == 'double':
        if cycle_step == 0:
            if time_step < params['nb_initial_titration']: # <- FIXME replace time_step by treatement_step
                return 0
            else:
                return 2
        else:
            return 2
        
    else:
        assert(False), 'Measurement strategy not support: {}'.format(params['measurement_strategy'])

def get_meas_std(measurement_step, params):
    
    if is_iterable(params['meas_std']):
        meas_std = params['meas_std'][measurement_step]
    else:
        meas_std = params['meas_std']
    
    return meas_std

def make_measurements(df_next, nb_measurements, params):
    """
        Make a single or multiple measurements
    """
    
    if not params['is_processed']:
        params = process_params(params)

    for measurement_step in range(nb_measurements):
    
        if measurement_step == 0:
            
            # every patients receive a first measurement
            
            df_next['meas_step'] = measurement_step
                
            if params['meas_dist'] == 'point_mass':
                df_next['mSBP'] = df_next['tSBP']
                
            elif params['meas_dist'] == 'norm':
                df_next['meas_std']  = get_meas_std(measurement_step, params)
                
                df_next['mSBP'] = \
                    df_next['tSBP'] + params['meas_shift'] \
                    + df_next['meas_std'] \
                    * stats.norm.rvs(loc=0, 
                                     scale=1, 
                                     size=len(df_next))
                
            elif params['meas_dist'] == 'skewnorm':
                df_next['meas_std']  = get_meas_std(measurement_step, params)
                
                df_next['mSBP'] = \
                    df_next['tSBP'] + params['meas_shift'] \
                    + df_next['meas_std'] \
                    * stats.skewnorm.rvs(a=params['meas_skew'], 
                                         loc=0, 
                                         scale=1, 
                                         size=len(df_next))
            else:
                assert(False), 'Not implemented'
            
        if measurement_step == 1:
            
            # NOTE the syntax of `query()` does not support dictionaries/list/arrays?
            controlled_threshold = params['controlled_threshold']
            treatment_threshold  = params['treatment_threshold']
            
            # get the index of patients who need a second measurement
            if params['measurement_strategy'] == 'double_original':
                # second measurement: we make a second measurement if the first one fell between 120mmHg and 150mmHg
                idx = df_next.query('(meas_step == 0) & (mSBP > @controlled_threshold) & (mSBP < 150)').index

            elif params['measurement_strategy'] == 'double_new':
                # second measurement: we make a second measurement if the first one is above 140mmHg
                idx = df_next.query('(meas_step == 0) & (mSBP > @treatment_threshold)').index
            
            elif params['measurement_strategy'] == 'double':
                # second measurement to everyone
                idx = df_next.query('(meas_step == 0)').index
                
            elif params['measurement_strategy'] == 'rnd':
                # second measurement to a random sample of patients
                frac = np.random.uniform(0, 1)
                idx = df_next.query('(meas_step == 0)').sample(frac=frac).index
                
            else:
                assert(False), 'Measurement strategy not supported: {}'.format(params['measurement_strategy'])
            
            tmp = df_next.loc[idx].copy(deep=True)
            tmp['meas_step'] = measurement_step
            
            if params['meas_dist'] == 'point_mass':
                tmp['mSBP'] = tmp['tSBP']
                
            elif params['meas_dist'] == 'norm':
                tmp['meas_std'] = get_meas_std(measurement_step, params)

                tmp['mSBP'] = \
                    tmp['tSBP'] + params['meas_shift'] \
                    + tmp['meas_std'] \
                    * stats.norm.rvs(loc=0, 
                                     scale=1, 
                                     size=len(tmp))
                
            elif params['meas_dist'] == 'skewnorm':
                tmp['meas_std'] = get_meas_std(measurement_step, params)

                tmp['mSBP'] = \
                    tmp['tSBP'] + params['meas_shift'] \
                    + tmp['meas_std'] \
                    * stats.skewnorm.rvs(a=params['meas_skew'], 
                                         loc=0, 
                                         scale=1, 
                                         size=len(tmp))
            
            df_next = pd.concat([df_next, tmp], axis='index')
    
    return df_next

def merge_measurements(df):
    """
        discard all measurements other than the one used for decision
    """
    
    columns = ['iteration_id', 'cycle_step', 'patient_id', 'time_step']
    
    # CAVEAT 'iteration_id', 'cycle_step', 'time_step' cannot be NaN
    # https://stackoverflow.com/questions/18429491/pandas-groupby-columns-with-nan-missing-values
    assert(not df[columns].isnull().values.any())
    
    # NOTE https://cmdlinetips.com/2019/03/how-to-get-top-n-rows-with-in-each-group-in-pandas/
    # NOTE the double brackets ensure that iloc[] returns a dataframe and not a series. A series would cast everything into floats
    # https://stackoverflow.com/questions/45990001/forcing-pandas-iloc-to-return-a-single-row-dataframe/45990057
    # NOTE the first reset_index() avoid the `ValueError: cannot reindex from a duplicate axis` error
    return \
        df.reset_index(drop=True)\
          .groupby(columns, as_index=False)\
          .apply(lambda x: x.sort_values('meas_step', ascending=False).iloc[[0]])\
          .reset_index(drop=True)

def inertia(mSBP, params, cycle_step):
    """
        compute the probability `p` of a single patient to receive a treatment 
    """
    
    assert(not is_iterable(mSBP)) # mSBP is a scalar
    
    if mSBP <= params['treatment_threshold']:
        p = 0
    elif mSBP > params['inertia_threshold']:
        p = 1
    else:
        if cycle_step < params['start_inertia_at_cycle']:
            p = 1 # no inertia
        else:
            if params['inertia_fct'] == 'hard':
                p = 1
            elif params['inertia_fct'] == 'constant':
                p = .5
            elif params['inertia_fct'] == 'linear':
                # NOTE recompute this if any of the threshold changes
                assert((params['treatment_threshold']==140) & (params['inertia_threshold']==160))
                p = .05*mSBP - 7
            elif params['inertia_fct'] == 'sqrt':
                assert(False), 'sqrt is deprecated'
                p = ((mSBP - params['treatment_threshold']) / (params['inertia_threshold'] - params['treatment_threshold']) )**.5
            elif params['inertia_fct'] == 'quadratic':
                p = ((mSBP - params['treatment_threshold']) / (params['inertia_threshold'] - params['treatment_threshold']) )**2
            elif params['inertia_fct'] == 'cubic':
                p = ((mSBP - params['treatment_threshold']) / (params['inertia_threshold'] - params['treatment_threshold']) )**3
            elif params['inertia_fct'] == 'quartic':
                p = ((mSBP - params['treatment_threshold']) / (params['inertia_threshold'] - params['treatment_threshold']) )**4
            elif params['inertia_fct'] == 'quintic':
                p = ((mSBP - params['treatment_threshold']) / (params['inertia_threshold'] - params['treatment_threshold']) )**5

            else:
                assert(False), 'Treatment inertia function not supported: {}'.format(params['inertia_fct'])
            
    assert( (p >= 0) & (p <= 1) ) # p is a probability

    return p

def make_decision(df_next, 
                  nb_measurements, 
                  params, 
                  cycle_step):
    """
        Keep or drop patients from the simulation based on some criterias
    """

    if nb_measurements > 0:

        last_measurements = merge_measurements(df_next)

        # flag the patients to keep
        vinertia = np.vectorize(inertia, otypes=[np.float64])
        vbernoulli = np.vectorize(stats.bernoulli)
        assert(not last_measurements['mSBP'].isnull().all()) # TODO decide what to do with NaNs

        # NOTE np.vectorize won't work if `params` is a row from a DataFrame (i.e. a pandas Series). 
        # np.vectorize considers a Series a vector, not as an object in itself (as it should).
        # Solution: cast `params` into a dictionary
        last_measurements['keep'] = \
            [ bernoulli.rvs(1).item() > 0 for bernoulli in \
                vbernoulli(p=vinertia(last_measurements['mSBP'], dict(params), cycle_step)) ] 
                
        patients_to_keep = last_measurements.query('keep == True')['patient_id'].values
        patients_to_drop = last_measurements.query('keep == False')['patient_id'].values        
                
        df_dropped = df_next.query('patient_id in @patients_to_drop')
        df_next = df_next.query('patient_id in @patients_to_keep')

    else:
        df_dropped = create_empty_treatment_step(params) # empty dataframe
        
    return df_next, df_dropped

def generate_next_treatment_step(df_prev, 
                                 params, 
                                 iteration_id=None, 
                                 cycle_step=None,
                                 time_step=None):
    """
        Generate the next treatment step based on the treatment history.
        
        Workflow is as follows:
        1. a treatment is given
        2. one or multiple measurement are performed
        3. patients with mSBP above threshold are stored in df_next
           patients with mSBP below threshold are stored in df_dropped
        
        NOTE: update is only performed for patients that are kept
    """
    
    assert(df_prev is not None)

    if len(df_prev) == 0:
        return df_prev, df_prev
    
    if not params['is_processed']:
        params = process_params(params)
        
    # get constant columns
    prev_iteration_id = get_unique(df_prev['iteration_id'])
    prev_cycle_step = get_unique(df_prev['cycle_step'])
    param_id = get_unique(df_prev['param_id'])

    df_prev = merge_measurements(df_prev)

    # create an empty dataframe with the same patients as `df_prev` (filled with NaN)
    df_next = create_empty_treatment_step(params, 
                                          patient_id=df_prev['patient_id'])

    # set the current time step
    prev_time_step = get_unique(df_prev['time_step'])
    curr_time_step = prev_time_step + 1 if time_step is None else time_step
    assert(not np.isnan(curr_time_step))
    assert(curr_time_step >= 0)
    df_next['time_step'] = curr_time_step

    # set the tSBP, and iSBP
    df_prev = df_prev.set_index('patient_id')
    df_next = df_next.set_index('patient_id')
    df_next.update(df_prev['iSBP'])
    df_next.update(df_prev['tSBP'])
    df_prev = df_next.reset_index()
    df_next = df_next.reset_index()
    
    # set constant columns
    curr_iteration_id = prev_iteration_id if iteration_id is None else iteration_id
    df_next['iteration_id'] = curr_iteration_id
    curr_cycle_step = prev_cycle_step if cycle_step is None else cycle_step
    df_next['cycle_step'] = curr_cycle_step
    df_next['param_id'] = param_id
        
    # deliver a single treatment
    # if no treatment is delivered, the previous tSBP is carried forward.
    do_treat = treat(curr_cycle_step, curr_time_step)
    if do_treat:
        df_next = deliver_treatement(df_next, params)

    assert(df_next['tSBP'].isna().sum() == 0) # there shouldn't be any NaN
    
    # make measurement(s)
    nb_measurements = nb_measurements_at(curr_cycle_step, 
                                         curr_time_step, 
                                         params)
    df_next = make_measurements(df_next, 
                                nb_measurements, 
                                params)
    
    if nb_measurements == 0:
        # `nb_meas` is carried forward
        assert(df_next['meas_step'].isnull().all() and
               df_next['mSBP'].isnull().all() and 
               df_next['meas_std'].isnull().all())

    # split treated and non-treated patients
    df_next, df_dropped = make_decision(df_next, 
                                        nb_measurements, 
                                        params, 
                                        curr_cycle_step)

    return df_next, df_dropped

def do_compute_next_time_step(params, time_step):
    """
        shoudl we compute the next time_step ?
    """
    
    if params['max_nb_time_steps_per_cycle'] == None:
        compute_next_time_step = True
    else:
        if time_step < params['max_nb_time_steps_per_cycle']:
            compute_next_time_step = True
        else:
            compute_next_time_step = False
        
    return compute_next_time_step

def generate_data_single_iteration(params):
    
    assert(not isinstance(params, pd.DataFrame))
    
    if not params['is_processed']:
        params = process_params(params)
    
    iteration_id = params['iteration_id']
    
    # scipy.stats just uses numpy.random to generate its random numbers, 
    #    so numpy.random.seed() will work here as well.
    if is_iterable(params['rnd_seed']):
        seed = params['rnd_seed'][iteration_id]
    else:
        seed = params['rnd_seed']
    np.random.seed(seed)
    
    data = { 'kept':[], 'dropped':[] }
    for cycle_step in range(0, params['nb_cycles']):
        
        # create the seed dataframe used in computing the first treatment step at the beginning of each cycle
        # NOTE: the seed dataframe is never added to the list `data['kept']`
        if cycle_step == 0:
            df_prev = generate_seed_treatment_step(params, iteration_id, cycle_step)
        else:
            
            # get all dropped patients at the previous cycle step and iteration_id
            tmp = pd.concat(data['dropped']) # TODO data['dropped'] is can be an empty list when we stop prematurely
            df_last_cycle_step = tmp.loc[(tmp['cycle_step']==cycle_step-1) & (tmp['iteration_id']==iteration_id)]
            df_last_cycle_step = merge_measurements(df_last_cycle_step)
            assert(df_last_cycle_step['patient_id'].nunique() == params['nb_patients'])
#             # in cases where `params['max_nb_time_steps_per_cycle']` prevented all patients to be dropped 
#             # by prematurely stopping the previous cycle
#             if df_last_cycle_step['patient_id'].nunique() < params['nb_patients']:
#                 # fetch the missing patients from `data['kept']`
# #                 assert(False), 'TODO'
#                 pass
                

            # re-introduce all patients into the simulation
            df_prev = generate_seed_treatment_step(params, iteration_id, cycle_step)
            assert(len(df_prev) == params['nb_patients'])

            # update values
            df_last_cycle_step = df_last_cycle_step.set_index('patient_id')
            df_prev = df_prev.set_index('patient_id')
            df_prev.update(df_last_cycle_step['tSBP']) # see also df1.combine_first(df2)
            df_prev = df_prev.reset_index()
    
            df_prev['iteration_id']  = iteration_id
            df_prev['cycle_step']    = cycle_step
            # tSBP at the last treatment step of the previous cycle step is set to be the iSBP of the current cycle step
            df_prev['iSBP']          = df_prev['tSBP']

        time_step = get_unique(df_prev['time_step'])
        assert(time_step == -1)
        
        compute_next_time_step = do_compute_next_time_step(params, time_step)
        while compute_next_time_step:

            df_next, df_dropped = generate_next_treatment_step(df_prev, 
                                                               params,
                                                               iteration_id, 
                                                               cycle_step,
                                                               time_step=None)
            time_step += 1

            if len(df_dropped) > 0: #NOTE here to avoid this: https://stackoverflow.com/questions/49940511/why-does-pd-concat-change-the-resulting-datatype-from-int-to-float
                data['dropped'].append(df_dropped)

            if len(df_next) == 0:
                # all patients are below threshold, start a new cycle
                break

            # there are still some patients to be treated
            data['kept'].append(df_next)
            df_prev = df_next
            
            compute_next_time_step = do_compute_next_time_step(params, time_step)

    data = concat_data(data)
    
    return data

def generate_data(params):
    """
        Run all experiements and iterations in a separate process, and combine the results
            `params`: either a dictionary, a pd.Series, or a pd.DataFrame
    """
    
    df_params = unroll_df_params(params)
    
    # convert each row into a list of pd.Series()
    params_list = [ params_ for row_id, params_ in df_params.iterrows() ]

    pool = multiprocessing.Pool(nb_cores)
    data_list = pool.map(generate_data_single_iteration, params_list)
    
    pool.close()
    pool.join()
    
    data = concat_data_(data_list)
    
    return data
