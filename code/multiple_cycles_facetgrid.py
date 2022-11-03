#!/usr/bin/env python
# coding: utf-8

"""
    Plots for Multiple Cycles
"""

import math
import random
import pickle
import copy
import functools
import itertools
from sys import stdout
import os
from functools import partial

import numpy as np

import scipy.stats as stats
import pandas as pd

import seaborn as sns

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from libs.core import now
from libs.core import get_unique
from libs.core import params_default_df
from libs.core import set_df_params_ids
from libs.core import generate_data
from libs.core import query_all
from libs.core import concat_data_
from libs.core import pname_to_latex_df
from libs.core import has_columns
from libs.core import is_iterable
from libs.core import create_dir
from libs.core import to_z_score
from libs.core import merge_measurements

from libs.plot import text_ax

# helper functions

def first_hook_function(data, 
                        params, 
                        query,
                        do_mean,
                        cumulative):
    """
        Mean or number of treatments per patient at each cycle step
    """

    data_combined = pd.concat([data['kept'], data['dropped']])

    def f(x):
                
        # keep only steps with treatments
        x = x[x['treatment_step'].notna()]
        
        x = merge_measurements(x)
        
        x = query_all(x, query)
        
        return len(x)
    
    raw_counts_per_patient = data_combined.groupby(['iteration_id', 'cycle_step', 'patient_id'], as_index=False).apply(f)

    if do_mean:
        df_plot = raw_counts_per_patient.groupby(level=['iteration_id', 'cycle_step']).mean()
    else:
        df_plot = raw_counts_per_patient.groupby(level=['iteration_id', 'cycle_step']).sum()
    
    # convert series to dataframe
    df_plot = df_plot.rename('value').to_frame().reset_index()
    
    if cumulative:
        # build cumulative plots
        # https://stackoverflow.com/questions/22650833/pandas-groupby-cumulative-sum
        df_plot = df_plot.groupby(by=['iteration_id', 'cycle_step']).sum().groupby(level='iteration_id').cumsum().reset_index()
    
    return df_plot

def second_hook_function(data, 
                         params, 
                         query,
                         do_ratio,
                         cumulative=False):
    """
        Dropped patients given a query
    """
    
    def f(x):
        
        # discard all but the decisive measurements
        x = merge_measurements(x)

        x = query_all(x, query)
        
        return len(x)

    raw_counts_per_patient = data['dropped'].groupby(['iteration_id', 'cycle_step']).apply(f)
    
    if do_ratio:
        df_plot = raw_counts_per_patient.divide(float(params['nb_patients']))
    else:
        raw_counts_per_patient = df_plot
        
    # convert series to dataframe
    df_plot = df_plot.rename('value').to_frame().reset_index()
    
    if cumulative:
        # build cumulative plots
        # https://stackoverflow.com/questions/22650833/pandas-groupby-cumulative-sum
        df_plot = df_plot.groupby(by=['iteration_id', 'cycle_step']).sum().groupby(level='iteration_id').cumsum().reset_index()

    return df_plot

def third_hook_function(data, 
                        params,
                        query):
    """
        Mean tSBP per cycle step
    """
    
    # keep only the columns we are interrested in
    df_plot = data['dropped'][['iteration_id', 'cycle_step', 'tSBP']].copy(deep=True)

    # convert form 'object' to 'float' so that we can compute the mean
    assert(pd.isnull(df_plot).any(axis=1).sum() == 0) # first make sure there is no NaN anywhere
    df_plot['tSBP'] = pd.to_numeric(df_plot['tSBP'])
    
    # mean across patient_id
    df_plot = query_all(df_plot, query).groupby(['iteration_id', 'cycle_step'], as_index=False).mean()
    
    df_plot = df_plot.rename(columns={'tSBP': 'value'})
    
    return df_plot

def forth_hook_function(data, 
                        params,
                        query,
                        decisive_only,
                        do_mean,
                        cumulative):
    """
        Mean or number of (decisive or total) measurements per cycle given a query
    """

    data_combined = pd.concat([data['kept'], data['dropped']])

    def f(x):
        
        x = x[x['meas_step'].notna()] # keep only steps with measurements
        
        if decisive_only:
            x = merge_measurements(x)

        x = query_all(x, query)
        
        return len(x)
    
    raw_counts_per_patient = data_combined.groupby(['iteration_id', 'cycle_step', 'patient_id'], as_index=False).apply(f)
    
    if do_mean:
        df_plot = raw_counts_per_patient.groupby(level=['iteration_id', 'cycle_step']).mean()
    else:
        df_plot = raw_counts_per_patient.groupby(level=['iteration_id', 'cycle_step']).sum()
    
    # convert series to dataframe
    df_plot = df_plot.rename('value').to_frame().reset_index()
    
    if cumulative:
        # build cumulative plots
        # https://stackoverflow.com/questions/22650833/pandas-groupby-cumulative-sum
        df_plot = df_plot.groupby(by=['iteration_id', 'cycle_step']).sum().groupby(level='iteration_id').cumsum().reset_index()
    
    return df_plot

def build_single_df_plot(data, params):
    """
        generate `df_plot` from `params`
        combine results from `hook_function` from multiple iterations into a single dataframe (long format)
    """

    hook_function = hook_functions[params['plot_type']]
    df_plot = hook_function(data, params)
    
    df_plot['axis_id']  = params['axis_id']
    df_plot['layer_id'] = params['layer_id']

    if 'time_step' in df_plot.columns:
        df_plot['time_step'] += 1 # such that time step starts at 1 instead of 0 in the plots
    if 'cycle_step' in df_plot.columns:
        df_plot['cycle_step'] += 1 # such that cycle step starts at 1 instead of 0 in the plots

    return df_plot

def build_overlayed_df_plot(df_params):
    """
        generate `data` and `df_plot` from `df_params` for overlayed plots
    """
    
    data = []
    df_plot = []
    for row_id, params in df_params.iterrows():
        stdout.write('\rProcessing parameter set {}/{}'.format(row_id+1, len(df_params)))
        stdout.flush()

        data_ = generate_data(params)
            
        df_plot_ = build_single_df_plot(data_, params)

        data.append(data_)
        df_plot.append(df_plot_)

    data = concat_data_(data)
    df_plot = pd.concat(df_plot, axis='index')
    
    return data, df_plot

def build_grid_df_plot(df_params):
    """
        generate `data` and `df_plot` from `df_params` for facet grid plots
    """
    
    data = []
    df_plot = []
    for axis_id, df_params_ in df_params.groupby('axis_id'): 
        
        data_, df_plot_ = build_overlayed_df_plot(df_params_)
        for k in data_.keys():
            if data_[k] is not None:
                data_[k]['axis_id'] = axis_id
        
        data.append(data_)
        df_plot.append(df_plot_)

    data = concat_data_(data)
    df_plot = pd.concat(df_plot, axis='index')
    
    return data, df_plot

def build_axis(ax, 
               axis_id, 
               df_plot, 
               df_params):
    """
        Returns a single axe for a single/overlayed plot
    """
    
    palette='cubehelix'
    
    # set the legend
    if 'show_legend' in df_params.columns:
        show_legend = get_unique(df_params['show_legend'])
    else:
        show_legend = True
        
    if show_legend:
        legend_str = 'full'
    else:
        legend_str = False

    conf_level = get_unique(df_params['conf_level'])
    
    ax = sns.pointplot(ax=ax,
                       x='cycle_step', 
                       y='value', 
                       ci=conf_level*100, # confidence level (1SD: 95 (default), 2SD: 99.7). If None, no error bars will not be drawn.
                       hue='layer_id', 
                       capsize=.2, 
                       errwidth=1.5, # thickness of error bar lines and caps (default = 3?)
                       legend=legend_str,
                       palette=palette, 
                       data=df_plot)
    plt.setp(ax.collections, sizes=[25]) # fix marker size (https://github.com/mwaskom/seaborn/issues/899) (default=50?)
    
    # transparency
    alpha = 1
    plt.setp(ax.collections, alpha=alpha) #for the markers
    plt.setp(ax.lines, alpha=alpha)  #for the lines
        
    legend = ax.get_legend()
    if not show_legend:
        legend.remove()
    else:
        
        if 'legend_loc' in df_params.columns:
            legend_loc = get_unique(df_params['legend_loc'])
        else:
            legend_loc = 'best'
            
        # remove the legend title
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles=handles, labels=labels, title='')

        # remap `layer_id` to `layer_name` in the legend
        for text_obj in legend.get_texts():
            layer_id = int(text_obj.get_text())
            layer_name = get_unique(df_params.query('(axis_id==@axis_id) & (layer_id==@layer_id)')['layer_name'])
            text_obj.set_text(layer_name)

    # set the title
    if 'show_title' in df_params.columns:
        show_title = get_unique(df_params['show_title'])
        if show_title:
            title = get_unique(df_params['title'])
            ax.set_title(title)
    
    if 'xmin' in df_params.columns:
        xmin = get_unique(df_params['xmin'])
        ax.set_xlim(left=xmin) # FIXME not sure why it has to be -1
        
    if 'xmax' in df_params.columns:
        xmax = get_unique(df_params['xmax'])
        ax.set_xlim(right=xmax)
    
    if 'ymin' in df_params.columns:
        ymin = get_unique(df_params['ymin'])
        if not np.isnan(ymin):
            ax.set_ylim(bottom=ymin)
        
    if 'ymax' in df_params.columns:
        ymax = get_unique(df_params['ymax'])
        if not np.isnan(ymax):
            ax.set_ylim(top=ymax)

    if 'xticks' in df_params.columns:
        xticks = get_unique(df_params['xticks'])
        if is_iterable(xticks):
            ax.set_xticks(xticks)
    
    if 'yticks' in df_params.columns:
        yticks = get_unique(df_params['yticks'])
        if is_iterable(yticks):
            ax.set_yticks(yticks)

    if 'show_xticklabels' in df_params.columns:
        show_xticklabels = get_unique(df_params['show_xticklabels'])
        if not show_xticklabels:
            ax.get_xaxis().set_ticklabels([])
    
    if 'show_yticklabels' in df_params.columns:
        show_yticklabels = get_unique(df_params['show_yticklabels'])
        if not show_yticklabels:
            ax.get_yaxis().set_ticklabels([])
    
    if 'ytick_label_side' in df_params.columns:
        ytick_label_side = get_unique(df_params['ytick_label_side'])
        if ytick_label_side == 'right':
            ax.yaxis.tick_right()
    
    # set x axis labels
    if 'show_xlabel' in df_params.columns:
        show_xlabel = get_unique(df_params['show_xlabel'])
        if show_xlabel:
            xlabel = get_unique(df_params['xlabel'])
            ax.set_xlabel(xlabel)
        else:
            ax.set_xlabel('')
        
    # set y axis label 
    if 'show_ylabel' in df_params.columns:
        show_ylabel = get_unique(df_params['show_ylabel'])
        if show_ylabel:

            # NOTE only a single plot type is allowed per axis
            plot_type = get_unique(df_params['plot_type'])
            ylabel = get_unique(df_params['ylabel'])

            if 'ylabel_font_size' in df_params.columns:
                ylabel_font_size = get_unique(df_params['ylabel_font_size'])
                ax.set_ylabel(ylabel, size=ylabel_font_size)
            else:
                ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel('')  
    
    ax.grid(1)
    
    return ax

# functions to build the description dataframe

def build_single_description(df_plot):
    
    assert(df_plot['axis_id'].nunique() == 1)
    assert(df_plot['layer_id'].nunique() == 1)
        
    desc = df_plot.pivot(index='iteration_id', 
                         columns='cycle_step', 
                         values='value')
    desc = desc.describe(percentiles=[]) # default percentiles=[.25, .5, .75]
    desc = desc.transpose()
    
    desc = desc.reset_index()
    
    return desc

def build_overlayed_description(df_plot):
    
    assert(df_plot['axis_id'].nunique() == 1)
    
    df_desc = []
    for layer_id, df in df_plot.groupby('layer_id'):
        
        df = build_single_description(df)

        df['layer_id'] = layer_id
        
        df_desc.append(df)

    df_desc = pd.concat(df_desc, axis='index')
    df_desc = df_desc.sort_values(['layer_id', 'cycle_step'])

    return df_desc

def build_facetgrid_description(df_plot):
    
    df_desc = []
    for axis_id, df in df_plot.groupby('axis_id'):
        
        df_desc_ = build_overlayed_description(df)

        # add a level to the index
        df_desc_['axis_id'] = axis_id

        df_desc.append(df_desc_)

    df_desc = pd.concat(df_desc, axis='index')
    
    df_desc = df_desc.sort_values(['axis_id', 'layer_id', 'cycle_step'])
    
    return df_desc

def add_params_to_description(df_desc, df_params):
    """
        add additional data from `df_params`
    """
    
    df_desc = pd.merge(df_desc, 
                       df_params, 
                       how='inner', 
                       on=['axis_id', 'layer_id'])

    df_desc['z_score']       = df_desc['conf_level'].apply(lambda x: to_z_score(x))
    df_desc['conf_interval'] = df_desc.apply(lambda x: '{:.4f} +/- {:.4f}'.format(x['mean'], x['z_score']*x['std']), axis='columns')
    
    return df_desc

def build_facetgrid_description_(df_plot, df_params):
    
    df_desc = build_facetgrid_description(df_plot)
    df_desc = add_params_to_description(df_desc, df_params)

    df_desc = df_desc[[

         'xlabel', 
         'ylabel',
         'title',

        'cycle_step', 

        'conf_level', 
        'conf_interval',

        # params

        'nb_iterations',
        'nb_cycles',
        'nb_patients',
        'iSBP',
        'age',
        'is_male',
        'ten_year_risk',
        'dose',
        'drug_eff',
        'drug_std',
        'drug_dist',
        'measurement_strategy',
        'nb_initial_titration',
        'meas_std',
        'meas_dist',
        'inertia_fct',
        'start_inertia_at_cycle'

    ]]

    df_desc['start_inertia_at_cycle'] += 1
    
    if False:
        df_desc_pretty = df_desc[['row', 'col', 'cycle_step', 'layer_name', 'conf_interval']]                             .set_index(['row', 'col', 'cycle_step', 'layer_name'])                             .unstack('layer_name')
        df_desc_pretty.columns = df_desc_pretty.columns.droplevel(0)
    
    return df_desc

def build_fixed_params_text(df_params, 
                            use_latex, 
                            pname_to_latex_df):
    """
        build the text in plots' right-hand side as a dictionary of (row, text) pairs
    """
    text = {}
    
    add_time = False
    
    # columns we are interrested in
    columns = ['row', 
               'iSBP', 
               'conf_level', 
               'meas_std', 
               'drug_std', 
               'dose', 
               'nb_initial_titration', 
               'nb_patients', 
               'inertia_fct',
               'start_inertia_at_cycle', 
               'measurement_strategy', 
               'nb_iterations',
               'drug_eff',
               'drug_dist',
               'meas_dist',
               'meas_shift',
               'meas_skew']
    if add_time:
        columns.extend(['started_at',
                        'ended_at'])
        
    columns = has_columns(df_params, columns)
    
    for row, df_ in df_params[columns].groupby('row'):
        df_ = df_.reset_index()
         
        # identify constant columns
        df_ = df_.drop('row', axis=1) # 'row' is a constant column we don't need
        df_ = df_.loc[0, (df_ == df_.iloc[0]).all()].to_frame().rename({0:'value'}, axis='columns')
        
        # keep 'drug_eff' only if at least one parameter set in df_params has 'dose' equal to 'FixedDose'
        if df_.at['dose', 'value'] != 'FixedDose':
            df_ = df_.drop(labels='drug_eff')
        
        df_.at['start_inertia_at_cycle', 'value'] += 1 # cycle 0 is labelled as cycle 1 on the plots
        
        if add_time:
            # add additional fields
            df_.at['now', 'value'] = now('%d %b %Y %H:%M:%S')
        
        def build_string(x, use_latex):
            if 'latex' in x.index and not pd.isna(x['latex']):
                text = '${}$'.format(x['latex'])
            else:
                text = '{}'.format(x.name)
            
            text += '= {}'.format(x['value'])

            text = ' '*5 + text # add tabulation
            
            return text
        
        if use_latex:
            # fetch and apply the latex
            df_ = df_.merge(pname_to_latex_df, 
                            how='left', 
                            left_index=True, 
                            right_index=True)
            
        df_['text'] = df_.apply(lambda x: build_string(x, use_latex), axis='columns')
        
        text[row] = '\n'.join(df_['text'].values)

    text = { k:'Fixed parameters:\n'+v for (k, v) in text.items() }

    return text

def facet_grid_plot(df_plot, 
                    df_params, 
                    text):
    
    nb_rows = df_params['row'].nunique()
    nb_cols = df_params['col'].nunique()
    total_nb_cols = nb_cols + 1 # including the 'Fixed parameter' axis
    figsize=(total_nb_cols*15/4., nb_rows*10/3.)
    
    fig, axes = plt.subplots(nb_rows, 
                             total_nb_cols, 
                             figsize=figsize)

    if 'layout_w_pad' in df_params.columns and 'layout_h_pad' in df_params.columns:
        w_pad = get_unique(df_params['layout_w_pad'])
        h_pad = get_unique(df_params['layout_h_pad'])
        plt.tight_layout(w_pad=w_pad, h_pad=h_pad) # pad=
    else:
        fig.tight_layout()
    
    for row_idx, col_idx in itertools.product(range(nb_rows), range(total_nb_cols)):
        
        if nb_rows == 1:
            ax = axes[col_idx]
        else:
            ax = axes[row_idx][col_idx]

        if col_idx < nb_cols:
            axis_id = row_idx*nb_cols + col_idx

            df_ = df_plot.query('axis_id == @axis_id')
            df_params_ = df_params[df_params['axis_id']==axis_id]

            # plot the graph
            ax = build_axis(ax, 
                            axis_id, 
                            df_, 
                            df_params_)
            
            # get the axis dimensions
            xlim   = ax.get_xlim()
            ylim   = ax.get_ylim()
            width  = xlim[1] - xlim[0]
            height = ylim[1] - ylim[0]
            
            # show axis id
            if 'show_axis_label' in df_params_.columns:
                show_axis_label = get_unique(df_params_['show_axis_label'])
            else:
                show_axis_label = False
            
            if show_axis_label:
                axis_label = get_unique(df_params_['axis_label'])
                ax.text(.5*width, .5*height, 
                         str('Axis {}'.format(axis_id)),
                         fontsize=60, 
                         alpha=.2, 
                         color='red', 
                         ha='center', 
                         va='center')

        elif col_idx == nb_cols:
            # sets the bottom left corner of text area in bottom left corner of the axis
            ax = text_ax(ax, text[row_idx], loc=[0, ylim[0]])
            
    return fig

# Parameters

def build_overlayed_plot_params():

    inertia_fcts = ['hard', 'constant', 'linear', 'quadratic', 'quartic']

    df_params = pd.concat([params_default_df]*len(inertia_fcts)).reset_index(drop=True)
    
    df_params['plot_type']              = 0
    df_params['iSBP']                   = 160
    df_params['nb_patients']            = 100
    df_params['nb_cycles']              = 10
    df_params['nb_iterations']          = 5
    df_params['start_inertia_at_cycle'] = 0
    
    # treatment
    df_params['inertia_fct']          =  inertia_fcts
    df_params['drug_dist']            = 'truncnorm'
    df_params['dose']                 = 'LawStandardDose' # 'FixedDose' 'LawHalfDose2'
    df_params['drug_std']             = 1
    df_params['nb_initial_titration'] = 0

    # measurement
    df_params['meas_dist']            = 'norm'
    df_params['measurement_strategy'] = 'double_new'
    df_params['meas_std']             = 10

    df_params['rnd_seed']             = df_params.apply(lambda x: np.random.randint(2**32, size=x['nb_iterations']).tolist(), axis='columns')

    # plot specific
    df_params['conf_level']          = .997
    df_params['axis_id']             = 0
    df_params['layer_id']            = list(range(len(df_params)))
    df_params['layer_name']          = df_params.apply(lambda x: inertia_fcts[x['layer_id']], axis='columns') # string to be shown in the legend
    df_params['hue']                 = df_params['layer_id']
    df_params['show_title']          = True
    df_params['show_legend']         = True
    df_params['show_xlabel']         = True
    df_params['show_ylabel']         = True
    df_params['title']               = df_params.apply(lambda x: '${}$ mmHg'.format(x['iSBP']), axis='columns')
    df_params['xlabel']              = 'Cycle'
    df_params['ylabel']              = df_params.apply(lambda x: ylabels[x['plot_type']], axis='columns')

    df_params['show_xticklabels']     = True
    df_params['show_yticklabels']     = True

    df_params['row']                  = 0
    df_params['col']                  = 0
    
    df_params = set_df_params_ids(df_params)

    return df_params

def treatment_inertia_paper_figure_02_grid_params():

    df_params = [None]*4
    
    inertia_fcts = ['hard', 'constant', 'linear', 'quadratic', 'quartic' ]
    measurement_strategy = 'single'
    
    # row 0, col 0
    df_params[0] = pd.concat([params_default_df]*len(inertia_fcts))
    df_params[0]['row'] = 0
    df_params[0]['col'] = 0
    df_params[0]['plot_type'] = 4
    df_params[0]['inertia_fct'] = inertia_fcts
    df_params[0]['xticks'] = np.nan
    df_params[0]['yticks'] = [list(np.arange(0, 1.1, .2))] * len(df_params[0])
    df_params[0]['ymin'] = -.05
    df_params[0]['ymax'] = 1.1
    df_params[0]['show_legend'] = False
    
    # row 0, col 1
    df_params[1] = pd.concat([params_default_df] * len(inertia_fcts))
    df_params[1]['row'] = 0
    df_params[1]['col'] = 1
    df_params[1]['plot_type'] = 5
    df_params[1]['inertia_fct'] = inertia_fcts
    df_params[1]['xticks'] = np.nan
    df_params[1]['yticks'] = [list(np.arange(0, .6, .2))] * len(df_params[1])
    df_params[1]['ymin'] = -.05
    df_params[1]['ymax'] = .6
    df_params[1]['show_legend'] = True
    
    # row 1, col 0
    df_params[2] = pd.concat([params_default_df] * len(inertia_fcts))
    df_params[2]['row'] = 1
    df_params[2]['col'] = 0
    df_params[2]['plot_type'] = 3
    df_params[2]['inertia_fct'] = inertia_fcts
    df_params[2]['xticks'] = np.nan
    df_params[2]['yticks'] = [list(np.arange(120, 155, 5))] * len(df_params[2])
    df_params[2]['ymin'] = 120
    df_params[2]['ymax'] = 153
    df_params[2]['show_legend'] = False
    
    # row 1, col 1
    df_params[3] = pd.concat([params_default_df] * len(inertia_fcts))
    df_params[3]['row'] = 1
    df_params[3]['col'] = 1
    df_params[3]['plot_type'] = 0
    df_params[3]['inertia_fct'] = inertia_fcts
    df_params[3]['xticks'] = np.nan
    if measurement_strategy == 'single':
        df_params[3]['yticks'] = [list(np.arange(0, 6, 1))] * len(df_params[3])
        df_params[3]['ymin'] = 0
        df_params[3]['ymax'] = 6
    elif measurement_strategy == 'double_new':
        df_params[3]['yticks'] = [list(np.arange(0, 5, 1))] * len(df_params[3])
        df_params[3]['ymin'] = 0
        df_params[3]['ymax'] = 4.5
    else:
        assert(False)
    df_params[3]['show_legend'] = False
    
    df_params = pd.concat(df_params).reset_index(drop=True)

    df_params['axis_id'] = df_params['row']*(df_params['col'].max()+1) + df_params['col']

    df_params['iSBP']                   = 160
    df_params['start_inertia_at_cycle'] = 0
    df_params['nb_initial_titration']   = 0
    df_params['dose']                   = 'LawStandardDose'
    df_params['drug_dist']              = 'truncnorm'
    df_params['drug_std']               = 1
    df_params['meas_dist']              = 'norm'
    df_params['meas_std']               = 10
    df_params['nb_patients']            = 100
    df_params['nb_iterations']          = 15
    df_params['nb_cycles']              = 10
    df_params['measurement_strategy']   = measurement_strategy

    # string to be shown in the legend
    df_params['layer_name'] = df_params['inertia_fct']
    
    df_params['rnd_seed'] = df_params.apply(lambda x: np.random.randint(2**32, size=x['nb_iterations']).tolist(), axis='columns')
    
    df_params['layer_id'] = -1
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'layer_id'] = list(range(len(inertia_fcts)))
        
    # plot parameters
    df_params['conf_level'] = .95
    
    df_params['show_xlabel'] = False
    df_params.loc[df_params['row']==df_params['row'].max(), 'show_xlabel'] = True
    
    df_params['show_ylabel'] = True
    
    df_params['show_title'] = False
    df_params['title']      = ''

    df_params['xlabel']               = 'Cycle'
    df_params['ylabel']               = df_params.apply(lambda x: ylabels[x['plot_type']], axis='columns')
    
    # axis label for each axis
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'show_axis_label'] = False
        df_params.loc[df_params['axis_id']==axis_id, 'axis_label'] = df_params.apply(lambda x: 'Axis {}'.format(x['axis_id']), axis='columns')
    
    df_params['show_yticklabels'] = True

    df_params['show_xticklabels'] = False
    df_params.loc[df_params['row'] == df_params['row'].max(), 'show_xticklabels'] = True
    
    df_params['ytick_label_side'] = 'left'

    df_params['legend_loc'] = 'upper right'
    
    df_params['xmin'] = -1
    df_params['xmax'] = get_unique(df_params['nb_cycles'])
    
    df_params = set_df_params_ids(df_params)
    
    return df_params

def treatment_inertia_paper_figure_03_grid_params(iSBP):

    df_params = [None]*4
    
    measurement_strategy = 'single'
    meas_std = [5, 10, 15]
    inertia_fcts = ['hard', 'constant', 'linear', 'quadratic', 'quartic' ]
    
    # row 0
    df_params[0] = pd.concat([params_default_df]*len(meas_std)*len(inertia_fcts))
    df_params[0]['row'] = 0
    df_params[0]['plot_type'] = 4
    df_params[0]['inertia_fct'] = np.tile(inertia_fcts, len(meas_std))
    df_params[0]['meas_std'] = np.repeat(meas_std, len(inertia_fcts))
    df_params[0]['xticks'] = np.nan
    df_params[0]['yticks'] = [list(np.arange(0, 1.1, .2))] * len(df_params[0])
    df_params[0]['ymin'] = 0
    df_params[0]['ymax'] = 1.1
    
    # row 1
    df_params[1] = pd.concat([params_default_df]*len(meas_std)*len(inertia_fcts))
    df_params[1]['row'] = 1
    df_params[1]['plot_type'] = 5
    df_params[1]['inertia_fct'] = np.tile(inertia_fcts, len(meas_std))
    df_params[1]['meas_std'] = np.repeat(meas_std, len(inertia_fcts))
    df_params[1]['xticks'] = np.nan
    df_params[1]['yticks'] = [list(np.arange(0, .6, .2))] * len(df_params[1])
    df_params[1]['ymin'] = -.05
    df_params[1]['ymax'] = .6
    
    # row 2
    df_params[2] = pd.concat([params_default_df]*len(meas_std)*len(inertia_fcts))
    df_params[2]['row'] = 2
    df_params[2]['plot_type'] = 3
    df_params[2]['inertia_fct'] = np.tile(inertia_fcts, len(meas_std))
    df_params[2]['meas_std'] = np.repeat(meas_std, len(inertia_fcts))
    df_params[2]['xticks'] = np.nan
    df_params[2]['yticks'] = [list(np.arange(115, 155, 5))] * len(df_params[2])
    df_params[2]['ymin'] = 115
    df_params[2]['ymax'] = 153
    
    # row 3
    df_params[3] = pd.concat([params_default_df]*len(meas_std)*len(inertia_fcts))
    df_params[3]['row'] = 3
    df_params[3]['plot_type'] = 0
    df_params[3]['inertia_fct'] = np.tile(inertia_fcts, len(meas_std))
    df_params[3]['meas_std'] = np.repeat(meas_std, len(inertia_fcts))
    df_params[3]['xticks'] = np.nan
    if measurement_strategy == 'single':
        if iSBP == 170:
            df_params[3]['yticks'] = [list(np.arange(0, 7, 1))] * len(df_params[3])
            df_params[3]['ymin'] = 0
            df_params[3]['ymax'] = 6.5
        else:
            df_params[3]['yticks'] = [list(np.arange(0, 6, 1))] * len(df_params[3])
            df_params[3]['ymin'] = 0
            df_params[3]['ymax'] = 6
    elif measurement_strategy == 'double_new':
        df_params[3]['yticks'] = [list(np.arange(0, 5, 1))] * len(df_params[3])
        df_params[3]['ymin'] = 0
        df_params[3]['ymax'] = 5
    else:
        assert(False)
        
    df_params = pd.concat(df_params).reset_index(drop=True)

    df_params['col'] = -1
    df_params.loc[df_params['meas_std']==5,  'col'] = 0
    df_params.loc[df_params['meas_std']==10, 'col'] = 1
    df_params.loc[df_params['meas_std']==15, 'col'] = 2

    df_params['axis_id'] = df_params['row'] * (df_params['col'].max()+1) + df_params['col']

    df_params['iSBP']                   = iSBP
    df_params['start_inertia_at_cycle'] = 0
    df_params['nb_initial_titration']   = 0
    df_params['dose']                   = 'LawStandardDose'
    df_params['drug_dist']              = 'truncnorm'
    df_params['drug_std']               = 1
    df_params['meas_dist']              = 'norm'

    # string to be shown in the legend
    df_params['layer_name'] = df_params['inertia_fct']
    
    df_params['nb_patients']          = 100
    df_params['nb_iterations']        = 15
    df_params['nb_cycles']            = 10
    df_params['measurement_strategy'] = measurement_strategy
    
    df_params['rnd_seed'] = df_params.apply(lambda x: np.random.randint(2**32, size=x['nb_iterations']).tolist(), axis='columns')
    
    df_params['layer_id'] = -1
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'layer_id'] = list(range(len(inertia_fcts)))
        
    # plot specific
    df_params['conf_level'] = .95
    
    df_params['show_title'] = False
    df_params.loc[df_params['row']==df_params['row'].min(), 'show_title'] = True
    
    df_params['show_xlabel'] = False
    df_params.loc[df_params['row']==df_params['row'].max(), 'show_xlabel'] = True
    
    df_params['show_ylabel'] = False
    df_params.loc[df_params['col']==df_params['col'].min(), 'show_ylabel'] = True
    
    df_params['show_legend'] = False
    df_params.loc[(df_params['row']==df_params['row'].min())
                  & (df_params['col']==df_params['col'].max()) , 
                  'show_legend'] = True # top-right axis
    
    df_params['title']  = df_params.apply(lambda x: '${}$ mmHg'.format(x['meas_std']), axis='columns')
    df_params['xlabel'] = 'Cycle'
    df_params['ylabel'] = df_params.apply(lambda x: ylabels[x['plot_type']], axis='columns')

    # axis label for each axis
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'show_axis_label'] = False
        df_params.loc[df_params['axis_id']==axis_id, 'axis_label'] = df_params.apply(lambda x: 'Axis {}'.format(x['axis_id']), axis='columns')
        
    df_params['show_yticklabels'] = False
    df_params.loc[df_params['col'] == df_params['col'].min(), 'show_yticklabels'] = True

    df_params['show_xticklabels'] = False
    df_params.loc[df_params['row'] == df_params['row'].max(), 'show_xticklabels'] = True
    
    # pad between subplots
    df_params['layout_w_pad'] = -3
    df_params['layout_h_pad'] = -2
    
    df_params['legend_loc'] = 'lower right'
    
    df_params['xmin'] = -1
    df_params['xmax'] = get_unique(df_params['nb_cycles'])
    
    df_params = set_df_params_ids(df_params)
    
    return df_params

def treatment_inertia_paper_figure_03_extension_grid_params(iSBP):

    df_params = [None]*2
    
    measurement_strategy = 'single'
    meas_std = [5, 10, 15]
    inertia_fcts = ['hard', 'constant', 'linear', 'quadratic', 'quartic' ]
    
    # row 0
    df_params[0] = pd.concat([params_default_df]*len(meas_std)*len(inertia_fcts))
    df_params[0]['row'] = 0
    df_params[0]['plot_type'] = 15
    df_params[0]['inertia_fct'] = np.tile(inertia_fcts, len(meas_std))
    df_params[0]['meas_std'] = np.repeat(meas_std, len(inertia_fcts))
    df_params[0]['xticks'] = np.nan
    df_params[0]['yticks'] = [list(np.arange(115, 155, 5))] * len(df_params[0])
    df_params[0]['ymin'] = 115
    df_params[0]['ymax'] = 153
    
    # row 1
    df_params[1] = pd.concat([params_default_df]*len(meas_std)*len(inertia_fcts))
    df_params[1]['row'] = 1
    df_params[1]['plot_type'] = 16
    df_params[1]['inertia_fct'] = np.tile(inertia_fcts, len(meas_std))
    df_params[1]['meas_std'] = np.repeat(meas_std, len(inertia_fcts))
    df_params[1]['xticks'] = np.nan
    df_params[1]['yticks'] = [list(np.arange(115, 155, 5))] * len(df_params[1])
    df_params[1]['ymin'] = 115
    df_params[1]['ymax'] = 153
        
    df_params = pd.concat(df_params).reset_index(drop=True)

    df_params['col'] = -1
    df_params.loc[df_params['meas_std']==5,  'col'] = 0
    df_params.loc[df_params['meas_std']==10, 'col'] = 1
    df_params.loc[df_params['meas_std']==15, 'col'] = 2

    df_params['axis_id'] = df_params['row'] * (df_params['col'].max()+1) + df_params['col']

    df_params['iSBP']                   = iSBP
    df_params['start_inertia_at_cycle'] = 0
    df_params['nb_initial_titration']   = 0
    df_params['dose']                   = 'LawStandardDose'
    df_params['drug_dist']              = 'truncnorm'
    df_params['drug_std']               = 1
    df_params['meas_dist']              = 'norm'

    # string to be shown in the legend
    df_params['layer_name'] = df_params['inertia_fct']
    
    df_params['nb_patients']          = 1000
    df_params['nb_iterations']        = 15
    df_params['nb_cycles']            = 10
    df_params['measurement_strategy'] = measurement_strategy
    
    df_params['rnd_seed'] = df_params.apply(lambda x: np.random.randint(2**32, size=x['nb_iterations']).tolist(), axis='columns')
    
    df_params['layer_id'] = -1
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'layer_id'] = list(range(len(inertia_fcts)))
        
    # plot specific
    df_params['conf_level'] = .95
    
    df_params['show_title'] = False
    df_params.loc[df_params['row']==df_params['row'].min(), 'show_title'] = True
    
    df_params['show_xlabel'] = False
    df_params.loc[df_params['row']==df_params['row'].max(), 'show_xlabel'] = True
    
    df_params['show_ylabel'] = False
    df_params.loc[df_params['col']==df_params['col'].min(), 'show_ylabel'] = True
    
    df_params['show_legend'] = False
    df_params.loc[(df_params['row']==df_params['row'].min())
                  & (df_params['col']==df_params['col'].max()) , 
                  'show_legend'] = True # top-right axis
    
    df_params['title']  = df_params.apply(lambda x: '${}$ mmHg'.format(x['meas_std']), axis='columns')
    df_params['xlabel'] = 'Cycle'
    df_params['ylabel'] = df_params.apply(lambda x: ylabels[x['plot_type']], axis='columns')

    # axis label for each axis
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'show_axis_label'] = False
        df_params.loc[df_params['axis_id']==axis_id, 'axis_label'] = df_params.apply(lambda x: 'Axis {}'.format(x['axis_id']), axis='columns')
        
    df_params['show_yticklabels'] = False
    df_params.loc[df_params['col'] == df_params['col'].min(), 'show_yticklabels'] = True

    df_params['show_xticklabels'] = False
    df_params.loc[df_params['row'] == df_params['row'].max(), 'show_xticklabels'] = True
    
    # pad between subplots
    df_params['layout_w_pad'] = -3
    df_params['layout_h_pad'] = -2
    
    df_params['legend_loc'] = 'lower right'
    
    df_params['xmin'] = -1
    df_params['xmax'] = get_unique(df_params['nb_cycles'])
    
    df_params = set_df_params_ids(df_params)
    
    return df_params

def treatment_inertia_paper_figure_04_grid_params():

    df_params = [None]*3
    
    meas_std = [5, 10, 15]
    inertia_fcts = ['hard', 'constant', 'linear', 'quadratic', 'quartic' ]
    
    # row 0
    df_params[0] = pd.concat([params_default_df]*len(meas_std)*len(inertia_fcts))
    df_params[0]['row'] = 0
    df_params[0]['start_inertia_at_cycle'] = 0
    df_params[0]['measurement_strategy'] = 'double_new'
    df_params[0]['inertia_fct'] = np.tile(inertia_fcts, len(meas_std))
    df_params[0]['meas_std'] = np.repeat(meas_std, len(inertia_fcts))
    df_params[0]['xticks'] = np.nan
    df_params[0]['yticks'] = [list(np.arange(0, 1.1, .2))] * len(df_params[0])
    df_params[0]['ymin'] = 0
    df_params[0]['ymax'] = 1.1
    
    # row 1
    df_params[1] = pd.concat([params_default_df]*len(meas_std)*len(inertia_fcts))
    df_params[1]['row'] = 1
    df_params[1]['start_inertia_at_cycle'] = 1
    df_params[1]['measurement_strategy'] = 'single'
    df_params[1]['inertia_fct'] = np.tile(inertia_fcts, len(meas_std))
    df_params[1]['meas_std'] = np.repeat(meas_std, len(inertia_fcts))
    df_params[1]['xticks'] = np.nan
    df_params[1]['yticks'] = [list(np.arange(0, 1.1, .2))] * len(df_params[1])
    df_params[1]['ymin'] = 0
    df_params[1]['ymax'] = 1.1
    
    # row 2
    df_params[2] = pd.concat([params_default_df]*len(meas_std)*len(inertia_fcts))
    df_params[2]['row'] = 2
    df_params[2]['start_inertia_at_cycle'] = 1
    df_params[2]['measurement_strategy'] = 'double_new'
    df_params[2]['inertia_fct'] = np.tile(inertia_fcts, len(meas_std))
    df_params[2]['meas_std'] = np.repeat(meas_std, len(inertia_fcts))
    df_params[2]['xticks'] = np.nan
    df_params[2]['yticks'] = [list(np.arange(0, 1.1, .2))] * len(df_params[2])
    df_params[2]['ymin'] = 0
    df_params[2]['ymax'] = 1.1
    
    df_params = pd.concat(df_params).reset_index(drop=True)

    df_params['col'] = -1
    df_params.loc[df_params['meas_std']==5,  'col'] = 0
    df_params.loc[df_params['meas_std']==10, 'col'] = 1
    df_params.loc[df_params['meas_std']==15, 'col'] = 2

    df_params['axis_id'] = df_params['row'] * (df_params['col'].max()+1) + df_params['col']

    df_params['plot_type']              = 4
    df_params['iSBP']                   = 160
    df_params['nb_initial_titration']   = 0
    df_params['dose']                   = 'LawStandardDose'
    df_params['drug_dist']              = 'truncnorm'
    df_params['drug_std']               = 1
    df_params['meas_dist']              = 'norm'

    # string to be shown in the legend
    df_params['layer_name'] = df_params['inertia_fct']
    
    df_params['nb_patients']          = 100
    df_params['nb_iterations']        = 15
    df_params['nb_cycles']            = 10
    
    df_params['rnd_seed'] = df_params.apply(lambda x: np.random.randint(2**32, size=x['nb_iterations']).tolist(), axis='columns')
    
    df_params['layer_id'] = -1
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'layer_id'] = list(range(len(inertia_fcts)))
        
    # plot specific
    df_params['conf_level'] = .95
    
    df_params['show_title'] = False
    df_params.loc[df_params['row']==df_params['row'].min(), 'show_title'] = True
    
    df_params['show_xlabel'] = False
    df_params.loc[df_params['row']==df_params['row'].max(), 'show_xlabel'] = True
    
    df_params['show_ylabel'] = False
    df_params.loc[df_params['col']==df_params['col'].min(), 'show_ylabel'] = True
    
    df_params['show_legend'] = False
    df_params.loc[(df_params['row']==df_params['row'].min())
                  & (df_params['col']==df_params['col'].max()) , 
                  'show_legend'] = True # top-left axis
    
    df_params['title']  = df_params.apply(lambda x: '${}$ mmHg'.format(x['meas_std']), axis='columns')
    df_params['xlabel'] = 'Cycle'
    df_params['ylabel'] = df_params.apply(lambda x: ylabels[x['plot_type']], axis='columns')

    # axis label for each axis
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'show_axis_label'] = False
        df_params.loc[df_params['axis_id']==axis_id, 'axis_label'] = df_params.apply(lambda x: 'Axis {}'.format(x['axis_id']), axis='columns')
        
    df_params['show_yticklabels'] = False
    df_params.loc[df_params['col'] == df_params['col'].min(), 'show_yticklabels'] = True

    df_params['show_xticklabels'] = False
    df_params.loc[df_params['row'] == df_params['row'].max(), 'show_xticklabels'] = True
    
    # pad between subplots
    df_params['layout_w_pad'] = -3
    df_params['layout_h_pad'] = -2
    
    df_params['ylabel_font_size'] = 12
    
    df_params['legend_loc'] = 'lower right'
    
    df_params['xmin'] = -1
    df_params['xmax'] = get_unique(df_params['nb_cycles'])
    
    df_params = set_df_params_ids(df_params)
    
    return df_params

def treatment_inertia_paper_figure_04_extension_grid_params():

    df_params = [None]*6
    
    meas_std = [5, 10, 15]
    inertia_fcts = ['hard', 'constant', 'linear', 'quadratic', 'quartic' ]
    
    # row 0
    df_params[0] = pd.concat([params_default_df]*len(meas_std)*len(inertia_fcts))
    df_params[0]['row'] = 0
    df_params[0]['plot_type'] = 15
    df_params[0]['measurement_strategy'] = 'double_new'
    df_params[0]['start_inertia_at_cycle'] = 0
    df_params[0]['inertia_fct'] = np.tile(inertia_fcts, len(meas_std))
    df_params[0]['meas_std'] = np.repeat(meas_std, len(inertia_fcts))
    df_params[0]['xticks'] = np.nan
    df_params[0]['yticks'] = [list(np.arange(115, 155, 5))] * len(df_params[0])
    df_params[0]['ymin'] = 115
    df_params[0]['ymax'] = 153
    
    # row 1
    df_params[1] = pd.concat([params_default_df]*len(meas_std)*len(inertia_fcts))
    df_params[1]['row'] = 1
    df_params[1]['plot_type'] = 16
    df_params[1]['measurement_strategy'] = 'double_new'
    df_params[1]['start_inertia_at_cycle'] = 0
    df_params[1]['inertia_fct'] = np.tile(inertia_fcts, len(meas_std))
    df_params[1]['meas_std'] = np.repeat(meas_std, len(inertia_fcts))
    df_params[1]['xticks'] = np.nan
    df_params[1]['yticks'] = [list(np.arange(115, 155, 5))] * len(df_params[1])
    df_params[1]['ymin'] = 115
    df_params[1]['ymax'] = 153
        
    # row 2
    df_params[2] = pd.concat([params_default_df]*len(meas_std)*len(inertia_fcts))
    df_params[2]['row'] = 2
    df_params[2]['plot_type'] = 15
    df_params[2]['measurement_strategy'] = 'single'
    df_params[2]['start_inertia_at_cycle'] = 1
    df_params[2]['inertia_fct'] = np.tile(inertia_fcts, len(meas_std))
    df_params[2]['meas_std'] = np.repeat(meas_std, len(inertia_fcts))
    df_params[2]['xticks'] = np.nan
    df_params[2]['yticks'] = [list(np.arange(115, 155, 5))] * len(df_params[2])
    df_params[2]['ymin'] = 115
    df_params[2]['ymax'] = 153
    
    # row 3
    df_params[3] = pd.concat([params_default_df]*len(meas_std)*len(inertia_fcts))
    df_params[3]['row'] = 3
    df_params[3]['plot_type'] = 16
    df_params[3]['measurement_strategy'] = 'single'
    df_params[3]['start_inertia_at_cycle'] = 1
    df_params[3]['inertia_fct'] = np.tile(inertia_fcts, len(meas_std))
    df_params[3]['meas_std'] = np.repeat(meas_std, len(inertia_fcts))
    df_params[3]['xticks'] = np.nan
    df_params[3]['yticks'] = [list(np.arange(115, 155, 5))] * len(df_params[3])
    df_params[3]['ymin'] = 115
    df_params[3]['ymax'] = 153
        
    # row 4
    df_params[4] = pd.concat([params_default_df]*len(meas_std)*len(inertia_fcts))
    df_params[4]['row'] = 4
    df_params[4]['plot_type'] = 15
    df_params[4]['measurement_strategy'] = 'double_new'
    df_params[4]['start_inertia_at_cycle'] = 1
    df_params[4]['inertia_fct'] = np.tile(inertia_fcts, len(meas_std))
    df_params[4]['meas_std'] = np.repeat(meas_std, len(inertia_fcts))
    df_params[4]['xticks'] = np.nan
    df_params[4]['yticks'] = [list(np.arange(115, 155, 5))] * len(df_params[4])
    df_params[4]['ymin'] = 115
    df_params[4]['ymax'] = 153
    
    # row 5
    df_params[5] = pd.concat([params_default_df]*len(meas_std)*len(inertia_fcts))
    df_params[5]['row'] = 5
    df_params[5]['plot_type'] = 16
    df_params[5]['measurement_strategy'] = 'double_new'
    df_params[5]['start_inertia_at_cycle'] = 1
    df_params[5]['inertia_fct'] = np.tile(inertia_fcts, len(meas_std))
    df_params[5]['meas_std'] = np.repeat(meas_std, len(inertia_fcts))
    df_params[5]['xticks'] = np.nan
    df_params[5]['yticks'] = [list(np.arange(115, 155, 5))] * len(df_params[5])
    df_params[5]['ymin'] = 115
    df_params[5]['ymax'] = 153
        
    df_params = pd.concat(df_params).reset_index(drop=True)

    df_params['col'] = -1
    df_params.loc[df_params['meas_std']==5,  'col'] = 0
    df_params.loc[df_params['meas_std']==10, 'col'] = 1
    df_params.loc[df_params['meas_std']==15, 'col'] = 2

    df_params['axis_id'] = df_params['row'] * (df_params['col'].max()+1) + df_params['col']

    df_params['iSBP']                   = 160
    df_params['nb_initial_titration']   = 0
    df_params['dose']                   = 'LawStandardDose'
    df_params['drug_dist']              = 'truncnorm'
    df_params['drug_std']               = 1
    df_params['meas_dist']              = 'norm'

    # string to be shown in the legend
    df_params['layer_name'] = df_params['inertia_fct']
    
    df_params['nb_patients']          = 1000
    df_params['nb_iterations']        = 15
    df_params['nb_cycles']            = 10
    
    df_params['rnd_seed'] = df_params.apply(lambda x: np.random.randint(2**32, size=x['nb_iterations']).tolist(), axis='columns')
    
    df_params['layer_id'] = -1
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'layer_id'] = list(range(len(inertia_fcts)))
        
    # plot specific
    df_params['conf_level'] = .95
    
    df_params['show_title'] = False
    df_params.loc[df_params['row']==df_params['row'].min(), 'show_title'] = True
    
    df_params['show_xlabel'] = False
    df_params.loc[df_params['row']==df_params['row'].max(), 'show_xlabel'] = True
    
    df_params['show_ylabel'] = False
    df_params.loc[df_params['col']==df_params['col'].min(), 'show_ylabel'] = True
    
    df_params['show_legend'] = False
    df_params.loc[(df_params['row']==df_params['row'].min())
                  & (df_params['col']==df_params['col'].max()) , 
                  'show_legend'] = True # top-right axis
    
    df_params['title']  = df_params.apply(lambda x: '${}$ mmHg'.format(x['meas_std']), axis='columns')
    df_params['xlabel'] = 'Cycle'
    df_params['ylabel'] = df_params.apply(lambda x: ylabels[x['plot_type']], axis='columns')

    # axis label for each axis
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'show_axis_label'] = False
        df_params.loc[df_params['axis_id']==axis_id, 'axis_label'] = df_params.apply(lambda x: 'Axis {}'.format(x['axis_id']), axis='columns')
        
    df_params['show_yticklabels'] = False
    df_params.loc[df_params['col'] == df_params['col'].min(), 'show_yticklabels'] = True

    df_params['show_xticklabels'] = False
    df_params.loc[df_params['row'] == df_params['row'].max(), 'show_xticklabels'] = True
    
    # pad between subplots
    df_params['layout_w_pad'] = -3
    df_params['layout_h_pad'] = -2
    
    df_params['legend_loc'] = 'lower right'
    
    df_params['xmin'] = -1
    df_params['xmax'] = get_unique(df_params['nb_cycles'])
    
    df_params = set_df_params_ids(df_params)
    
    return df_params

def treatment_inertia_paper_sup_materials_figure_03_grid_params():

    df_params = [None]*2
    
    measurement_strategy = 'single'
    iSBP = [150, 160, 170]
    inertia_fcts = ['hard', 'constant', 'linear', 'quadratic', 'quartic' ]
    
    # row 0
    df_params[0] = pd.concat([params_default_df]*len(iSBP)*len(inertia_fcts))
    df_params[0]['row'] = 0
    df_params[0]['plot_type'] = 6
    df_params[0]['iSBP'] = np.repeat(iSBP, len(inertia_fcts))
    df_params[0]['inertia_fct'] = np.tile(inertia_fcts, len(iSBP))
    df_params[0]['xticks'] = np.nan
    df_params[0]['yticks'] = [list(np.arange(0, 1.1, .2))] * len(df_params[0])
    df_params[0]['ymin'] = 0
    df_params[0]['ymax'] = 1.1
    
    # row 1
    df_params[1] = pd.concat([params_default_df]*len(iSBP)*len(inertia_fcts))
    df_params[1]['row'] = 1
    df_params[1]['plot_type'] = 4
    df_params[1]['iSBP'] = np.repeat(iSBP, len(inertia_fcts))
    df_params[1]['inertia_fct'] = np.tile(inertia_fcts, len(iSBP))
    df_params[1]['xticks'] = np.nan
    df_params[1]['yticks'] = [list(np.arange(0, 1.1, .2))] * len(df_params[1])
    df_params[1]['ymin'] = 0
    df_params[1]['ymax'] = 1.1
    
    df_params = pd.concat(df_params).reset_index(drop=True)

    df_params['col'] = -1
    df_params.loc[df_params['iSBP']==150, 'col'] = 0
    df_params.loc[df_params['iSBP']==160, 'col'] = 1
    df_params.loc[df_params['iSBP']==170, 'col'] = 2

    df_params['axis_id'] = df_params['row'] * (df_params['col'].max()+1) + df_params['col']

    df_params['nb_initial_titration']   = 0
    df_params['start_inertia_at_cycle'] = 0
    df_params['measurement_strategy']   = measurement_strategy
    df_params['meas_std']               = 10
    df_params['dose']                   = 'LawStandardDose'
    df_params['drug_dist']              = 'truncnorm'
    df_params['drug_std']               = 1
    df_params['meas_dist']              = 'norm'

    # string to be shown in the legend
    df_params['layer_name'] = df_params['inertia_fct']
    
    df_params['nb_patients']          = 100
    df_params['nb_iterations']        = 15
    df_params['nb_cycles']            = 10
    
    df_params['rnd_seed'] = df_params.apply(lambda x: np.random.randint(2**32, size=x['nb_iterations']).tolist(), axis='columns')
    
    df_params['layer_id'] = -1
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'layer_id'] = list(range(len(inertia_fcts)))
        
    # plot specific
    df_params['conf_level'] = .95
    
    df_params['show_title'] = False
    df_params.loc[df_params['row']==df_params['row'].min(), 'show_title'] = True
    
    df_params['show_xlabel'] = False
    df_params.loc[df_params['row']==df_params['row'].max(), 'show_xlabel'] = True
    
    df_params['show_ylabel'] = False
    df_params.loc[df_params['col']==df_params['col'].min(), 'show_ylabel'] = True
    
    df_params['show_legend'] = False
    df_params.loc[(df_params['row']==df_params['row'].min())
                  & (df_params['col']==df_params['col'].max()) , 
                  'show_legend'] = True # top-left axis
    
    df_params['title']  = df_params.apply(lambda x: '${}$ mmHg'.format(x['iSBP']), axis='columns')
    df_params['xlabel'] = 'Cycle'
    df_params['ylabel'] = df_params.apply(lambda x: ylabels[x['plot_type']], axis='columns')

    # axis label for each axis
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'show_axis_label'] = False
        df_params.loc[df_params['axis_id']==axis_id, 'axis_label'] = df_params.apply(lambda x: 'Axis {}'.format(x['axis_id']), axis='columns')
        
    df_params['show_yticklabels'] = False
    df_params.loc[df_params['col'] == df_params['col'].min(), 'show_yticklabels'] = True

    df_params['show_xticklabels'] = False
    df_params.loc[df_params['row'] == df_params['row'].max(), 'show_xticklabels'] = True
    
    # pad between subplots
    df_params['layout_w_pad'] = -3
    df_params['layout_h_pad'] = -2
    
    df_params['ylabel_font_size'] = 12
    
    df_params['legend_loc'] = 'lower right'
    
    df_params['xmin'] = -1
    df_params['xmax'] = get_unique(df_params['nb_cycles'])
    
    df_params = set_df_params_ids(df_params)
    
    return df_params

def treatment_inertia_paper_sup_materials_figure_04_grid_params():

    df_params = [None]*4
    
    iSBP = [150, 160, 170]
    inertia_fcts = ['hard', 'constant', 'linear', 'quadratic', 'quartic' ]
    measurement_strategy = 'single'
    
    # row 0
    df_params[0] = pd.concat([params_default_df]*len(iSBP)*len(inertia_fcts))
    df_params[0]['row'] = 0
    df_params[0]['plot_type'] = 4
    df_params[0]['iSBP'] = np.repeat(iSBP, len(inertia_fcts))
    df_params[0]['inertia_fct'] = np.tile(inertia_fcts, len(iSBP))
    df_params[0]['xticks'] = np.nan
    df_params[0]['yticks'] = [list(np.arange(0, 1.1, .2))] * len(iSBP) * len(inertia_fcts)
    df_params[0]['ymin'] = -.05
    df_params[0]['ymax'] = 1.1
    
    # row 1
    df_params[1] = pd.concat([params_default_df]*len(iSBP)*len(inertia_fcts))
    df_params[1]['row'] = 1
    df_params[1]['plot_type'] = 5
    df_params[1]['iSBP'] = np.repeat(iSBP, len(inertia_fcts))
    df_params[1]['inertia_fct'] = np.tile(inertia_fcts, len(iSBP))
    df_params[1]['xticks'] = np.nan
    df_params[1]['yticks'] = [list(np.arange(0, .6, .2))] * len(iSBP) * len(inertia_fcts)
    df_params[1]['ymin'] = -.05
    df_params[1]['ymax'] = .6
    
    # row 2
    df_params[2] = pd.concat([params_default_df]*len(iSBP)*len(inertia_fcts))
    df_params[2]['row'] = 2
    df_params[2]['plot_type'] = 3
    df_params[2]['iSBP'] = np.repeat(iSBP, len(inertia_fcts))
    df_params[2]['inertia_fct'] = np.tile(inertia_fcts, len(iSBP))
    df_params[2]['xticks'] = np.nan
    df_params[2]['yticks'] = [list(np.arange(120, 155, 5))] * len(iSBP) * len(inertia_fcts)
    df_params[2]['ymin'] = 120
    df_params[2]['ymax'] = 153
    
    # row 3
    df_params[3] = pd.concat([params_default_df]*len(iSBP)*len(inertia_fcts))
    df_params[3]['row'] = 3
    df_params[3]['plot_type'] = 0
    df_params[3]['iSBP'] = np.repeat(iSBP, len(inertia_fcts))
    df_params[3]['inertia_fct'] = np.tile(inertia_fcts, len(iSBP))
    df_params[3]['ymin'] = 0
    df_params[3]['xticks'] = np.nan
    if measurement_strategy == 'single':
        df_params[3]['yticks'] = [list(np.arange(0, 6, 1))] * len(iSBP)*len(inertia_fcts)
        df_params[3]['ymin'] = 0
        df_params[3]['ymax'] = 6
    elif measurement_strategy == 'double_new':
        df_params[3]['yticks'] = [list(np.arange(0, 6, 1))] * len(iSBP) * len(inertia_fcts)
        df_params[3]['ymin'] = 0
        df_params[3]['ymax'] = 5
    else:
        assert(False)
    
    df_params = pd.concat(df_params).reset_index(drop=True)
   
    df_params['col'] = -1
    df_params.loc[df_params['iSBP']==150, 'col'] = 0
    df_params.loc[df_params['iSBP']==160, 'col'] = 1
    df_params.loc[df_params['iSBP']==170, 'col'] = 2

    df_params['axis_id'] = df_params['row']*(df_params['col'].max()+1) + df_params['col']

    df_params['start_inertia_at_cycle'] = 0
    df_params['nb_initial_titration']   = 0
    df_params['dose']                   = 'LawStandardDose'
    df_params['drug_dist']              = 'truncnorm'
    df_params['drug_std']               = 1
    df_params['meas_dist']              = 'norm'
    df_params['meas_std']               = 10

    # string to be shown in the legend
    df_params['layer_name'] = df_params['inertia_fct']
        
    df_params['nb_patients']          = 10
    df_params['nb_iterations']        = 5
    df_params['nb_cycles']            = 10
    df_params['measurement_strategy'] = measurement_strategy
    
    df_params['rnd_seed'] = df_params.apply(lambda x: np.random.randint(2**32, size=x['nb_iterations']).tolist(), axis='columns')
    
    df_params['layer_id'] = -1
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'layer_id'] = list(range(len(inertia_fcts)))
    
        
    # plot specific
    df_params['conf_level'] = .95
    
    df_params['show_title'] = False
    df_params.loc[df_params['row']==df_params['row'].min(), 'show_title'] = True
    
    df_params['show_xlabel'] = False
    df_params.loc[df_params['row']==df_params['row'].max(), 'show_xlabel'] = True
    
    df_params['show_ylabel'] = False
    df_params.loc[df_params['col']==df_params['col'].min(), 'show_ylabel'] = True
    
    df_params['show_legend'] = False
    df_params.loc[(df_params['row']==df_params['row'].min())
                  & (df_params['col']==df_params['col'].max()) , 
                  'show_legend'] = True # top-right axis
    
    df_params['title']  = df_params.apply(lambda x: '${}$ mmHg'.format(x['iSBP']), axis='columns')
    df_params['xlabel'] = 'Cycle'
    df_params['ylabel'] = df_params.apply(lambda x: ylabels[x['plot_type']], axis='columns')

    # axis label for each axis
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'show_axis_label'] = False
        df_params.loc[df_params['axis_id']==axis_id, 'axis_label'] = df_params.apply(lambda x: 'Axis {}'.format(x['axis_id']), axis='columns')
        
    df_params['show_yticklabels'] = False
    df_params.loc[df_params['col'] == df_params['col'].min(), 'show_yticklabels'] = True

    df_params['show_xticklabels'] = False
    df_params.loc[df_params['row'] == df_params['row'].max(), 'show_xticklabels'] = True
    
    # pad between subplots
    df_params['layout_w_pad'] = -3
    df_params['layout_h_pad'] = -2
    
    df_params['legend_loc'] = 'lower right'
    
    df_params['xmin'] = -1
    df_params['xmax'] = get_unique(df_params['nb_cycles'])
    
    df_params = set_df_params_ids(df_params)
    
    return df_params

def treatment_inertia_paper_sup_materials_figure_06_grid_params(t):

    df_params = [None]*4
    
    if t=='a':
        start_inertia_at_cycle = 0
        measurement_strategy = 'double_new'
    elif t=='b':
        start_inertia_at_cycle = 1
        measurement_strategy = 'single'
    elif t=='c':
        start_inertia_at_cycle = 1
        measurement_strategy = 'double_new'
    meas_std = [5, 10, 15]
    inertia_fcts = ['hard', 'constant', 'linear', 'quadratic', 'quartic' ]
    
    # row 0
    df_params[0] = pd.concat([params_default_df]*len(meas_std)*len(inertia_fcts))
    df_params[0]['row'] = 0
    df_params[0]['plot_type'] = 4
    df_params[0]['inertia_fct'] = np.tile(inertia_fcts, len(meas_std))
    df_params[0]['meas_std'] = np.repeat(meas_std, len(inertia_fcts))
    df_params[0]['xticks'] = np.nan
    df_params[0]['yticks'] = [list(np.arange(0, 1.1, .2))] * len(df_params[0])
    df_params[0]['ymin'] = 0
    df_params[0]['ymax'] = 1.1
    
    # row 1
    df_params[1] = pd.concat([params_default_df]*len(meas_std)*len(inertia_fcts))
    df_params[1]['row'] = 1
    df_params[1]['plot_type'] = 5
    df_params[1]['inertia_fct'] = np.tile(inertia_fcts, len(meas_std))
    df_params[1]['meas_std'] = np.repeat(meas_std, len(inertia_fcts))
    df_params[1]['xticks'] = np.nan
    df_params[1]['yticks'] = [list(np.arange(0, .6, .2))] * len(df_params[1])
    df_params[1]['ymin'] = -.05
    df_params[1]['ymax'] = .6
    
    # row 2
    df_params[2] = pd.concat([params_default_df]*len(meas_std)*len(inertia_fcts))
    df_params[2]['row'] = 2
    df_params[2]['plot_type'] = 3
    df_params[2]['inertia_fct'] = np.tile(inertia_fcts, len(meas_std))
    df_params[2]['meas_std'] = np.repeat(meas_std, len(inertia_fcts))
    df_params[2]['xticks'] = np.nan
    if t=='b':
        df_params[2]['yticks'] = [list(np.arange(115, 155, 5))] * len(df_params[2])
        df_params[2]['ymin'] = 115
        df_params[2]['ymax'] = 153
    else:
        df_params[2]['yticks'] = [list(np.arange(120, 155, 5))] * len(df_params[2])
        df_params[2]['ymin'] = 120
        df_params[2]['ymax'] = 153
    
    # row 3
    df_params[3] = pd.concat([params_default_df]*len(meas_std)*len(inertia_fcts))
    df_params[3]['row'] = 3
    df_params[3]['plot_type'] = 0
    df_params[3]['inertia_fct'] = np.tile(inertia_fcts, len(meas_std))
    df_params[3]['meas_std'] = np.repeat(meas_std, len(inertia_fcts))
    df_params[3]['xticks'] = np.nan
    if measurement_strategy == 'single':
        df_params[3]['yticks'] = [list(np.arange(0, 6, 1))] * len(df_params[3])
        df_params[3]['ymin'] = 0
        df_params[3]['ymax'] = 6
    elif measurement_strategy == 'double_new':
        df_params[3]['yticks'] = [list(np.arange(0, 5, 1))] * len(df_params[3])
        df_params[3]['ymin'] = 0
        df_params[3]['ymax'] = 5
    else:
        assert(False)
        
    df_params = pd.concat(df_params).reset_index(drop=True)

    df_params['col'] = -1
    df_params.loc[df_params['meas_std']==5,  'col'] = 0
    df_params.loc[df_params['meas_std']==10, 'col'] = 1
    df_params.loc[df_params['meas_std']==15, 'col'] = 2

    df_params['axis_id'] = df_params['row'] * (df_params['col'].max()+1) + df_params['col']

    df_params['iSBP']                   = 160
    df_params['start_inertia_at_cycle'] = start_inertia_at_cycle
    df_params['nb_initial_titration']   = 0
    df_params['dose']                   = 'LawStandardDose'
    df_params['drug_dist']              = 'truncnorm'
    df_params['drug_std']               = 1
    df_params['meas_dist']              = 'norm'

    # string to be shown in the legend
    df_params['layer_name'] = df_params['inertia_fct']
        
    df_params['nb_patients']          = 100
    df_params['nb_iterations']        = 15
    df_params['nb_cycles']            = 10
    df_params['measurement_strategy'] = measurement_strategy
    
    df_params['rnd_seed'] = df_params.apply(lambda x: np.random.randint(2**32, size=x['nb_iterations']).tolist(), axis='columns')
    
    df_params['layer_id'] = -1
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'layer_id'] = list(range(len(inertia_fcts)))
        
    # plot specific
    df_params['conf_level'] = .95
    
    df_params['show_title'] = False
    df_params.loc[df_params['row']==df_params['row'].min(), 'show_title'] = True
    
    df_params['show_xlabel'] = False
    df_params.loc[df_params['row']==df_params['row'].max(), 'show_xlabel'] = True
    
    df_params['show_ylabel'] = False
    df_params.loc[df_params['col']==df_params['col'].min(), 'show_ylabel'] = True
    
    df_params['show_legend'] = False
    df_params.loc[(df_params['row']==df_params['row'].min())
                  & (df_params['col']==df_params['col'].max()) , 
                  'show_legend'] = True # top-right axis
    
    df_params['title']  = df_params.apply(lambda x: '${}$ mmHg'.format(x['meas_std']), axis='columns')
    df_params['xlabel'] = 'Cycle'
    df_params['ylabel'] = df_params.apply(lambda x: ylabels[x['plot_type']], axis='columns')

    # axis label for each axis
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'show_axis_label'] = False
        df_params.loc[df_params['axis_id']==axis_id, 'axis_label'] = df_params.apply(lambda x: 'Axis {}'.format(x['axis_id']), axis='columns')
        
    df_params['show_yticklabels'] = False
    df_params.loc[df_params['col'] == df_params['col'].min(), 'show_yticklabels'] = True

    df_params['show_xticklabels'] = False
    df_params.loc[df_params['row'] == df_params['row'].max(), 'show_xticklabels'] = True
    
    # pad between subplots
    df_params['layout_w_pad'] = -3
    df_params['layout_h_pad'] = -2
    
    df_params['legend_loc'] = 'lower right'
    
    df_params['xmin'] = -1
    df_params['xmax'] = get_unique(df_params['nb_cycles'])
    
    df_params = set_df_params_ids(df_params)
    
    return df_params

def treatment_inertia_paper_sup_materials_figure_07_grid_params():

    df_params = [None]*4
    
    meas_shift = [0, 5, 10]
    inertia_fcts = ['hard', 'constant', 'linear', 'quadratic', 'quartic' ]
    measurement_strategy = 'single'
    
    # row 0
    df_params[0] = pd.concat([params_default_df]*len(meas_shift)*len(inertia_fcts))
    df_params[0]['row'] = 0
    df_params[0]['plot_type'] = 4
    df_params[0]['meas_shift'] = np.repeat(meas_shift, len(inertia_fcts))
    df_params[0]['inertia_fct'] = np.tile(inertia_fcts, len(meas_shift))
    df_params[0]['xticks'] = np.nan
    df_params[0]['yticks'] = [list(np.arange(0, 1.1, .2))] * len(meas_shift) * len(inertia_fcts)
    df_params[0]['ymin'] = -.05
    df_params[0]['ymax'] = 1.1
    
    # row 1
    df_params[1] = pd.concat([params_default_df]*len(meas_shift)*len(inertia_fcts))
    df_params[1]['row'] = 1
    df_params[1]['plot_type'] = 5
    df_params[1]['meas_shift'] = np.repeat(meas_shift, len(inertia_fcts))
    df_params[1]['inertia_fct'] = np.tile(inertia_fcts, len(meas_shift))
    df_params[1]['xticks'] = np.nan
    df_params[1]['yticks'] = [list(np.arange(0, 1.1, .2))] * len(meas_shift) * len(inertia_fcts)
    df_params[1]['ymin'] = -.05
    df_params[1]['ymax'] = 1.1
    
    # row 2
    df_params[2] = pd.concat([params_default_df]*len(meas_shift)*len(inertia_fcts))
    df_params[2]['row'] = 2
    df_params[2]['plot_type'] = 3
    df_params[2]['meas_shift'] = np.repeat(meas_shift, len(inertia_fcts))
    df_params[2]['inertia_fct'] = np.tile(inertia_fcts, len(meas_shift))
    df_params[2]['xticks'] = np.nan
    df_params[2]['yticks'] = [list(np.arange(110, 155, 5))] * len(meas_shift) * len(inertia_fcts)
    df_params[2]['ymin'] = 110
    df_params[2]['ymax'] = 153
    
    # row 3
    df_params[3] = pd.concat([params_default_df]*len(meas_shift)*len(inertia_fcts))
    df_params[3]['row'] = 3
    df_params[3]['plot_type'] = 0
    df_params[3]['meas_shift'] = np.repeat(meas_shift, len(inertia_fcts))
    df_params[3]['inertia_fct'] = np.tile(inertia_fcts, len(meas_shift))
    df_params[3]['ymin'] = 0
    df_params[3]['xticks'] = np.nan
    if measurement_strategy == 'single':
        df_params[3]['yticks'] = [list(np.arange(0, 6, 1))] * len(meas_shift)*len(inertia_fcts)
        df_params[3]['ymin'] = 0
        df_params[3]['ymax'] = 6.5
    elif measurement_strategy == 'double_new':
        df_params[3]['yticks'] = [list(np.arange(0, 6, 1))] * len(meas_shift) * len(inertia_fcts)
        df_params[3]['ymin'] = 0
        df_params[3]['ymax'] = 5
    else:
        assert(False)
    
    df_params = pd.concat(df_params).reset_index(drop=True)
   
    df_params['col'] = -1
    df_params.loc[df_params['meas_shift']==meas_shift[0], 'col'] = 0
    df_params.loc[df_params['meas_shift']==meas_shift[1], 'col'] = 1
    df_params.loc[df_params['meas_shift']==meas_shift[2], 'col'] = 2

    df_params['axis_id'] = df_params['row']*(df_params['col'].max()+1) + df_params['col']

    df_params['iSBP']                   = 160
    df_params['start_inertia_at_cycle'] = 0
    df_params['nb_initial_titration']   = 0
    df_params['dose']                   = 'LawStandardDose'
    df_params['drug_dist']              = 'truncnorm'
    df_params['drug_std']               = 1
    df_params['meas_dist']              = 'norm'
    df_params['meas_std']               = 10

    # string to be shown in the legend
    df_params['layer_name'] = df_params['inertia_fct']
        
    df_params['nb_patients']          = 1000
    df_params['nb_iterations']        = 15
    df_params['nb_cycles']            = 10
    df_params['measurement_strategy'] = measurement_strategy
    
    df_params['rnd_seed'] = df_params.apply(lambda x: np.random.randint(2**32, size=x['nb_iterations']).tolist(), axis='columns')
    
    df_params['layer_id'] = -1
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'layer_id'] = list(range(len(inertia_fcts)))
    
        
    # plot specific
    df_params['conf_level'] = .95
    
    df_params['show_title'] = False
    df_params.loc[df_params['row']==df_params['row'].min(), 'show_title'] = True
    
    df_params['show_xlabel'] = False
    df_params.loc[df_params['row']==df_params['row'].max(), 'show_xlabel'] = True
    
    df_params['show_ylabel'] = False
    df_params.loc[df_params['col']==df_params['col'].min(), 'show_ylabel'] = True
    
    df_params['show_legend'] = False
    df_params.loc[(df_params['row']==df_params['row'].min())
                  & (df_params['col']==df_params['col'].max()) , 
                  'show_legend'] = True # top-right axis
    
    df_params['title']  = df_params.apply(lambda x: '${}$ mmHg'.format(x['meas_shift']), axis='columns')
    df_params['xlabel'] = 'Cycle'
    df_params['ylabel'] = df_params.apply(lambda x: ylabels[x['plot_type']], axis='columns')

    # axis label for each axis
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'show_axis_label'] = False
        df_params.loc[df_params['axis_id']==axis_id, 'axis_label'] = df_params.apply(lambda x: 'Axis {}'.format(x['axis_id']), axis='columns')
        
    df_params['show_yticklabels'] = False
    df_params.loc[df_params['col'] == df_params['col'].min(), 'show_yticklabels'] = True

    df_params['show_xticklabels'] = False
    df_params.loc[df_params['row'] == df_params['row'].max(), 'show_xticklabels'] = True
    
    # pad between subplots
    df_params['layout_w_pad'] = -3
    df_params['layout_h_pad'] = -2
    
    df_params['legend_loc'] = 'lower right'
    
    df_params['xmin'] = -1
    df_params['xmax'] = get_unique(df_params['nb_cycles'])
    
    df_params = set_df_params_ids(df_params)
    
    return df_params

def treatment_inertia_paper_sup_materials_figure_08_grid_params():

    df_params = [None]*4
    
    meas_skew = [0, 1, 40]
    inertia_fcts = ['hard', 'constant', 'linear', 'quadratic', 'quartic' ]
    measurement_strategy = 'single'
    
    # row 0
    df_params[0] = pd.concat([params_default_df]*len(meas_skew)*len(inertia_fcts))
    df_params[0]['row'] = 0
    df_params[0]['plot_type'] = 4
    df_params[0]['meas_skew'] = np.repeat(meas_skew, len(inertia_fcts))
    df_params[0]['inertia_fct'] = np.tile(inertia_fcts, len(meas_skew))
    df_params[0]['xticks'] = np.nan
    df_params[0]['yticks'] = [list(np.arange(0, 1.1, .2))] * len(meas_skew) * len(inertia_fcts)
    df_params[0]['ymin'] = -.05
    df_params[0]['ymax'] = 1.1
    
    # row 1
    df_params[1] = pd.concat([params_default_df]*len(meas_skew)*len(inertia_fcts))
    df_params[1]['row'] = 1
    df_params[1]['plot_type'] = 5
    df_params[1]['meas_skew'] = np.repeat(meas_skew, len(inertia_fcts))
    df_params[1]['inertia_fct'] = np.tile(inertia_fcts, len(meas_skew))
    df_params[1]['xticks'] = np.nan
    df_params[1]['yticks'] = [list(np.arange(0, .6, .2))] * len(meas_skew) * len(inertia_fcts)
    df_params[1]['ymin'] = -.05
    df_params[1]['ymax'] = .6
    
    # row 2
    df_params[2] = pd.concat([params_default_df]*len(meas_skew)*len(inertia_fcts))
    df_params[2]['row'] = 2
    df_params[2]['plot_type'] = 3
    df_params[2]['meas_skew'] = np.repeat(meas_skew, len(inertia_fcts))
    df_params[2]['inertia_fct'] = np.tile(inertia_fcts, len(meas_skew))
    df_params[2]['xticks'] = np.nan
    df_params[2]['yticks'] = [list(np.arange(115, 155, 5))] * len(meas_skew) * len(inertia_fcts)
    df_params[2]['ymin'] = 115
    
    # row 3
    df_params[3] = pd.concat([params_default_df]*len(meas_skew)*len(inertia_fcts))
    df_params[3]['row'] = 3
    df_params[3]['plot_type'] = 0
    df_params[3]['meas_skew'] = np.repeat(meas_skew, len(inertia_fcts))
    df_params[3]['inertia_fct'] = np.tile(inertia_fcts, len(meas_skew))
    df_params[3]['ymin'] = 0
    df_params[3]['xticks'] = np.nan
    if measurement_strategy == 'single':
        df_params[3]['yticks'] = [list(np.arange(0, 7, 1))] * len(meas_skew)*len(inertia_fcts)
        df_params[3]['ymin'] = 0
        df_params[3]['ymax'] = 6.5
    elif measurement_strategy == 'double_new':
        df_params[3]['yticks'] = [list(np.arange(0, 6, 1))] * len(meas_skew) * len(inertia_fcts)
        df_params[3]['ymin'] = 0
        df_params[3]['ymax'] = 5
    else:
        assert(False)
    
    df_params = pd.concat(df_params).reset_index(drop=True)
   
    df_params['col'] = -1
    df_params.loc[df_params['meas_skew']==meas_skew[0], 'col'] = 0
    df_params.loc[df_params['meas_skew']==meas_skew[1], 'col'] = 1
    df_params.loc[df_params['meas_skew']==meas_skew[2], 'col'] = 2

    df_params['axis_id'] = df_params['row']*(df_params['col'].max()+1) + df_params['col']

    df_params['iSBP']                   = 160
    df_params['start_inertia_at_cycle'] = 0
    df_params['nb_initial_titration']   = 0
    df_params['dose']                   = 'LawStandardDose'
    df_params['drug_dist']              = 'truncnorm'
    df_params['drug_std']               = 1
    df_params['meas_dist']              = 'skewnorm'
    df_params['meas_std']               = 10

    # string to be shown in the legend
    df_params['layer_name'] = df_params['inertia_fct']
        
    df_params['nb_patients']          = 1000
    df_params['nb_iterations']        = 15
    df_params['nb_cycles']            = 10
    df_params['measurement_strategy'] = measurement_strategy
    
    df_params['rnd_seed'] = df_params.apply(lambda x: np.random.randint(2**32, size=x['nb_iterations']).tolist(), axis='columns')
    
    df_params['layer_id'] = -1
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'layer_id'] = list(range(len(inertia_fcts)))
    
        
    # plot specific
    df_params['conf_level'] = .95
    
    df_params['show_title'] = False
    df_params.loc[df_params['row']==df_params['row'].min(), 'show_title'] = True
    
    df_params['show_xlabel'] = False
    df_params.loc[df_params['row']==df_params['row'].max(), 'show_xlabel'] = True
    
    df_params['show_ylabel'] = False
    df_params.loc[df_params['col']==df_params['col'].min(), 'show_ylabel'] = True
    
    df_params['show_legend'] = False
    df_params.loc[(df_params['row']==df_params['row'].min())
                  & (df_params['col']==df_params['col'].max()) , 
                  'show_legend'] = True # top-right axis
    
    df_params['title']  = df_params.apply(lambda x: '${}$'.format(x['meas_skew']), axis='columns')
    df_params['xlabel'] = 'Cycle'
    df_params['ylabel'] = df_params.apply(lambda x: ylabels[x['plot_type']], axis='columns')

    # axis label for each axis
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'show_axis_label'] = False
        df_params.loc[df_params['axis_id']==axis_id, 'axis_label'] = df_params.apply(lambda x: 'Axis {}'.format(x['axis_id']), axis='columns')
        
    df_params['show_yticklabels'] = False
    df_params.loc[df_params['col'] == df_params['col'].min(), 'show_yticklabels'] = True

    df_params['show_xticklabels'] = False
    df_params.loc[df_params['row'] == df_params['row'].max(), 'show_xticklabels'] = True
    
    # pad between subplots
    df_params['layout_w_pad'] = -3
    df_params['layout_h_pad'] = -2
    
    df_params['legend_loc'] = 'lower right'
    
    df_params['xmin'] = -1
    df_params['xmax'] = get_unique(df_params['nb_cycles'])
    
    df_params = set_df_params_ids(df_params)
    
    return df_params

def treatment_inertia_paper_figure_extra_grid_params():

    df_params = [None]*3
    
    inertia_fcts = ['hard', 'constant', 'linear', 'quadratic', 'quartic' ]
    measurement_strategy = 'single'
#     measurement_strategy = 'double_new'
    
    # row 0, col 0   
    df_params[0] = pd.concat([params_default_df] * len(inertia_fcts))
    df_params[0]['row'] = 0
    df_params[0]['col'] = 0
    df_params[0]['plot_type'] = 0
    df_params[0]['inertia_fct'] = inertia_fcts
    df_params[0]['xticks'] = np.nan
    df_params[0]['ymin'] = 0
    df_params[0]['show_legend'] = False
    
    # row 0, col 1
    df_params[1] = pd.concat([params_default_df] * len(inertia_fcts))
    df_params[1]['row'] = 0
    df_params[1]['col'] = 1
    df_params[1]['plot_type'] = 8
    df_params[1]['inertia_fct'] = inertia_fcts
    df_params[1]['xticks'] = np.nan
    df_params[1]['yticks'] = [list(np.arange(0, 15, 2))] * len(df_params[1])
    df_params[1]['ymin'] = -.5
    df_params[1]['ymax'] = 15
    df_params[1]['show_legend'] = False
    
    # row 0, col 2
    df_params[2] = pd.concat([params_default_df] * len(inertia_fcts))
    df_params[2]['row'] = 0
    df_params[2]['col'] = 2
    df_params[2]['plot_type'] = 9#7
    df_params[2]['inertia_fct'] = inertia_fcts
    df_params[2]['xticks'] = np.nan
    df_params[2]['ymin'] = -5
    df_params[2]['show_legend'] = True
    
    df_params = pd.concat(df_params).reset_index(drop=True)

    df_params['axis_id'] = df_params['row']*(df_params['col'].max()+1) + df_params['col']

    df_params['iSBP']                   = 160
    df_params['start_inertia_at_cycle'] = 0
    df_params['nb_initial_titration']   = 0
    df_params['dose']                   = 'LawStandardDose'
    df_params['drug_dist']              = 'point_mass' #'truncnorm'
    df_params['drug_std']               = 1
    df_params['meas_dist']              = 'point_mass' #'norm'
    df_params['meas_std']               = 10
    df_params['nb_patients']            = 100
    df_params['nb_iterations']          = 15
    df_params['nb_cycles']              = 10
    df_params['measurement_strategy']   = measurement_strategy

    # string to be shown in the legend
    df_params['layer_name'] = df_params['inertia_fct']
    
    df_params['rnd_seed'] = df_params.apply(lambda x: np.random.randint(2**32, size=x['nb_iterations']).tolist(), axis='columns')
    
    df_params['layer_id'] = -1
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'layer_id'] = list(range(len(inertia_fcts)))
        
    # plot parameters
    df_params['conf_level'] = .95
    
    df_params['show_xlabel'] = False
    df_params.loc[df_params['row']==df_params['row'].max(), 'show_xlabel'] = True
    
    df_params['show_ylabel'] = True
    
    df_params['show_title'] = False
    df_params['title']      = ''

    df_params['xlabel']               = 'Cycle'
    df_params['ylabel']               = df_params.apply(lambda x: ylabels[x['plot_type']], axis='columns')
    
    # axis label for each axis
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'show_axis_label'] = False
        df_params.loc[df_params['axis_id']==axis_id, 'axis_label'] = df_params.apply(lambda x: 'Axis {}'.format(x['axis_id']), axis='columns')
    
    df_params['show_yticklabels'] = True

    df_params['show_xticklabels'] = False
    df_params.loc[df_params['row'] == df_params['row'].max(), 'show_xticklabels'] = True
    
    df_params['ytick_label_side'] = 'left'

    df_params['legend_loc'] = 'upper right'
    
    df_params['xmin'] = -1
    df_params['xmax'] = get_unique(df_params['nb_cycles'])
    
    df_params = set_df_params_ids(df_params)
    
    return df_params

def treatment_inertia_paper_figure_extra_grid_params_02():

    df_params = [None]*8
    
    inertia_fcts = ['hard', 'constant', 'linear', 'quadratic', 'quartic' ]
    measurement_strategy = 'single'
    
    # row 0, col 0
    df_params[0] = pd.concat([params_default_df] * len(inertia_fcts))
    df_params[0]['row'] = 0
    df_params[0]['col'] = 0
    df_params[0]['plot_type'] = 7
    df_params[0]['inertia_fct'] = inertia_fcts
    
    # row 0, col 1
    df_params[1] = pd.concat([params_default_df] * len(inertia_fcts))
    df_params[1]['row'] = 0
    df_params[1]['col'] = 1
    df_params[1]['plot_type'] = 8
    df_params[1]['inertia_fct'] = inertia_fcts
    
    # row 1, col 0
    df_params[2] = pd.concat([params_default_df] * len(inertia_fcts))
    df_params[2]['row'] = 1
    df_params[2]['col'] = 0
    df_params[2]['plot_type'] = 9
    df_params[2]['inertia_fct'] = inertia_fcts
    
    # row 1, col 1
    df_params[3] = pd.concat([params_default_df] * len(inertia_fcts))
    df_params[3]['row'] = 1
    df_params[3]['col'] = 1
    df_params[3]['plot_type'] = 10
    df_params[3]['inertia_fct'] = inertia_fcts
    
    # row 2, col 0
    df_params[4] = pd.concat([params_default_df] * len(inertia_fcts))
    df_params[4]['row'] = 2
    df_params[4]['col'] = 0
    df_params[4]['plot_type'] = 11
    df_params[4]['inertia_fct'] = inertia_fcts
    
    # row 2, col 1
    df_params[5] = pd.concat([params_default_df] * len(inertia_fcts))
    df_params[5]['row'] = 2
    df_params[5]['col'] = 1
    df_params[5]['plot_type'] = 12
    df_params[5]['inertia_fct'] = inertia_fcts
    
    # row 3, col 0
    df_params[6] = pd.concat([params_default_df] * len(inertia_fcts))
    df_params[6]['row'] = 3
    df_params[6]['col'] = 0
    df_params[6]['plot_type'] = 13
    df_params[6]['inertia_fct'] = inertia_fcts
    
    # row 3, col 1
    df_params[7] = pd.concat([params_default_df] * len(inertia_fcts))
    df_params[7]['row'] = 3
    df_params[7]['col'] = 1
    df_params[7]['plot_type'] = 14
    df_params[7]['inertia_fct'] = inertia_fcts
    
    df_params = pd.concat(df_params).reset_index(drop=True)

    df_params['axis_id'] = df_params['row']*(df_params['col'].max()+1) + df_params['col']

    df_params['iSBP']                   = 160
    df_params['start_inertia_at_cycle'] = 0
    df_params['nb_initial_titration']   = 0
    df_params['dose']                   = 'LawStandardDose'
    df_params['drug_dist']              = 'point_mass'
    df_params['drug_std']               = 1
    df_params['meas_dist']              = 'point_mass'
    df_params['meas_std']               = 10
    df_params['nb_patients']            = 100
    df_params['nb_iterations']          = 5
    df_params['nb_cycles']              = 10
    df_params['measurement_strategy']   = measurement_strategy

    # string to be shown in the legend
    df_params['layer_name'] = df_params['inertia_fct']
    
    df_params['rnd_seed'] = df_params.apply(lambda x: np.random.randint(2**32, size=x['nb_iterations']).tolist(), axis='columns')
    
    df_params['layer_id'] = -1
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'layer_id'] = list(range(len(inertia_fcts)))
        
    # plot parameters
    df_params['conf_level'] = .95
    
    df_params['show_xlabel'] = False
    df_params.loc[df_params['row']==df_params['row'].max(), 'show_xlabel'] = True
    
    df_params['show_ylabel'] = True
    
    df_params['show_title'] = False
    df_params['title']      = ''

    df_params['xlabel']               = 'Cycle'
    df_params['ylabel']               = df_params.apply(lambda x: ylabels[x['plot_type']], axis='columns')
    
    # axis label for each axis
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'show_axis_label'] = False
        df_params.loc[df_params['axis_id']==axis_id, 'axis_label'] = df_params.apply(lambda x: 'Axis {}'.format(x['axis_id']), axis='columns')
    
    df_params['show_yticklabels'] = True

    df_params['show_xticklabels'] = False
    df_params.loc[df_params['row'] == df_params['row'].max(), 'show_xticklabels'] = True
    
    df_params['ytick_label_side'] = 'left'

    df_params['legend_loc'] = 'upper right'
    
    df_params['xmin'] = -1
    df_params['xmax'] = get_unique(df_params['nb_cycles'])
    
    df_params = set_df_params_ids(df_params)
    
    return df_params

def treatment_inertia_paper_figure_05_grid_params():

    df_params = [None]*2
    
    inertia_fcts = ['hard', 'constant', 'linear', 'quadratic', 'quartic' ]
    
    # row 0
    meas_shift = [0, 5, 10]
    df_params[0] = pd.concat([params_default_df]*len(meas_shift)*len(inertia_fcts))
    df_params[0]['row'] = 0
    df_params[0]['col'] = np.repeat(range(len(meas_shift)), len(inertia_fcts))
    df_params[0]['inertia_fct'] = np.tile(inertia_fcts, len(meas_shift))
    df_params[0]['meas_shift'] = np.repeat(meas_shift, len(inertia_fcts))
    df_params[0]['meas_skew'] = 0
    df_params[0]['xticks'] = np.nan
    df_params[0]['yticks'] = [list(np.arange(0, 1.1, .2))] * len(df_params[0])
    df_params[0]['ymin'] = 0
    df_params[0]['ymax'] = 1.1
    df_params[0]['title'] = [ r'${}$ mmHg'.format(e) for e in np.repeat(meas_shift, len(inertia_fcts))] # string to be shown in the legend
    
    # row 1
    meas_skew = [0, 1, 40]
    df_params[1] = pd.concat([params_default_df]*len(meas_skew)*len(inertia_fcts))
    df_params[1]['row'] = 1
    df_params[1]['col'] = np.repeat(range(len(meas_skew)), len(inertia_fcts))
    df_params[1]['inertia_fct'] = np.tile(inertia_fcts, len(meas_skew))
    df_params[1]['meas_shift'] = 0
    df_params[1]['meas_skew'] = np.repeat(meas_skew, len(inertia_fcts))
    df_params[1]['xticks'] = np.nan
    df_params[1]['yticks'] = [list(np.arange(0, 1.1, .2))] * len(df_params[1])
    df_params[1]['ymin'] = 0
    df_params[1]['ymax'] = 1.1
    df_params[1]['title'] = [ '{}'.format(e) for e in np.repeat(meas_skew, len(inertia_fcts))] # string to be shown in the legend
    
    df_params = pd.concat(df_params).reset_index(drop=True)

    df_params['axis_id'] = df_params['row'] * (df_params['col'].max()+1) + df_params['col']

    # string to be shown in the legend
    df_params['layer_name'] = df_params['inertia_fct']
    
    df_params['start_inertia_at_cycle'] = 0
    df_params['measurement_strategy']   = 'single'
    df_params['plot_type']              = 4
    df_params['iSBP']                   = 160
    df_params['nb_initial_titration']   = 0
    df_params['meas_std']               = 10
    df_params['dose']                   = 'LawStandardDose'
    df_params['drug_dist']              = 'truncnorm'
    df_params['drug_std']               = 1
    df_params['meas_dist']              = 'norm'
    df_params['nb_patients']            = 1000
    df_params['nb_iterations']          = 15
    df_params['nb_cycles']              = 10
    
    df_params['rnd_seed'] = df_params.apply(lambda x: np.random.randint(2**32, size=x['nb_iterations']).tolist(), axis='columns')
    
    df_params['layer_id'] = -1
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'layer_id'] = list(range(len(inertia_fcts)))
        
    # plot specific
    df_params['conf_level'] = .95
    
    df_params['show_title'] = True
    
    df_params['show_xlabel'] = False
    df_params.loc[df_params['row']==df_params['row'].max(), 'show_xlabel'] = True
    
    df_params['show_ylabel'] = False
    df_params.loc[df_params['col']==df_params['col'].min(), 'show_ylabel'] = True
    
    df_params['show_legend'] = False
    df_params.loc[(df_params['row']==df_params['row'].min())
                  & (df_params['col']==df_params['col'].max()) , 
                  'show_legend'] = True # top-left axis
    
    df_params['xlabel'] = 'Cycle'
    df_params['ylabel'] = df_params.apply(lambda x: ylabels[x['plot_type']], axis='columns')

    # axis label for each axis
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'show_axis_label'] = False
        df_params.loc[df_params['axis_id']==axis_id, 'axis_label'] = df_params.apply(lambda x: 'Axis {}'.format(x['axis_id']), axis='columns')
        
    df_params['show_yticklabels'] = False
    df_params.loc[df_params['col'] == df_params['col'].min(), 'show_yticklabels'] = True

    df_params['show_xticklabels'] = False
    df_params.loc[df_params['row'] == df_params['row'].max(), 'show_xticklabels'] = True
    
    # pad between subplots
    df_params['layout_w_pad'] = -3
    df_params['layout_h_pad'] = 0 #-2
    
    df_params['ylabel_font_size'] = 12
    
    df_params['legend_loc'] = 'lower right'
    
    df_params['xmin'] = -1
    df_params['xmax'] = get_unique(df_params['nb_cycles'])
    
    df_params = set_df_params_ids(df_params)
    
    return df_params

def treatment_inertia_paper_sup_materials_figure_09_grid_params():

    df_params = [None]*3
    
    meas_shift   = [0, 5, 10]
    inertia_fcts = ['hard', 'constant', 'linear', 'quadratic', 'quartic' ]
    
    # row 0
    df_params[0] = pd.concat([params_default_df]*len(meas_shift)*len(inertia_fcts))
    df_params[0]['row'] = 0
    df_params[0]['start_inertia_at_cycle'] = 0
    df_params[0]['measurement_strategy'] = 'double_new'
    df_params[0]['inertia_fct'] = np.tile(inertia_fcts, len(meas_shift))
    df_params[0]['meas_shift'] = np.repeat(meas_shift, len(inertia_fcts))
    df_params[0]['xticks'] = np.nan
    df_params[0]['yticks'] = [list(np.arange(0, 1.1, .2))] * len(df_params[0])
    df_params[0]['ymin'] = 0
    df_params[0]['ymax'] = 1.1
    
    # row 1
    df_params[1] = pd.concat([params_default_df]*len(meas_shift)*len(inertia_fcts))
    df_params[1]['row'] = 1
    df_params[1]['start_inertia_at_cycle'] = 1
    df_params[1]['measurement_strategy'] = 'single'
    df_params[1]['inertia_fct'] = np.tile(inertia_fcts, len(meas_shift))
    df_params[1]['meas_shift'] = np.repeat(meas_shift, len(inertia_fcts))
    df_params[1]['xticks'] = np.nan
    df_params[1]['yticks'] = [list(np.arange(0, 1.1, .2))] * len(df_params[1])
    df_params[1]['ymin'] = 0
    df_params[1]['ymax'] = 1.1
    
    # row 2
    df_params[2] = pd.concat([params_default_df]*len(meas_shift)*len(inertia_fcts))
    df_params[2]['row'] = 2
    df_params[2]['start_inertia_at_cycle'] = 1
    df_params[2]['measurement_strategy'] = 'double_new'
    df_params[2]['inertia_fct'] = np.tile(inertia_fcts, len(meas_shift))
    df_params[2]['meas_shift'] = np.repeat(meas_shift, len(inertia_fcts))
    df_params[2]['xticks'] = np.nan
    df_params[2]['yticks'] = [list(np.arange(0, 1.1, .2))] * len(df_params[2])
    df_params[2]['ymin'] = 0
    df_params[2]['ymax'] = 1.1
    
    df_params = pd.concat(df_params).reset_index(drop=True)

    df_params['col'] = -1
    df_params.loc[df_params['meas_shift']==meas_shift[0], 'col'] = 0
    df_params.loc[df_params['meas_shift']==meas_shift[1], 'col'] = 1
    df_params.loc[df_params['meas_shift']==meas_shift[2], 'col'] = 2

    df_params['axis_id'] = df_params['row'] * (df_params['col'].max()+1) + df_params['col']

    df_params['plot_type']              = 4
    df_params['iSBP']                   = 160
    df_params['nb_initial_titration']   = 0
    df_params['dose']                   = 'LawStandardDose'
    df_params['drug_dist']              = 'truncnorm'
    df_params['drug_std']               = 1
    df_params['meas_dist']              = 'norm'
    df_params['meas_std']               = 10
    df_params['meas_skew']              = 0

    # string to be shown in the legend
    df_params['layer_name'] = df_params['inertia_fct']
    
    df_params['nb_patients']          = 1000
    df_params['nb_iterations']        = 15
    df_params['nb_cycles']            = 10
    
    df_params['rnd_seed'] = df_params.apply(lambda x: np.random.randint(2**32, size=x['nb_iterations']).tolist(), axis='columns')
    
    df_params['layer_id'] = -1
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'layer_id'] = list(range(len(inertia_fcts)))
        
    # plot specific
    df_params['conf_level'] = .95
    
    df_params['show_title'] = False
    df_params.loc[df_params['row']==df_params['row'].min(), 'show_title'] = True
    
    df_params['show_xlabel'] = False
    df_params.loc[df_params['row']==df_params['row'].max(), 'show_xlabel'] = True
    
    df_params['show_ylabel'] = False
    df_params.loc[df_params['col']==df_params['col'].min(), 'show_ylabel'] = True
    
    df_params['show_legend'] = False
    df_params.loc[(df_params['row']==df_params['row'].min())
                  & (df_params['col']==df_params['col'].max()) , 
                  'show_legend'] = True # top-left axis
    
    df_params['title']  = df_params.apply(lambda x: '${}$ mmHg'.format(x['meas_shift']), axis='columns')
    df_params['xlabel'] = 'Cycle'
    df_params['ylabel'] = df_params.apply(lambda x: ylabels[x['plot_type']], axis='columns')

    # axis label for each axis
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'show_axis_label'] = False
        df_params.loc[df_params['axis_id']==axis_id, 'axis_label'] = df_params.apply(lambda x: 'Axis {}'.format(x['axis_id']), axis='columns')
        
    df_params['show_yticklabels'] = False
    df_params.loc[df_params['col'] == df_params['col'].min(), 'show_yticklabels'] = True

    df_params['show_xticklabels'] = False
    df_params.loc[df_params['row'] == df_params['row'].max(), 'show_xticklabels'] = True
    
    # pad between subplots
    df_params['layout_w_pad'] = -3
    df_params['layout_h_pad'] = -2
    
    df_params['ylabel_font_size'] = 12
    
    df_params['legend_loc'] = 'lower right'
    
    df_params['xmin'] = -1
    df_params['xmax'] = get_unique(df_params['nb_cycles'])
    
    df_params = set_df_params_ids(df_params)
    
    return df_params

def treatment_inertia_paper_sup_materials_figure_10_grid_params():

    df_params = [None]*3
    
    meas_skew    = [0, 1, 40]
    inertia_fcts = ['hard', 'constant', 'linear', 'quadratic', 'quartic' ]
    
    # row 0
    df_params[0] = pd.concat([params_default_df]*len(meas_skew)*len(inertia_fcts))
    df_params[0]['row'] = 0
    df_params[0]['start_inertia_at_cycle'] = 0
    df_params[0]['measurement_strategy'] = 'double_new'
    df_params[0]['inertia_fct'] = np.tile(inertia_fcts, len(meas_skew))
    df_params[0]['meas_skew'] = np.repeat(meas_skew, len(inertia_fcts))
    df_params[0]['xticks'] = np.nan
    df_params[0]['yticks'] = [list(np.arange(0, 1.1, .2))] * len(df_params[0])
    df_params[0]['ymin'] = 0
    df_params[0]['ymax'] = 1.1
    
    # row 1
    df_params[1] = pd.concat([params_default_df]*len(meas_skew)*len(inertia_fcts))
    df_params[1]['row'] = 1
    df_params[1]['start_inertia_at_cycle'] = 1
    df_params[1]['measurement_strategy'] = 'single'
    df_params[1]['inertia_fct'] = np.tile(inertia_fcts, len(meas_skew))
    df_params[1]['meas_skew'] = np.repeat(meas_skew, len(inertia_fcts))
    df_params[1]['xticks'] = np.nan
    df_params[1]['yticks'] = [list(np.arange(0, 1.1, .2))] * len(df_params[1])
    df_params[1]['ymin'] = 0
    df_params[1]['ymax'] = 1.1
    
    # row 2
    df_params[2] = pd.concat([params_default_df]*len(meas_skew)*len(inertia_fcts))
    df_params[2]['row'] = 2
    df_params[2]['start_inertia_at_cycle'] = 1
    df_params[2]['measurement_strategy'] = 'double_new'
    df_params[2]['inertia_fct'] = np.tile(inertia_fcts, len(meas_skew))
    df_params[2]['meas_skew'] = np.repeat(meas_skew, len(inertia_fcts))
    df_params[2]['xticks'] = np.nan
    df_params[2]['yticks'] = [list(np.arange(0, 1.1, .2))] * len(df_params[2])
    df_params[2]['ymin'] = 0
    df_params[2]['ymax'] = 1.1
    
    df_params = pd.concat(df_params).reset_index(drop=True)

    df_params['col'] = -1
    df_params.loc[df_params['meas_skew']==meas_skew[0], 'col'] = 0
    df_params.loc[df_params['meas_skew']==meas_skew[1], 'col'] = 1
    df_params.loc[df_params['meas_skew']==meas_skew[2], 'col'] = 2

    df_params['axis_id'] = df_params['row'] * (df_params['col'].max()+1) + df_params['col']

    df_params['plot_type']              = 4
    df_params['iSBP']                   = 160
    df_params['nb_initial_titration']   = 0
    df_params['dose']                   = 'LawStandardDose'
    df_params['drug_dist']              = 'truncnorm'
    df_params['drug_std']               = 1
    df_params['meas_dist']              = 'skewnorm'
    df_params['meas_std']               = 10
    df_params['meas_shift']             = 0

    # string to be shown in the legend
    df_params['layer_name'] = df_params['inertia_fct']
    
    df_params['nb_patients']          = 1000
    df_params['nb_iterations']        = 15
    df_params['nb_cycles']            = 10
    
    df_params['rnd_seed'] = df_params.apply(lambda x: np.random.randint(2**32, size=x['nb_iterations']).tolist(), axis='columns')
    
    df_params['layer_id'] = -1
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'layer_id'] = list(range(len(inertia_fcts)))
        
    # plot specific
    df_params['conf_level'] = .95
    
    df_params['show_title'] = False
    df_params.loc[df_params['row']==df_params['row'].min(), 'show_title'] = True
    
    df_params['show_xlabel'] = False
    df_params.loc[df_params['row']==df_params['row'].max(), 'show_xlabel'] = True
    
    df_params['show_ylabel'] = False
    df_params.loc[df_params['col']==df_params['col'].min(), 'show_ylabel'] = True
    
    df_params['show_legend'] = False
    df_params.loc[(df_params['row']==df_params['row'].min())
                  & (df_params['col']==df_params['col'].max()) , 
                  'show_legend'] = True # top-left axis
    
    df_params['title']  = df_params.apply(lambda x: '${}$'.format(x['meas_skew']), axis='columns')
    df_params['xlabel'] = 'Cycle'
    df_params['ylabel'] = df_params.apply(lambda x: ylabels[x['plot_type']], axis='columns')

    # axis label for each axis
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'show_axis_label'] = False
        df_params.loc[df_params['axis_id']==axis_id, 'axis_label'] = df_params.apply(lambda x: 'Axis {}'.format(x['axis_id']), axis='columns')
        
    df_params['show_yticklabels'] = False
    df_params.loc[df_params['col'] == df_params['col'].min(), 'show_yticklabels'] = True

    df_params['show_xticklabels'] = False
    df_params.loc[df_params['row'] == df_params['row'].max(), 'show_xticklabels'] = True
    
    # pad between subplots
    df_params['layout_w_pad'] = -3
    df_params['layout_h_pad'] = -2
    
    df_params['ylabel_font_size'] = 12
    
    df_params['legend_loc'] = 'lower right'
    
    df_params['xmin'] = -1
    df_params['xmax'] = get_unique(df_params['nb_cycles'])
    
    df_params = set_df_params_ids(df_params)
    
    return df_params

if __name__ == '__main__': 

    sns.set()
    matplotlib.style.use('ggplot')

    output_path = './results/'
    create_dir(output_path)

    hook_functions = [partial(first_hook_function, query='all', do_mean=True, cumulative=True),                  
                      partial(second_hook_function, query='(tSBP < 140) & (mSBP < 140)', do_ratio=True), #.format(params['treatment_threshold'])
                      partial(second_hook_function, query='(tSBP < 120) & (mSBP < 140)', do_ratio=True),  #.format(params['treatment_threshold'])
                      partial(third_hook_function, query='all'), 
                      partial(second_hook_function, query='(tSBP < 140)', do_ratio=True), 
                      partial(second_hook_function, query='(tSBP < 120)', do_ratio=True), 
                      partial(second_hook_function, query='(mSBP < 140)', do_ratio=True),
                      
                      partial(forth_hook_function, query='mSBP >= 140', decisive_only=False, do_mean=True, cumulative=True),
                      partial(forth_hook_function, query='mSBP >= 140', decisive_only=True, do_mean=True, cumulative=True),
                      
                      partial(forth_hook_function, query='mSBP >= 140', decisive_only=False, do_mean=False, cumulative=True),
                      partial(forth_hook_function, query='mSBP >= 140', decisive_only=True, do_mean=False, cumulative=True),
                      
                      partial(forth_hook_function, query='all', decisive_only=False, do_mean=True, cumulative=True),
                      partial(forth_hook_function, query='all', decisive_only=True, do_mean=True, cumulative=True),
                      
                      partial(forth_hook_function, query='all', decisive_only=False, do_mean=False, cumulative=True),
                      partial(forth_hook_function, query='all', decisive_only=True, do_mean=False, cumulative=True),
                      
                      partial(third_hook_function, query='(tSBP < 140)'), 
                      partial(third_hook_function, query='(tSBP >= 140)')                  
                     ]

    ylabels = ['Mean Medications', # Avg. Nb. Treament Steps
               'Proportion tSBP<140 mmHg | \nmSBP<140 mmHg',
               'Proportion tSBP<120 mmHg | \nmSBP<140 mmHg',
               'Mean tSBP (mmHg)',
               'Proportion tSBP<140 mmHg',
               'Proportion tSBP<120 mmHg',
               'Proportion mSBP<140 mmHg',
               
               'Mean mSBP >= 140 mmHg (All)', 
               'Mean mSBP >= 140 mmHg (Decisive)', 
               
               'Nb. mSBP >= 140 mmHg (All)', 
               'Nb. mSBP >= 140 mmHg (Decisive)', 
               
               'Mean Measurements (All)',
               'Mean Measurements (Decisive)',
               
               'Nb. Measurements (All)', 
               'Nb. Measurements (Decisive)', 

               'Mean tSBP | tSBP<140 mmHg',
               'Mean tSBP | tSBP>=140 mmHg'           
              ]

    # build the dataframe of parameters

    # manusript
    # df_params = treatment_inertia_paper_figure_02_grid_params() # figure_02
    # df_params = treatment_inertia_paper_figure_03_grid_params(160) # figure_03
    # df_params = treatment_inertia_paper_figure_03_extension_grid_params(160) # figure_03_ext
    # df_params = treatment_inertia_paper_figure_04_grid_params() # figure_04
    # df_params = treatment_inertia_paper_figure_04_extension_grid_params() # figure_04_ext
    # df_params = treatment_inertia_paper_figure_05_grid_params() # UNSED figure_06 white coat comparison

    # supplementary materials
    # df_params = treatment_inertia_paper_sup_materials_figure_03_grid_params() # sup_figure_03
    # df_params = treatment_inertia_paper_sup_materials_figure_04_grid_params() # sup_figure_04
    # df_params = treatment_inertia_paper_figure_03_grid_params(150) # sup_figure_05a
    # df_params = treatment_inertia_paper_figure_03_grid_params(170) # sup_figure_05b
    # df_params = treatment_inertia_paper_sup_materials_figure_06_grid_params('a')  # sup_figure_06
    # df_params = treatment_inertia_paper_sup_materials_figure_06_grid_params('b')  # sup_figure_07
    # df_params = treatment_inertia_paper_sup_materials_figure_06_grid_params('c')  # sup_figure_08
    # df_params = treatment_inertia_paper_sup_materials_figure_07_grid_params() # sup_figure_09 white_coat_shift
    # df_params = treatment_inertia_paper_sup_materials_figure_08_grid_params() # sup_figure_10 white_coat_skew
    # df_params = treatment_inertia_paper_sup_materials_figure_09_grid_params() # white_coat_shift comparison
    df_params = treatment_inertia_paper_sup_materials_figure_10_grid_params() # white_coat_skewn comparison

    assert('id' in df_params.columns) # To remove eventually

    # build the dataframe used for plotting

    df_params['started_at'] = now('%d %b %Y %H:%M:%S')
    data, df_plot = build_grid_df_plot(df_params)
    df_params['ended_at'] = now('%d %b %Y %H:%M:%S')

    # build text labels for each row

    text = build_fixed_params_text(df_params, 
                                   use_latex=True, 
                                   pname_to_latex_df=pname_to_latex_df)

    # do the plotting

    fig = facet_grid_plot(df_plot, 
                          df_params, 
                          text)

    save_fig  = True
    if save_fig: 
        
        path = output_path
        
        filename = 'figure'
        file_format = 'svg'
        figure_fname = '{}.{}'.format(filename, file_format)
        
        fig.savefig(os.path.join(path, figure_fname), 
                    format=file_format, 
                    bbox_inches='tight')

    # build description dataframe

    df_desc = build_facetgrid_description_(df_plot, df_params)

    save_desc = True
    if save_desc:
        
        path = output_path
        
        filename = 'desc'
        file_format = 'csv'
        table_fname = '{}.{}'.format(filename, file_format)

        if file_format == 'html':
            df_desc.to_html(os.path.join(path, table_fname), na_rep='', index=False)
        elif file_format == 'csv':
            df_desc.to_csv(os.path.join(path, table_fname), na_rep='', index=False)
