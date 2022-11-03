#!/usr/bin/env python
# coding: utf-8

"""
    Plots for a Single Cycle
"""

import math
import random
import pickle
import copy
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

from libs.plot import text_ax

# helper functions

def first_hook_function(data, 
                        params, 
                        query, 
                        max_nb_time_steps,
                        do_ratio,
                        cumulative,
                        shifted):
    """
        ratio of dropped patients per treatment step given a query
    """
    
    # remove unwanted patients
    dropped = query_all(data['dropped'], query)
    
    # nb of dropped patients per time step
    # NOTE nunique() also take care of cases with multiple measurements at a single time step (i.e. params['nb_measurements'] > 1) 
    raw_count = dropped.groupby(['iteration_id', 'cycle_step', 'time_step'])['patient_id'].nunique()
    raw_count = raw_count.rename('value')
    
    if not shifted:
        # HACK shift time_step in cycle_step 0 by `params['nb_initial_titration']`
        raw_count = raw_count.reset_index()
        raw_count.loc[raw_count['cycle_step']==0, 'time_step'] -= params['nb_initial_titration']
        raw_count = raw_count.set_index(['iteration_id', 'cycle_step', 'time_step']).sort_index()
        raw_count = raw_count['value'] # to series
    
    # fill missing time steps
    idx = pd.MultiIndex.from_product([range(0, params['nb_iterations']),
                                      range(0, params['nb_cycles']),
                                      range(0, max_nb_time_steps)], 
                                     names=['iteration_id', 'cycle_step', 'time_step'])
    df_plot = pd.Series(data=0, 
                        index=idx, 
                        name='value')
    df_plot.update(raw_count)
    
    if shifted:
        # hide initial titration steps on the plots
        df_plot.loc[(df_plot.index.get_level_values('cycle_step') == 0) &                     (df_plot.index.get_level_values('time_step') < params['nb_initial_titration'])] = np.nan

    if cumulative:
        # cumulative nb of dropped patients per treatment step
        df_plot = df_plot.groupby(level=['iteration_id', 'cycle_step']).cumsum().rename('value')
    
    if do_ratio:
        # ratio of dropped patients per treatment step
        df_plot = df_plot.divide(params['nb_patients'])
    
    df_plot = df_plot.to_frame().reset_index()
    
    return df_plot    

def build_single_df_plot(data, params):
    """
        generate `df_plot` from `params`
        combine results from `hook_function` from multiple iterations into a single dataframe (long format)
    """
    
    hook_function = hook_functions[params['plot_type']]
    df_plot = hook_function(data, params)
    
    df_plot['axis_id']         = params['axis_id']
    df_plot['layer_id']        = params['layer_id']

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
        
        df_plot.append(df_plot_)
        data.append(data_)

    data = concat_data_(data)
    df_plot = pd.concat(df_plot, axis='index')
    
    return data, df_plot

def build_axis(ax, 
               axis_id, 
               df_plot, 
               df_params, 
               plot_style='lineplot_with_error'):
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
    
    # grouped boxplot
    if plot_style == 'boxplot':
        ax = sns.boxplot(ax=ax,
                         x='time_step', 
                         y='value', 
                         data=df_plot.reset_index(), 
                         hue='layer_id', 
                         showfliers=True,
                         palette=palette)
        ax = sns.pointplot(ax=ax,
                           x='time_step', 
                           y='value', 
                           hue='layer_id', 
                           palette=palette, 
                           legend=legend_str,
                           data=df_plot.reset_index().groupby('time_step', as_index=False).mean())

    elif plot_style == 'lineplot':
        # TODO
        assert(False)

    # Lineplot with Error Bars
    elif plot_style == 'lineplot_with_error': 
        ax = sns.pointplot(ax=ax,
                           x='time_step', 
                           y='value', 
                           ci=conf_level*100, # confidence level (default is 95). If None, no error bars will not be drawn.
                           hue='layer_id', 
                           capsize=.2, 
                           errwidth=1.5, # thickness of error bar lines and caps (default = 3?)
                           legend=legend_str,
                           palette=palette, 
                           data=df_plot)
        plt.setp(ax.collections, sizes=[25]) # fix marker size (https://github.com/mwaskom/seaborn/issues/899) (default=50?)
        
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
        legend = ax.legend(handles=handles, 
                  labels=labels, 
                  title='',
                  loc=legend_loc)

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
        ax.set_xlim(left=xmin)
        
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
                         columns='time_step', 
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
    df_desc = df_desc.sort_values(['layer_id', 'time_step'])

    return df_desc

def build_facetgrid_description(df_plot):
    
    df_desc = []
    for axis_id, df in df_plot.groupby('axis_id'):
        
        df_desc_ = build_overlayed_description(df)

        # add a level to the index
        df_desc_['axis_id'] = axis_id

        df_desc.append(df_desc_)

    df_desc = pd.concat(df_desc, axis='index')
    
    df_desc = df_desc.sort_values(['axis_id', 'layer_id', 'time_step'])
    
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

        'time_step',

         'conf_level', 
         'conf_interval',

        # params

        'axis_id', 
        'row',
        'col',

        'nb_iterations',
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
        'start_inertia_at_cycle',
        
        'meas_shift', 
        'meas_skew'

    ]]

    df_desc['start_inertia_at_cycle'] += 1
    
    if False:
        df_desc_pretty = df_desc[['xlabel', 'ylabel', 'time_step', 'layer_name', 'conf_interval']].set_index(['xlabel', 'ylabel', 'time_step', 'layer_name']).unstack('layer_name')
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
               'white_coat_effect_tsbp_delta']
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
                    text, 
                    plot_style):
    
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
            df_params_ = df_params[df_params['axis_id'] == axis_id]
            
            # plot the graph
            ax = build_axis(ax, 
                            axis_id, 
                            df_, 
                            df_params_, 
                            plot_style=plot_style)
            
            # get the axis dimensions
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            width = xlim[1] - xlim[0]
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
            ax = text_ax(ax, text[row_idx], loc=[0, .4])
            
    return fig


# Parameters

def build_single_plot_params():
    
    params = copy.deepcopy(params_default)
    
    params['id']             = 0
    params['iSBP']                 = 150
    params['nb_patients']          = 50
    params['nb_cycles']            = 1
    params['nb_iterations']        = 5
    params['rnd_seed']             = np.random.randint(2**32, size=params['nb_iterations']).tolist()

    # treatment
    params['inertia_fct']          = 'linear'
    params['nb_initial_titration'] = 0
    params['drug_dist']            = 'truncnorm'
    params['dose']                 = 'LawStandardDose'
    # params['drug_eff']             = 5
    params['drug_std']             = 1

    # measurement
    params['meas_dist']            = 'norm'
    params['measurement_strategy'] = 'double_new'
    params['meas_std']             = 10

    # plot specific
    params['plot_type']            = 0
    params['conf_level']           = .997
    params['layer_name']           = 'layer_name' # string to be shown in the legend
    params['axis_id']              = 0
    params['layer_id']             = 0
    params['hue']                  = params['layer_id']
    params['show_title']           = True
    params['show_legend']          = False
    params['show_xlabel']          = True
    params['show_ylabel']          = True
    params['title']                = '${}$ mmHg'.format(params['iSBP'])
    params['xlabel']               = 'Treatment Step'
    params['ylabel']               = ylabels[params['plot_type']]

    params['row']                  = 0
    params['col']                  = 0
    
    params['xticks'] = np.nan
    params['yticks'] = list(np.arange(0, 1.1, .2))
    params['ymin']   = -.05
    params['ymax']   = 1.1
    
    assert(params['nb_cycles'] == 1)

    df_params = params_to_df(params)

    return df_params

def build_overlayed_plot_params():
    
    df_params = pd.concat([params_default_df]*3).reset_index(drop=True)
    
    df_params['iSBP']                 = 150
    df_params['nb_patients']          = 50
    df_params['nb_cycles']            = 1
    df_params['nb_iterations']        = 5
    df_params['rnd_seed']             = df_params.apply(lambda x: np.random.randint(2**32, size=x['nb_iterations']).tolist(), axis='columns')

    # treatment
    df_params['inertia_fct']          = 'constant'
    df_params['nb_initial_titration'] = [0, 1, 2]
    # df_params['nb_initial_titration'] = 0
    df_params['drug_dist']            = 'truncnorm'
    df_params['dose']                 = 'LawStandardDose'
    # df_params['drug_eff']             = [5, 10, 15]
    df_params['drug_std']             = 1

    # measurement
    df_params['meas_dist']            = 'norm'
    df_params['measurement_strategy'] = 'double_new'
    df_params['meas_std']             = 10

    # plot specific
    df_params['plot_type']            = 2
    df_params['conf_level']           = .997
    df_params['axis_id']              = 0
    df_params['layer_id']             = list(range(3))
    df_params['layer_name']           = ['mono therapy', 'dual therapy', 'triple therapy'] # string to be shown in the legend
    df_params['hue']                  = df_params['layer_id']
    df_params['show_title']           = True
    df_params['show_legend']          = True
    df_params['show_xlabel']          = True
    df_params['show_ylabel']          = True
    # df_params['title']                = '${}$ mmHg'.format(params['iSBP'])
    df_params['title']                = df_params.apply(lambda x: '${}$ mmHg'.format(x['iSBP']), axis='columns')
    df_params['xlabel']               = 'Treatment Step'
    df_params['ylabel']               = df_params.apply(lambda x: ylabels[x['plot_type']], axis='columns')

    df_params['row']                  = 0
    df_params['col']                  = 0
    
    df_params = set_df_params_ids(df_params)
    
    assert(get_unique(df_params['nb_cycles']) == 1)
    
    return df_params

def default_facet_grid_params(nb_patients, 
                              nb_iterations, 
                              pname_to_latex,
                              plot_type, 
                              inertia_fct):
    df_params = [None]*3

    # row 0
    df_params[0] = pd.concat([params_default_df]*9)
    df_params[0]['iSBP'],     df_params[0]['drug_std'],     df_params[0]['meas_std'],     df_params[0]['drug_eff'] =         list(zip(*itertools.product([150, 160, 170], [1, 5, 10], [10], [10])))
    df_params[0]['row'] = 0
    df_params[0]['layer_name'] =         df_params[0].apply(lambda x: '${} = {}$'.format(pname_to_latex['drug_std'], x['drug_std']), axis='columns') # string to be shown in the legend
    
    # row 1
    df_params[1] = pd.concat([params_default_df]*9)
    df_params[1]['iSBP'],     df_params[1]['drug_std'],     df_params[1]['meas_std'],     df_params[1]['drug_eff'] =         list(zip(*itertools.product([150, 160, 170], [5], [5, 10, 15], [10])))
    df_params[1]['row'] = 1
    df_params[1]['layer_name'] =         df_params[1].apply(lambda x: '${} = {}$'.format(pname_to_latex['meas_std'], x['meas_std']), axis='columns') # string to be shown in the legend
    
    # row 2
    df_params[2] = pd.concat([params_default_df]*9)
    df_params[2]['iSBP'],     df_params[2]['drug_std'],     df_params[2]['meas_std'],     df_params[2]['drug_eff'] =         list(zip(*itertools.product([150, 160, 170], [5], [10], [5, 10, 15])))
    df_params[2]['row'] = 2
    df_params[2]['layer_name'] =         df_params[2].apply(lambda x: '${} = {}$'.format(pname_to_latex['drug_eff'], x['drug_eff']), axis='columns') # string to be shown in the legend

    df_params = pd.concat(df_params).reset_index(drop=True)

    df_params['col'] = -1
    df_params.loc[df_params['iSBP']==150, 'col'] = 0
    df_params.loc[df_params['iSBP']==160, 'col'] = 1
    df_params.loc[df_params['iSBP']==170, 'col'] = 2

    df_params['axis_id'] = df_params['row']*(df_params['col'].max()+1) + df_params['col']

    df_params['dose'] = 'FixedDose'
    df_params['measurement_strategy'] = 'single'
       
    df_params['nb_patients']   = nb_patients
    df_params['nb_iterations'] = nb_iterations
    df_params['inertia_fct']   = inertia_fct
    
    df_params['rnd_seed']      = df_params.apply(lambda x: np.random.randint(2**32, size=x['nb_iterations']).tolist(), axis='columns')
    
    df_params['layer_id'] = -1
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'layer_id'] = list(range(3))
        
    df_params['plot_type'] = plot_type
    
    df_params['xlabel']               = 'Treatment Step'
    df_params['ylabel']               = df_params.apply(lambda x: ylabels[x['plot_type']], axis='columns')
    
    df_params = set_df_params_ids(df_params)
    
    return df_params

def law_facet_grid_params(nb_patients, 
                          nb_iterations, 
                          pname_to_latex, 
                          dose, 
                          plot_type, 
                          inertia_fct):
    df_params = [None]*2

    # row 0
    df_params[0] = pd.concat([params_default_df]*9)
    df_params[0]['iSBP'],     df_params[0]['drug_std'],     df_params[0]['meas_std'] =         list(zip(*itertools.product([150, 160, 170], [1, 5, 10], [10])))
    df_params[0]['row'] = 0
    df_params[0]['layer_name'] =         df_params[0].apply(lambda x: '${} = {}$'.format(pname_to_latex['drug_std'], x['drug_std']), axis='columns') # string to be shown in the legend
    
    # row 1
    df_params[1] = pd.concat([params_default_df]*9)
    df_params[1]['iSBP'],     df_params[1]['drug_std'],     df_params[1]['meas_std'] =         list(zip(*itertools.product([150, 160, 170], [5], [5, 10, 15])))
    df_params[1]['row'] = 1
    df_params[1]['layer_name'] =         df_params[1].apply(lambda x: '${} = {}$'.format(pname_to_latex['meas_std'], x['meas_std']), axis='columns') # string to be shown in the legend
    
    df_params = pd.concat(df_params).reset_index(drop=True)

    df_params['col'] = -1
    df_params.loc[df_params['iSBP']==150, 'col'] = 0
    df_params.loc[df_params['iSBP']==160, 'col'] = 1
    df_params.loc[df_params['iSBP']==170, 'col'] = 2

    df_params['axis_id'] = df_params['row']*(df_params['col'].max()+1) + df_params['col']

    df_params['dose'] = dose
    
    df_params['nb_patients']   = nb_patients
    df_params['nb_iterations'] = nb_iterations
    df_params['inertia_fct']   = inertia_fct
    
    df_params['rnd_seed'] = df_params.apply(lambda x: np.random.randint(2**32, size=x['nb_iterations']).tolist(), axis='columns')
    
    df_params['layer_id'] = -1
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'layer_id'] = list(range(3))
        
    df_params['plot_type'] = plot_type
    
    df_params['xlabel']               = 'Treatment Step'
    df_params['ylabel']               = df_params.apply(lambda x: ylabels[x['plot_type']], axis='columns')
    
    df_params = set_df_params_ids(df_params)
    
    return df_params

def law_facet_grid_params2(nb_patients, 
                           nb_iterations, 
                           pname_to_latex, 
                           doses, 
                           plot_type,
                           inertia_fct):
    df_params = [None]*2

    # row 0
    df_params[0] = pd.concat([params_default_df]*9)
    df_params[0]['iSBP'],     df_params[0]['dose'] = list(zip(*itertools.product([150, 160, 170], doses)))
    df_params[0]['drug_std'] = 1
    df_params[0]['meas_std'] = 5
    df_params[0]['row'] = 0
    df_params[0]['layer_name'] =         df_params[0]['dose']
    
    # row 1
    df_params[1] = pd.concat([params_default_df]*9)
    df_params[1]['iSBP'],     df_params[1]['dose'] = list(zip(*itertools.product([150, 160, 170], doses)))
    df_params[1]['drug_std'] = 1
    df_params[1]['meas_std'] = 10
    df_params[1]['row'] = 1
    df_params[1]['layer_name'] =         df_params[1]['dose']
    
    df_params = pd.concat(df_params).reset_index(drop=True)

    df_params['col'] = -1
    df_params.loc[df_params['iSBP']==150, 'col'] = 0
    df_params.loc[df_params['iSBP']==160, 'col'] = 1
    df_params.loc[df_params['iSBP']==170, 'col'] = 2

    df_params['axis_id'] = df_params['row']*(df_params['col'].max()+1) + df_params['col']
    
    df_params['nb_patients']   = nb_patients
    df_params['nb_iterations'] = nb_iterations
    df_params['plot_type']     = plot_type
    df_params['measurement_strategy'] = 'single'
    df_params['inertia_fct']   = inertia_fct
    
    df_params['rnd_seed'] = df_params.apply(lambda x: np.random.randint(2**32, size=x['nb_iterations']).tolist(), axis='columns')
    
    df_params['layer_id'] = -1
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'layer_id'] = list(range(3))
        
    df_params['xlabel']               = 'Treatment Step'
    df_params['ylabel']               = df_params.apply(lambda x: ylabels[x['plot_type']], axis='columns')
    
    df_params = set_df_params_ids(df_params)
    
    return df_params

def multiple_titration_grid_params(nb_patients, 
                                   nb_iterations, 
                                   dose, 
                                   plot_type,
                                   inertia_fct):
    df_params = [None]*3

    # row 0
    df_params[0] = pd.concat([params_default_df]*9)
    df_params[0]['row'] = 0
    df_params[0]['meas_std'] = 5
    df_params[0]['iSBP'] = np.repeat([150, 160, 170], 3)
    df_params[0]['nb_initial_titration'] = np.tile([0, 1, 2], 3)
    
    # row 1
    df_params[1] = pd.concat([params_default_df]*9)
    df_params[1]['row'] = 1
    df_params[1]['meas_std'] = 10
    df_params[1]['iSBP'] = np.repeat([150, 160, 170], 3)
    df_params[1]['nb_initial_titration'] = np.tile([0, 1, 2], 3)
    
    # row 2
    df_params[2] = pd.concat([params_default_df]*9)
    df_params[2]['row'] = 2
    df_params[2]['meas_std'] = 15
    df_params[2]['iSBP'] = np.repeat([150, 160, 170], 3)
    df_params[2]['nb_initial_titration'] = np.tile([0, 1, 2], 3)
    
    df_params = pd.concat(df_params).reset_index(drop=True)
    
    df_params['col'] = -1
    df_params.loc[df_params['iSBP']==150, 'col'] = 0
    df_params.loc[df_params['iSBP']==160, 'col'] = 1
    df_params.loc[df_params['iSBP']==170, 'col'] = 2

    df_params['axis_id'] = df_params['row']*(df_params['col'].max()+1) + df_params['col']

    df_params['dose'] = dose
    df_params['drug_eff'] = 10
    df_params['drug_std'] = 1 #5

    # string to be shown in the legend
    df_params.loc[df_params['nb_initial_titration'].apply(lambda x: x==0), 'layer_name'] = 'mono therapy'
    df_params.loc[df_params['nb_initial_titration'].apply(lambda x: x==1), 'layer_name'] = 'dual therapy'
    df_params.loc[df_params['nb_initial_titration'].apply(lambda x: x==2), 'layer_name'] = 'triple therapy'
    
    df_params['nb_patients']   = nb_patients
    df_params['nb_iterations'] = nb_iterations
    df_params['plot_type']     = plot_type
    df_params['measurement_strategy'] = 'single'
    df_params['inertia_fct']   = inertia_fct

    df_params['rnd_seed'] = df_params.apply(lambda x: np.random.randint(2**32, size=x['nb_iterations']).tolist(), axis='columns')
    
    df_params['layer_id'] = -1
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'layer_id'] = list(range(3))
        
    df_params['xlabel']               = 'Treatment Step'
    df_params['ylabel']               = df_params.apply(lambda x: ylabels[x['plot_type']], axis='columns')
    
    df_params = set_df_params_ids(df_params)
    
    return df_params

def multiple_titration_grid_params2(nb_patients, 
                                    nb_iterations, 
                                    doses,
                                    plot_type,
                                    inertia_fct):
    df_params = [None]*3

    # row 0
    df_params[0] = pd.concat([params_default_df]*9)
    df_params[0]['row'] = 0
    df_params[0]['iSBP'] = np.repeat([150, 160, 170], 3)
    df_params[0]['nb_initial_titration'] = np.tile([0, 1, 2], 3)
    df_params[0]['dose'] = doses[0]
    
    # row 1
    df_params[1] = pd.concat([params_default_df]*9)
    df_params[1]['row'] = 1
    df_params[1]['iSBP'] = np.repeat([150, 160, 170], 3)
    df_params[1]['nb_initial_titration'] = np.tile([0, 1, 2], 3)
    df_params[1]['dose'] = doses[1]
    
    # row 2
    df_params[2] = pd.concat([params_default_df]*9)
    df_params[2]['row'] = 2
    df_params[2]['iSBP'] = np.repeat([150, 160, 170], 3)
    df_params[2]['nb_initial_titration'] = np.tile([0, 1, 2], 3)
    df_params[2]['dose'] = doses[2]
    
    df_params = pd.concat(df_params).reset_index(drop=True)
    
    df_params['col'] = -1
    df_params.loc[df_params['iSBP']==150, 'col'] = 0
    df_params.loc[df_params['iSBP']==160, 'col'] = 1
    df_params.loc[df_params['iSBP']==170, 'col'] = 2

    df_params['axis_id'] = df_params['row']*(df_params['col'].max()+1) + df_params['col']

    df_params['drug_eff']    = 10
    df_params['drug_std']    = 1
    df_params['meas_std']    = 15
    df_params['measurement_strategy'] = 'single'
    df_params['inertia_fct'] = inertia_fct

    # string to be shown in the legend
    df_params.loc[df_params['nb_initial_titration'].apply(lambda x: x==0), 'layer_name'] = 'mono therapy'
    df_params.loc[df_params['nb_initial_titration'].apply(lambda x: x==1), 'layer_name'] = 'dual therapy'
    df_params.loc[df_params['nb_initial_titration'].apply(lambda x: x==2), 'layer_name'] = 'triple therapy'
    
    df_params['nb_patients']   = nb_patients
    df_params['nb_iterations'] = nb_iterations
    df_params['plot_type']     = plot_type
    
    df_params['rnd_seed'] = df_params.apply(lambda x: np.random.randint(2**32, size=x['nb_iterations']).tolist(), axis='columns')
    
    df_params['layer_id'] = -1
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'layer_id'] = list(range(3))
        
    df_params['xlabel']               = 'Treatment Step'
    df_params['ylabel']               = df_params.apply(lambda x: ylabels[x['plot_type']], axis='columns')
    
    df_params = set_df_params_ids(df_params)
    
    return df_params

def treatment_inertia_paper_figure_01_grid_params():
    
    df_params = [None]*2
    
    inertia_fcts = ['hard', 'constant', 'linear', 'quadratic', 'quartic' ]
    measurement_strategy = 'single'

    # row 0, col 0
    df_params[0] = pd.concat([params_default_df]*len(inertia_fcts))
    df_params[0]['col'] = 0
    df_params[0]['plot_type'] = 0
    df_params[0]['inertia_fct'] = inertia_fcts
    df_params[0]['xticks'] = np.nan
    df_params[0]['yticks'] = [list(np.arange(0, 1.1, .2))] * len(df_params[0])
    df_params[0]['ymin'] = -.05
    df_params[0]['ymax'] = 1.1
    
    # row 0, col 1
    df_params[1] = pd.concat([params_default_df]*len(inertia_fcts))
    df_params[1]['col'] = 1
    df_params[1]['plot_type'] = 3
    df_params[1]['inertia_fct'] = inertia_fcts
    df_params[1]['xticks'] = np.nan
    df_params[1]['yticks'] = [list(np.arange(0, 1.1, .2))] * len(df_params[0])
    df_params[1]['ymin'] = -.05
    df_params[1]['ymax'] = 1.1
    
    df_params = pd.concat(df_params).reset_index(drop=True)

    df_params['iSBP']                   = 160
    df_params['start_inertia_at_cycle'] = 0
    df_params['nb_initial_titration']   = 0
    df_params['dose']                   = 'LawStandardDose'
    df_params['drug_dist']              = 'truncnorm'
    df_params['drug_std']               = 1
    df_params['meas_dist']              = 'norm'
    df_params['meas_std']               = 10
    df_params['measurement_strategy']   = measurement_strategy
    df_params['nb_patients']            = 100
    df_params['nb_iterations']          = 15

    # string to be shown in the legend
    df_params['layer_name'] = df_params['inertia_fct']
    
    df_params['rnd_seed'] = df_params.apply(lambda x: np.random.randint(2**32, size=x['nb_iterations']).tolist(), axis='columns')
    
    df_params['row']      = 0
    df_params['axis_id']  = df_params['row']*(df_params['col'].max()+1) + df_params['col']
    df_params['layer_id'] = -1
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'layer_id'] = list(range(len(inertia_fcts)))
    
    # plot parameters
    df_params['conf_level']  = .95
    
    df_params['show_title']  = False
    df_params['title']       = ''
    df_params['show_xlabel'] = True
    df_params['show_ylabel'] = True
    
    df_params['show_legend'] = False
    df_params.loc[df_params['col']==df_params['col'].max(), 'show_legend'] = True

    # axis label for each axis
    for axis_id in df_params['axis_id'].unique():
        df_params.loc[df_params['axis_id']==axis_id, 'show_axis_label'] = False
        df_params.loc[df_params['axis_id']==axis_id, 'axis_label'] = df_params.apply(lambda x: 'Axis {}'.format(x['axis_id']), axis='columns')
        
    df_params['xlabel']               = 'Treatment Step'
    df_params['ylabel']               = df_params.apply(lambda x: ylabels[x['plot_type']], axis='columns')
    
    df_params['legend_loc'] = 'upper left'
    
    df_params = set_df_params_ids(df_params)
    
    return df_params

if __name__ == '__main__':

    sns.set()
    matplotlib.style.use('ggplot')

    max_nb_time_steps = 10
    shifted = False

    hook_functions = [partial(first_hook_function, query='(mSBP < 140)',                max_nb_time_steps=max_nb_time_steps, do_ratio=True, cumulative=True, shifted=shifted), 
                      partial(first_hook_function, query='(tSBP < 140) & (mSBP < 140)', max_nb_time_steps=max_nb_time_steps, do_ratio=True, cumulative=True, shifted=shifted),
                      partial(first_hook_function, query='(tSBP < 120) & (mSBP < 140)', max_nb_time_steps=max_nb_time_steps, do_ratio=True, cumulative=True, shifted=shifted), 
                      partial(first_hook_function, query='(tSBP < 140)',                max_nb_time_steps=max_nb_time_steps, do_ratio=True, cumulative=True, shifted=shifted)] 

    ylabels = ['Proportion mSBP<140 mmHg', 
               'Proportion tSBP<140 mmHg | \nmSBP<140 mmHg',
               'Proportion tSBP<120 mmHg | \nmSBP<140 mmHg', 
               'Proportion tSBP<140 mmHg']

    # build the dataframe of parameters

    df_params = treatment_inertia_paper_figure_01_grid_params()

    assert(get_unique(df_params['nb_cycles']) == 1)
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
                    text, 
                    plot_style='lineplot_with_error')




    output_path = './results/'
    create_dir(output_path)

    save_fig  = True
    if save_fig:  

        path = output_path

        filename = 'single_cycle'
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

        filename = 'single_cycle'
        file_format = 'csv'
        table_fname = '{}.{}'.format(filename, file_format)

        if file_format == 'html':
            df_desc.to_html(os.path.join(path, table_fname), na_rep='', index=False)
        elif file_format == 'csv':
            df_desc.to_csv(os.path.join(path, table_fname), na_rep='', index=False)
