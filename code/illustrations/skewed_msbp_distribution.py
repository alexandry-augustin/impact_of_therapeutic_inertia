#!/usr/bin/env python
# coding: utf-8

"""
    Skewed mSBP Distribution
"""

import numpy as np
import os
import itertools

import sys
# appending the root path of the `libs` directory in the sys.path list
sys.path.append(os.path.join('..'))

import scipy.stats as stats
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import seaborn as sns

from libs.core import create_dir

def build_data(params):
    
    f = lambda a: stats.skewnorm(a=a,
                       loc=params['loc'], 
                       scale=params['meas_std'])

    data = []
    stats_data = []
    for layer_id, a in enumerate(params['skew']):

        data_ = pd.DataFrame()
        data_['mSBP'] = np.arange(110, 190, .2) #, dtype=int)
        data_['layer_id'] = layer_id
        data_['name'] = r'$\alpha = {}$'.format(a)
        data_['value'] =  data_['mSBP'].apply(lambda x: f(a).pdf(x)) 

        data.append(data_)
        
        stats_ = pd.DataFrame()
        stats_['layer_id'] = [layer_id]
        
        stats_['mean']     = [f(a).mean()]
        stats_['median']   = [f(a).median()]
        stats_['mode']     = data_['mSBP'].iloc[f(a).pdf(data_['mSBP']).argmax()]
        
        stats_['value_at_mean']   = stats_['mean'].apply(lambda x: f(a).pdf(x))
        stats_['value_at_median'] = stats_['median'].apply(lambda x: f(a).pdf(x))
        stats_['value_at_mode']   = stats_['mode'].apply(lambda x: f(a).pdf(x))
        
        stats_data.append(stats_)

    data = pd.concat(data)
    stats_data = pd.concat(stats_data)
    
    return data, stats_data

def build_skewed_msbp_axis(ax, 
                    data, 
                    stats_data, 
                    stat, 
                    params,
                    
                    show_xlabel,
                    xlabel,
                    show_xticklabels,
                    xticks,
                    
                    show_ylabel,
                    ylabel,
                    show_yticklabels,
                    yticks,
                    
                    xmin, xmax,
                    ymin, ymax,
                    
                    show_title,
                    title,
                    palette):
    
    ax = sns.lineplot(ax=ax,
                  x='mSBP', 
                  y='value',
                  hue='layer_id',
                  palette=palette, 
                  legend='full',
                  lw=2,
                  data=data)

    ax.set_xlim(left=xmin, right=xmax)
    ax.set_ylim(bottom=ymin, top=ymax)

    if show_xlabel:
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel('')
    if show_ylabel:
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel('')
        
    if show_title:
        ax.set_title(title)
        
    if not show_xticklabels:
        ax.get_xaxis().set_ticklabels([])
    if not show_yticklabels:
        ax.get_yaxis().set_ticklabels([])
        
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.fill_between([xmin, 140],
                    [ymax, ymax],
                    [ymin, ymin],
                    facecolor="none", 
                    hatch="XXX", 
                    edgecolor="g", 
                    linewidth=1,
                    alpha=.3, 
                    label='no drug')
    ax.fill_between([140, 160],
                    [ymax, ymax],
                    [ymin, ymin],
                    facecolor="none", 
                    hatch="XXX",
                    edgecolor="k", 
                    linewidth=1,
                    alpha=.3,
                    label='inertia')

    lw=3
    ax.vlines(params['loc'], 
              ymin, ymax, 
              colors='#1a5f8a', 
              lw=lw, 
              linestyles='--', 
              label='tSBP')
    
    # vertical lines (mean, median, mode)
    palette_iter = itertools.cycle(palette)
    for _, row in stats_data.iterrows():
        plt.plot(np.repeat(row[stat], 2), 
                 [0, row['value_at_{}'.format(stat)]], 
                 color=next(palette_iter), 
                 linestyle="-.",
                 label=stat)
    
    # remove the legend title
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles=handles[1:], labels=labels[1:], title='')

    # remap `layer_id` to `layer_name` in the legend
    to_name = data[['layer_id', 'name']].drop_duplicates().set_index('layer_id')['name']
    for text_obj in legend.get_texts():
        layer_id_str = text_obj.get_text()
        
        try:
            layer_id = int(layer_id_str)
            
            if layer_id in to_name.index: 
                name = to_name[layer_id]
                text_obj.set_text(name)
                
        except ValueError:
            pass
    
    return ax

if __name__ == '__main__':

    sns.set()
    matplotlib.style.use('ggplot')

    flatui = {
        'hard':'#192d48', 
        'constant':'#2b6f39', 
        'linear':'#a1794a', 
        'quadratic':'#d490c6',
        'quartic':'#c3d9f3',
    }
    palette = sns.color_palette(flatui.values())

    np.random.seed(0)

    output_path = '../results/'
    create_dir(output_path)

    params = {}
    params['loc'] = 160
    params['drug_eff'] = 10
    params['skew'] = [0, 1, 40] #[0, .75, 20] # [0, .5, 1, 15]
    params['meas_std'] = 10

    data, stats_data = build_data(params)

    fig, ax = plt.subplots(figsize=(10, 5))

    palette = sns.color_palette("Reds_d", n_colors=len(params['skew']))

    xticks = [120, 140, 160]
    xticks.append(params['loc'])
    ax = build_skewed_msbp_axis(ax,
                                data, 
                                stats_data, 
                                'mean',
                                params,
                            
                                show_xlabel=True,
                                xlabel='mmHg',
                                show_xticklabels=True,
                                xticks=xticks, 
                            
                                show_ylabel=True,
                                ylabel='Probability',
                                show_yticklabels=True,
                                yticks=np.arange(0, .1, .01),
                            
                                xmin=data['mSBP'].min(), 
                                xmax=data['mSBP'].max(),
                                ymin=-.001, ymax=.1,
                            
                                show_title=True,
                                title='Measured SBP',
                                palette=palette)
    
    save_fig = True
    if save_fig: 

        path = output_path
        
        file_format = 'svg'
        figure_fname = '{}.{}'.format('skewed_mSBP_mean', file_format)
        
        fig.savefig(os.path.join(path, figure_fname), 
                    format=file_format, 
                    bbox_inches='tight')

    s = stats_data[['layer_id', 'mean', 'median', 'mode']]

    save_stats = True
    if save_stats:
        
        path = output_path
        
        file_format = 'csv'
        table_fname = '{}.{}'.format('test', file_format)
        
        if file_format == 'html':
            s.to_html(os.path.join(path, table_fname), index=False)
        elif file_format == 'csv':
            s.to_csv(os.path.join(path, table_fname), index=False)
