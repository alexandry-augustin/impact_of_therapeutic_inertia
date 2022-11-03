#!/usr/bin/env python
# coding: utf-8

"""
    Bernouilli and Inertia Functions
"""

import numpy as np
import copy
import os

import sys
# appending the root path of the `libs` directory in the sys.path list
sys.path.append(os.path.join('..'))

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import seaborn as sns

from libs.core import create_dir
from libs.core import params_default
from libs.core import inertia

def build_bernouilli_axis(ax, 
                     y,
                     xticklabels=['Treat', 'Don\'t Treat'], 
                     ylim=(-.05, 1.05),
                     show_yticklabels=True,
                     ylabel='Probability',
                     xfontsize=15,
                     yfontsize=15):
    """
    """
    x = np.arange(2)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=xfontsize) # ['Treat', 'Don't Treat']
    ax.bar(x, y, width=1.0, align='center', color="#AAAAAA", ecolor='#000000')
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, fontsize=yfontsize)

    #frame customisation
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.tick_params(axis='x',
            which='both',
            bottom='off',
            top='off',
            labelbottom='on',
            direction='out'
        )
    ax.tick_params(axis='y',
            which='both',
            left='on',
            right='off',
            labelleft='on',
            direction='out'
        )
    
    ax.tick_params(axis=u'both', which=u'both',length=0) # hide tick lines
    
    if not show_yticklabels:
        ax.get_yaxis().set_ticklabels([])

    return ax

def build_inertia_df():
    
    params = copy.deepcopy(params_default)

    vinertia = np.vectorize(inertia, otypes=[np.float64])

    df_ = []
    inertia_fcts = ['hard', 'constant', 'linear', 'quadratic', 'quartic' ]
    for layer_id, name in enumerate(inertia_fcts):
        df = pd.DataFrame()
        
        df['mSBP'] = np.linspace(135, 165, 100)
        df['layer_id'] = layer_id
        
        params['inertia_fct'] = name
        df['name'] = name
        df['value'] = vinertia(df['mSBP'], params, 10)

        df_.append(df)

    df_ = pd.concat(df_)
    
    return df_

def build_inertia_axis(ax,
                 show_title=True,
                 title='f(mSBP)',
                 show_ylabel=True,
                 ylabel='Probability of Treatment', # 'value of \'p\''
                 ylim=(-.05, 1.05),
                 show_yticklabels=True,
                 xfontsize=15,
                 yfontsize=15,
                 title_fontsize=20):
    """
    """
    
    df = build_inertia_df()
    
    ax = sns.lineplot(ax=ax,
                  x='mSBP', 
                  y='value',
                  hue='layer_id',
                  palette=palette, 
                  legend='full',
                  data=df)

    plt.setp(ax.lines, linewidth=3)

    if show_title:
        ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel('mSBP', fontsize=xfontsize) 
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=yfontsize)
    else:
        ax.set_ylabel('')
    ax.set_ylim(ylim)
    
    if not show_yticklabels:
        ax.get_yaxis().set_ticklabels([])
        
    # remove the legend title
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles=handles[1:], labels=labels[1:], title='')

    # remap `layer_id` to `layer_name` in the legend
    to_name = df[['layer_id', 'name']].drop_duplicates().set_index('layer_id')['name']
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

def build_inertia_axis2(ax,
                 show_title=True,
                 title='f(mSBP)',
                 show_ylabel=True,
                 ylabel='Probability of Treatment', # 'value of \'p\''
                 ylim=(-.05, 1.05),
                 show_yticklabels=True):
    """
    """
    
    df = build_inertia_df()
    
    palette='cubehelix'

    ax = sns.pointplot(ax=ax,
                  x='mSBP', 
                  y='value',
                  hue='layer_id',
                  palette=palette, 
                  legend='full',
                  scale=0, # size of the markers
                  data=df)

    plt.setp(ax.lines, linewidth=3)

    if show_title:
        ax.set_title(title, fontsize=20)
    ax.set_xlabel('mSBP', fontsize=15) 
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=15)
    else:
        ax.set_ylabel('')
    ax.set_ylim(ylim)
    
    if not show_yticklabels:
        ax.get_yaxis().set_ticklabels([])
        
    # remove the legend title
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles=handles, labels=labels, title='')

    # remap `layer_id` to `layer_name` in the legend
    to_name = df[['layer_id', 'name']].drop_duplicates().set_index('layer_id')['name']
    for text_obj in legend.get_texts():
        layer_id_str = text_obj.get_text()
        
        try:
            layer_id = int(layer_id_str)
            
            if layer_id in to_name.index: 
                name = to_name[layer_id]
                text_obj.set_text(name)
                
        except ValueError:
            pass
        
    step = 5
    xticks = np.arange(df['mSBP'].min(), df['mSBP'].max()+step, step)
    ax.set_xticks(xticks)
            
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

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ylim = (-.05, 1.05)

    axes[0] = build_bernouilli_axis(axes[0], 
                            np.array([.65, .35]),
                            ylim=ylim, 
                            xticklabels=['Treatment', 'No Treatment'], )
    axes[1] = build_inertia_axis(axes[1], 
                            show_title=False,
                            show_ylabel=True, 
                            ylim=ylim,
                            show_yticklabels=False)

    # Labels
    axes[0].text(-.5, 0.85, 'A', dict(size=15))
    axes[1].text(135, 0.85, 'B', dict(size=15))

    save_fig = True
    if save_fig: 
        
        path = output_path

        file_format = 'svg'
        figure_fname = '{}.{}'.format('figure_00', file_format)
        
        fig.savefig(os.path.join(path, figure_fname), 
                    format=file_format, 
                    bbox_inches='tight')