#!/usr/bin/env python
# coding: utf-8

"""
    Normal
"""

import numpy as np
import os

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

def build_tsbp_df(params):

    # absolute clipping
    a = (0 - params['loc']) / params['drug_std']
    b = (params['iSBP'] - params['loc']) / params['drug_std']


    f = stats.truncnorm(loc=params['loc'], 
                         scale=params['drug_std'], 
                             a=a, 
                             b=b).pdf
    # truncnorm_pdf = loc + sigma*stats.truncnorm(loc=0, scale=1, a=a, b=b).pdf(x)

    df_plot = pd.DataFrame()
    df_plot['tSBP'] = np.linspace(params['iSBP']-50, params['iSBP']+50, 200)
    df_plot['value'] = df_plot['tSBP'].apply(lambda x: f(x))
    
    return df_plot




def build_msbp_df(params):

    f = stats.norm(loc=params['loc'], 
                   scale=params['meas_std']).pdf

    df_plot = pd.DataFrame()
    df_plot['mSBP'] = np.linspace(params['iSBP']-50, params['iSBP']+50, 200)
    df_plot['value'] = df_plot['mSBP'].apply(lambda x: f(x))
    
    return df_plot

def build_tsbp_axis(ax, 
                    df_plot, 
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
                    title):

    ax = sns.lineplot(ax=ax,
                  x='tSBP', 
                  y='value',
                  legend='full',
                  lw=2,
                  data=df_plot)

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

    ax.fill_between([120, 140],
                    [ymax, ymax],
                    [ymin, ymin],
                    facecolor="none", 
                    hatch="XXX", 
                    edgecolor="b", 
                    linewidth=1,
                    alpha=.5, 
                    label='controlled')

    ax.vlines(params['iSBP'], 
              ymin, ymax, 
              colors='k', 
              lw=2, 
              linestyles=':', 
              label='iSBP')
    ax.vlines(params['iSBP'] - params['drug_eff'], 
              ymin, ymax, 
              colors='k', 
              lw=1, 
              linestyles=':', 
              label='_nolegend_')

    # arrow
    head_length = 2
    ax.arrow(params['iSBP'], .02, 
             -10+head_length, 0, 
             width=.0002,
             head_width=.002, 
             head_length=head_length, 
             fc='k', 
             ec='k'
            )
    ax.text(params['iSBP']-15, 0.017, 'drug_eff', dict(size=12))  
    
    ax.legend(loc='upper right')
    
    return ax

def build_msbp_axis(ax, 
                    df_plot, 
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
                    title):

    ax = sns.lineplot(ax=ax,
                  x='mSBP', 
                  y='value',
                  legend='full',
                  lw=2,
                  data=df_plot)

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
                    alpha=.5, 
                    label='no drug')
    ax.fill_between([140, 160],
                    [ymax, ymax],
                    [ymin, ymin],
                    facecolor="none", 
                    hatch="XXX",
                    edgecolor="k", 
                    linewidth=1,
                    alpha=.5,
                    label='inertia')

    lw=2
    ax.vlines(params['iSBP'] - params['drug_eff'], 
              ymin, ymax, 
              colors='k', 
              lw=lw, 
              linestyles=':', 
              label='tSBP')
    
    ax.legend(loc='upper right')
    
    return ax

def build_text(params):

    text = []
    text.append('Parameters: ')
    for k, v in params.items():
        text.append('    {}={}'.format(k, str(v)))
    text = '\n'.join(text)
    
    return text

def text_ax(ax,
            text, 
            loc,
            fontsize):
    """
        Build 'Fixed parameter' text axis
        loc: bottom left coordinate of the text area bounding box
    """
    
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    ax.set_facecolor('white')

    ax.text(loc[0],
            loc[1], 
            text, 
            horizontalalignment='left', 
            verticalalignment='bottom',
            fontsize=fontsize)
    
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

    params = {
        'iSBP':160,
        'drug_eff':10,
        'drug_std':15,
        'drug_dist':'truncnorm',
        
        'meas_std':15,
        'meas_dist':'norm',
        
        'threshold':140
    }
    params['loc'] = params['iSBP'] - params['drug_eff']

    nb_rows = 1
    nb_cols = 2
    total_nb_cols = nb_cols # + 1 # including the 'Fixed parameter' axis
    figsize=(total_nb_cols*15/4., nb_rows*10/3.)
    fig, axes = plt.subplots(nb_rows, 
                            total_nb_cols, 
                            figsize=figsize)

    plt.tight_layout(w_pad=-2)

    df_plot = build_msbp_df(params)
    axes[0] = build_msbp_axis(axes[0],
                                df_plot, 
                                params,
                            
                                show_xlabel=True,
                                xlabel='mmHg',
                                show_xticklabels=True,
                                xticks=[120, 140, 160], 
                            
                                show_ylabel=True,
                                ylabel='Probability',
                                show_yticklabels=True,
                                yticks=np.arange(0, 0.045, 0.01),
                            
                                xmin=110, xmax=190,
                                ymin=-.001, ymax=.04,
                            
                                show_title=True,
                                title='Measured SBP')

    df_plot = build_tsbp_df(params)
    axes[1] = build_tsbp_axis(axes[1], 
                                df_plot, 
                                params,
                                show_xlabel=True,
                                xlabel='mmHg',
                                show_xticklabels=True,
                                xticks=[120, 140, 160], 
                            
                                show_ylabel=False,
                                ylabel='Probability',
                                show_yticklabels=False,
                                yticks=np.arange(0, 0.045, 0.01),
                            
                                xmin=110, xmax=190,
                                ymin=-.001, ymax=.04,
                            
                                show_title=True,
                                title='True SBP')


    # Labels
    axes[0].text(180, 0.002, 'A', dict(size=30))
    axes[1].text(180, 0.002, 'B', dict(size=30))

    save_fig = True
    if save_fig: 
        
        path = output_path
        
        file_format = 'svg'
        figure_fname = '{}.{}'.format('sup_figure_01', file_format)
        
        fig.savefig(os.path.join(path, figure_fname), 
                    format=file_format, 
                    bbox_inches='tight')