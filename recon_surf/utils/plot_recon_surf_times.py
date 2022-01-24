#!/usr/bin/env python3
import os
import sys
import argparse

import yaml
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

##TODO: Set relative import once integrated in package
from plotting_utils import extract_cmd_runtime_data, separate_hemis, get_yaml_data, get_top_x_cmds, get_selected_cmds, get_runtimes_exceeding_threshold

tab10_color_palette_ = sns.color_palette('tab10', 10)


def plot_bar(df):
    sns.barplot(x='cmd_name', y='execution_time', data=df,
                order=df.groupby('cmd_name').mean()['execution_time'].sort_values().index,
                hue='hemi', ci='sd', capsize=.1,
                palette={'lh': tab10_color_palette_[1],
                         'both': tab10_color_palette_[0],
                         'rh': tab10_color_palette_[2]},
                hue_order=['lh', 'both', 'rh'])

def plot_box(df):
    sns.boxplot(x='cmd_name', y='execution_time', data=df,
                order=df.groupby('cmd_name').mean()['execution_time'].sort_values().index,
                hue='hemi',
                palette={'lh': tab10_color_palette_[1],
                         'both': tab10_color_palette_[0],
                         'rh': tab10_color_palette_[2]},
                hue_order=['lh', 'both', 'rh'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--root_dir', type=str,
                        default='.', help='Root directory containing subject directories')
    parser.add_argument('-s','--subject_dirs', nargs='+',
                        help='Directories of subjects to plot for', default=[])
    parser.add_argument('-p','--plot_type', type=str, default='bar',
                        help='One of [\'bar\',\'box\']')
    parser.add_argument('--top_x', type=int, default=None,
                        help='If given, only the cmds with the x highest execution times are plotted')
    parser.add_argument('--select_cmds', nargs='+', default=None,
                        help='If given, only the listed cmds are plotted')
    parser.add_argument('-t', '--time_threshold', type=float, default=None,
                        help='If given, only the cmds whose execution times exceed t are plotted')
    parser.add_argument('--fig_save_dir', type=str,
                        default='/tmp', help='Directory in which plot images are to be saved')
    parser.add_argument('--save_fig', dest='save_fig', action='store_true')
    parser.set_defaults(save_fig=False)

    args = parser.parse_args()

    if not args.subject_dirs:
        print('[INFO] Subject list not specified. Including all data in root_dir...')
        subject_dirs = os.listdir(args.root_dir)
    else:
        subject_dirs = args.subject_dirs

    yaml_dicts, subject_dirs = get_yaml_data(args.root_dir, subject_dirs)
    if len(yaml_dicts) == 0:
        print('[ERROR] No data could be read for processing! Exiting')
        sys.exit()

    ## Extract recon-surf time information:
    print('[INFO] Extracting command execution times...')
    cmd_names, execution_times, hemis_list, subject_ids = extract_cmd_runtime_data(yaml_dicts, split_recon_all_stages=True)

    df = pd.DataFrame({'cmd_name': cmd_names, 'execution_time': execution_times, 'subject_id': subject_ids})

    filtered_df = df.copy()
    filtered_df = separate_hemis(filtered_df, hemis_list)

    ## Apply filters:
    if args.top_x is not None:
        filtered_df, top_unique_cmds = get_top_x_cmds(filtered_df, args.top_x)
        print('[INFO] Plotting only top {} commands:'.format(args.top_x))
        print(' - ' + '\n - '.join(top_unique_cmds))

    if args.select_cmds is not None:
        filtered_df = get_selected_cmds(filtered_df, args.select_cmds)
        print('[INFO] Plotting only the desired commands:')
        print(' - ' + '\n - '.join(args.select_cmds))

    if args.time_threshold is not None:
        filtered_df = get_runtimes_exceeding_threshold(filtered_df, args.time_threshold)
        print('[INFO] Plotting only commands whose durations exceed {} minutes'.format(args.time_threshold))

    ## Plot:
    print('[INFO] Plotting results')
    plt.figure(figsize=(12, 8))
    if args.plot_type == 'bar':
        plot_bar(filtered_df)
    elif args.plot_type == 'box':
        plot_box(filtered_df)
    else:
        print('[WARN] Invalid plot type: {}. Defaulting to bar plot...'.format(args.plot_type))
        plot_bar(filtered_df)

    plt.xticks(rotation='80', fontsize=None)
    plt.title('recon-surf Command Execution Times (Average over {} runs)'.format(len(yaml_dicts)), fontsize=15, pad=15)
    plt.ylabel('Time (minutes)', fontsize=None)
    plt.subplots_adjust(bottom=0.5)

    plt.grid()
    fig = plt.gcf()
    if args.save_fig:
        fig.savefig(os.path.join(args.fig_save_dir, 'recon-surf_times_plot.png'))

    plt.show()

