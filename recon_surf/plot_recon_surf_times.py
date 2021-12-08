#!/usr/bin/env python3
import os
import sys
import argparse

import yaml
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


tab10_color_palette_ = sns.color_palette('tab10', 10)

def get_nonunique_cmd_execution_times(yaml_dicts, split_recon_all_stages=False, return_recon_all_info=True):
    cmd_names = []
    cmd_times = []
    if return_recon_all_info:
        recon_all_stage_names = []
        recon_all_stage_times = []

    for yaml_dict in yaml_dicts:
        for stage_num in range(len(yaml_dict['recon-surf_commands'])):
            entry_list = list(yaml_dict['recon-surf_commands'][stage_num].values())[0]

            for cmd_entry in entry_list:
                if cmd_entry['cmd'].split(' ')[0] == 'recon-all':
                    if not split_recon_all_stages:
                        cmd_names.append(cmd_entry['cmd'].split(' ')[0])
                        if 'duration_m' in cmd_entry.keys():
                            cmd_times.append(cmd_entry['duration_m'])
                        elif 'duration_s' in cmd_entry.keys():
                            cmd_times.append(cmd_entry['duration_s'])

                    for stage_dict in cmd_entry['stages']:
                        if split_recon_all_stages:
                            cmd_names.append('recon-all:'+stage_dict['stage_name'])
                            if 'duration_m' in stage_dict.keys():
                                cmd_times.append(stage_dict['duration_m'])
                            elif 'duration_s' in stage_dict.keys():
                                cmd_times.append(stage_dict['duration_s'])

                        if return_recon_all_info:
                            recon_all_stage_names.append(stage_dict['stage_name'])
                            if 'duration_m' in stage_dict.keys():
                                recon_all_stage_times.append(stage_dict['duration_m'])
                            elif 'duration_s' in stage_dict.keys():
                                recon_all_stage_times.append(stage_dict['duration_s'])

                else:
                    ## If python3.8, get script name:
                    if cmd_entry['cmd'].split(' ')[0] == 'python3.8':
                        cmd_names.append(cmd_entry['cmd'].split(' ')[1].split('/')[-1])
                    else:
                        cmd_names.append(cmd_entry['cmd'].split(' ')[0])
                    if 'duration_m' in cmd_entry.keys():
                        cmd_times.append(cmd_entry['duration_m'])
                    elif 'duration_s' in cmd_entry.keys():
                        cmd_times.append(cmd_entry['duration_s'])

    return cmd_names, cmd_times, recon_all_stage_names, recon_all_stage_times

def separate_hemis(filtered_df):
    cols = filtered_df.columns.values.tolist()
    cols.append('Side')
    rows_list = []

    for index, row in filtered_df.iterrows():
        side = 'full'
        cmd_name = row[cols[0]]
        cmd_time = row[cols[1]]

        if 'lh' in cmd_name:
            side = 'lh'
            cmd_name = cmd_name.replace('lh', '')
        if 'rh' in cmd_name:
            side = 'rh'
            cmd_name = cmd_name.replace('rh', '')

        rows_list.append([cmd_name, cmd_time, side])

    two_sided_filtered_df = pd.DataFrame(rows_list, columns=cols)

    return two_sided_filtered_df

def plot_bar(df, separate_hemis=True):
    if not separate_hemis:
        sns.barplot(x='cmd_names', y='cmd_times', data=df,
                    order=df.groupby('cmd_names').mean()['cmd_times'].sort_values().index,
                    ci='sd', capsize=.2)
    else:
        sns.barplot(x='cmd_names', y='cmd_times', data=df,
                    order=df.groupby('cmd_names').mean()['cmd_times'].sort_values().index,
                    hue='Side', ci='sd', capsize=.1,
                    palette={'lh': tab10_color_palette_[1],
                             'full': tab10_color_palette_[0],
                             'rh': tab10_color_palette_[2]},
                    hue_order=['lh', 'full', 'rh'])


def plot_box(df, separate_hemis=True):
    if not separate_hemis:
        sns.boxplot(x='cmd_names', y='cmd_times', data=df,
                    order=df.groupby('cmd_names').mean()['cmd_times'].sort_values().index)
    else:
        sns.boxplot(x='cmd_names', y='cmd_times', data=df,
                    order=df.groupby('cmd_names').mean()['cmd_times'].sort_values().index,
                    hue='Side',
                    palette={'lh': tab10_color_palette_[1],
                             'full': tab10_color_palette_[0],
                             'rh': tab10_color_palette_[2]},
                    hue_order=['lh', 'full', 'rh'])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--root_dir', type=str,
                        default='.', help='Root directory containing subject directories')
    parser.add_argument('-s','--subject_dirs', nargs='+',
                        help='Directories of subjects to plot for', required=True)
    parser.add_argument('-p','--plot_type', type=str, default='bar',
                        help='One of [\'bar\',\'box\']')
    parser.add_argument('--top_cmds', type=int, default=None,
                        help='If given, only the cmds with the x highest execution times are plotted')
    parser.add_argument('--select_cmds', nargs='+', default=None,
                        help='If given, only the listed cmds are plotted')
    parser.add_argument('-t', '--time_threshold', type=float, default=None,
                        help='If given, only the cmds whose execution times exceed t are plotted')
    parser.add_argument('--fig_save_dir', type=str,
                        default='/tmp', help='Directory in which plot images are to be saved')
    parser.add_argument('--save_fig', dest='save_fig', action='store_true')
    parser.add_argument('--plot_recon_all_stages', dest='plot_recon_all_stages', action='store_true')
    parser.add_argument('--separate_hemis', dest='separate_hemis', action='store_true')
    parser.set_defaults(save_fig=False, plot_recon_all_stages=False, separate_hemis=False)

    args = parser.parse_args()

    yaml_dicts = []

    print('[INFO] Processing data from the files:')
    for subject_dir in args.subject_dirs:
        file_path = os.path.join(args.root_dir, subject_dir, 'scripts/recon_surf_times.log'),
        print('  - {}'.format(file_path[0]))
        try:
            with open(file_path[0], 'r') as stream:
                try:
                    yaml_dicts.append(yaml.safe_load(stream))
                except yaml.YAMLError as e:
                    print(e)

        except FileNotFoundError as e:
            print(e)
            print('[INFO] Skipping this file...')

    if len(yaml_dicts) == 0:
        print('[ERROR] No data could be read for processing! Exiting')
        sys.exit()

    ## Extract recon-surf time information:
    print('[INFO] Extracting command execution times...')
    cmd_names, cmd_times, recon_all_stage_names, recon_all_stage_times = get_nonunique_cmd_execution_times(yaml_dicts, 
                                                                                                           True, True)

    df = pd.DataFrame({'cmd_names': cmd_names, 'cmd_times': cmd_times})
    recon_all_df = pd.DataFrame({'cmd_names': recon_all_stage_names, 'cmd_times': recon_all_stage_times})
    filtered_df = df.copy()

    ## Apply filters:
    if args.top_cmds is not None:
        increasing_avgs_df = df.groupby('cmd_names').mean()['cmd_times'].sort_values()
        top_avgs_df = increasing_avgs_df[-args.top_cmds:]
        bottom_avgs_df = increasing_avgs_df[:-args.top_cmds]

        print('[INFO] Plotting only top {} commands:'.format(args.top_cmds))
        print(' - ' + '\n - '.join(top_avgs_df.keys()))

        for cmd_name in bottom_avgs_df.keys():
            filtered_df = filtered_df.drop(filtered_df[filtered_df.cmd_names == cmd_name].index)

    if args.select_cmds is not None:
        excluded_cmds = [cmd_name for cmd_name in filtered_df.cmd_names.values if cmd_name not in args.select_cmds]

        for excluded_cmd in excluded_cmds:
            filtered_df = filtered_df.drop(filtered_df[filtered_df.cmd_names == excluded_cmd].index)

        print('[INFO] Plotting only the desired commands:')
        print(' - ' + '\n - '.join(args.select_cmds))

    if args.time_threshold is not None:
        filtered_df = filtered_df.drop(filtered_df[filtered_df.cmd_times < args.time_threshold].index)

        print('[INFO] Plotting only commands whose durations exceed {} minutes'.format(args.time_threshold))

    if args.separate_hemis:
        print('[INFO] Separating commands according to hemisphere')
        filtered_df = separate_hemis(filtered_df)
        recon_all_df = separate_hemis(recon_all_df)

    ## Plot:
    print('[INFO] Plotting results')
    plt.figure(figsize=(12, 8))
    if args.plot_type == 'bar':
        plot_bar(filtered_df, args.separate_hemis)
    elif args.plot_type == 'box':
        plot_box(filtered_df, args.separate_hemis)
    else:
        print('[WARN] Invalid plot type: {}. Defaulting to bar plot...'.format(args.plot_type))
        plot_bar(filtered_df, args.separate_hemis)

    plt.xticks(rotation='80', fontsize=None)
    plt.title('recon-surf Command Execution Times (Average over {} runs)'.format(len(yaml_dicts)), fontsize=15, pad=15)
    plt.ylabel('Time (minutes)', fontsize=None)
    plt.subplots_adjust(bottom=0.5)

    plt.grid()
    fig = plt.gcf()
    if args.save_fig:
        fig.savefig(os.path.join(args.fig_save_dir, 'recon-surf_times_plot.png'))

    if args.plot_recon_all_stages:
        print('[INFO] Plotting recon-all stage results on a separate plot')
        plt.figure(figsize=(12, 8))
        if args.plot_type == 'bar':
            plot_bar(recon_all_df, args.separate_hemis)
        elif args.plot_type == 'box':
            plot_box(recon_all_df, args.separate_hemis)
        else:
            print('[WARN] Invalid plot type: {}. Defaulting to bar plot...'.format(args.plot_type))
            plot_bar(recon_all_df, args.separate_hemis)

        plt.xticks(rotation='80', fontsize=None)
        plt.title('recon-surf Command Execution Times (Average over {} runs)'.format(len(yaml_dicts)), fontsize=15, pad=15)
        plt.ylabel('Time (minutes)', fontsize=None)
        plt.subplots_adjust(bottom=0.5)

        plt.grid()
        fig = plt.gcf()
        if args.save_fig:
            fig.savefig(os.path.join(args.fig_save_dir, 'recon-surf_times_recon_all_stage_plot.png'))

    plt.show()

