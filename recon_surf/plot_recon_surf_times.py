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

def get_nonunique_cmd_execution_times(yaml_dicts, split_recon_all_stages=True):
    cmd_names = []
    cmd_times = []
    sides_list = []
    subject_ids = []

    for yaml_dict in yaml_dicts:
        for stage_num in range(len(yaml_dict['recon-surf_commands'])):
            entry_list = list(yaml_dict['recon-surf_commands'][stage_num].values())[0]

            for cmd_entry in entry_list:
                side = None
                if any(lh_str in cmd_entry['cmd'] for lh_str in ['lh', '255 ']):
                    side = 'lh'
                elif any(rh_str in cmd_entry['cmd'] for rh_str in ['rh', '127 ']):
                    side = 'rh'
                else:
                    side = 'full'

                if cmd_entry['cmd'].split(' ')[0] == 'recon-all':
                    if not split_recon_all_stages:
                        cmd_names.append(cmd_entry['cmd'].split(' ')[0])
                        if 'duration_m' in cmd_entry.keys():
                            cmd_times.append(cmd_entry['duration_m'])
                        elif 'duration_s' in cmd_entry.keys():
                            cmd_times.append(cmd_entry['duration_s'])

                        sides_list.append(side)
                        subject_ids.append(yaml_dict['sid'])

                    else:
                        for stage_dict in cmd_entry['stages']:
                            cmd_names.append('recon-all:'+stage_dict['stage_name'])
                            if 'duration_m' in stage_dict.keys():
                                cmd_times.append(stage_dict['duration_m'])
                            elif 'duration_s' in stage_dict.keys():
                                cmd_times.append(stage_dict['duration_s'])
                            sides_list.append(side)
                            subject_ids.append(yaml_dict['sid'])

                else:
                    ## If python3 script, get script name:
                    if 'python3' in cmd_entry['cmd'].split(' ')[0]:
                        cmd_names.append(cmd_entry['cmd'].split(' ')[1].split('/')[-1])
                    else:
                        cmd_names.append(cmd_entry['cmd'].split(' ')[0])

                    if 'duration_m' in cmd_entry.keys():
                        cmd_times.append(cmd_entry['duration_m'])
                    elif 'duration_s' in cmd_entry.keys():
                        cmd_times.append(cmd_entry['duration_s'])

                    sides_list.append(side)
                    subject_ids.append(yaml_dict['sid'])

    return cmd_names, cmd_times, sides_list, subject_ids

def separate_hemis(filtered_df, sides_list):
    cols = filtered_df.columns.values.tolist()
    cols.append('Side')
    rows_list = []

    for index, row in filtered_df.iterrows():
        cmd_name = row['cmd_names']
        cmd_time = row['cmd_times']
        subject_id = row['subject_id']
        side = sides_list[index]

        if 'recon-all' in cmd_name:
            if 'lh' in cmd_name:
                cmd_name = cmd_name.replace('lh', '')
                if cmd_name[-1] == ' ':
                    cmd_name = cmd_name[:-1]
            if 'rh' in cmd_name:
                cmd_name = cmd_name.replace('rh', '')
                if cmd_name[-1] == ' ':
                    cmd_name = cmd_name[:-1]

        rows_list.append([cmd_name, cmd_time, subject_id, side])

    two_sided_filtered_df = pd.DataFrame(rows_list, columns=cols)

    return two_sided_filtered_df

def plot_bar(df):
    sns.barplot(x='cmd_names', y='cmd_times', data=df,
                order=df.groupby('cmd_names').mean()['cmd_times'].sort_values().index,
                hue='Side', ci='sd', capsize=.1,
                palette={'lh': tab10_color_palette_[1],
                         'full': tab10_color_palette_[0],
                         'rh': tab10_color_palette_[2]},
                hue_order=['lh', 'full', 'rh'])

def plot_box(df):
    sns.boxplot(x='cmd_names', y='cmd_times', data=df,
                order=df.groupby('cmd_names').mean()['cmd_times'].sort_values().index,
                hue='Side',
                palette={'lh': tab10_color_palette_[1],
                         'full': tab10_color_palette_[0],
                         'rh': tab10_color_palette_[2]},
                hue_order=['lh', 'full', 'rh'])

def get_yaml_data(root_dir, subject_dirs):
    yaml_dicts = []
    valid_dirs = []

    print('[INFO] Extracting data from the files:')
    for subject_dir in subject_dirs:
        if not os.path.isdir(os.path.join(root_dir, subject_dir)):
            continue
        file_path = os.path.join(root_dir, subject_dir, 'scripts/recon-surf_times.yaml')
        print('  - {}'.format(file_path))
        try:
            with open(file_path, 'r') as stream:
                try:
                    yaml_dicts.append(yaml.safe_load(stream))
                except yaml.YAMLError as e:
                    print(e)
            valid_dirs.append(subject_dir)

        except FileNotFoundError as e:
            print(e)
            print('[INFO] Skipping this file...')

    if len(yaml_dicts) == 0:
        print('[ERROR] No data could be read for processing! Exiting')
        sys.exit()

    return yaml_dicts, valid_dirs

def get_top_x_cmds(plotting_df, x):
    means_df = plotting_df.groupby(['cmd_names', 'Side'], as_index=False).mean()
    means_df = means_df.loc[means_df['cmd_times'] == means_df['cmd_times']]   ## Remove NaN entries produced by groupby

    ordered_means_df = means_df.loc[reversed(means_df['cmd_times'].sort_values().index)]
    top_unique_cmds = []
    counter = 0

    for index, row in ordered_means_df.iterrows():
        cmd_name = row['cmd_names']
        if cmd_name not in top_unique_cmds:
            top_unique_cmds.append(cmd_name)
            counter += 1

        if counter == x:
            break

    print('[INFO] Plotting only top {} commands:'.format(x))
    print(' - ' + '\n - '.join(top_unique_cmds))

    excluded_cmds = [cmd_name for cmd_name in np.unique(means_df.cmd_names.values).tolist() if cmd_name not in top_unique_cmds]

    for cmd_name in excluded_cmds:
        plotting_df = plotting_df.drop(plotting_df[plotting_df.cmd_names == cmd_name].index)

    return plotting_df

# def get_selected_cmds(plotting_df, selected_cmds):


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

    ## Extract recon-surf time information:
    print('[INFO] Extracting command execution times...')
    cmd_names, cmd_times, sides_list, subject_ids = get_nonunique_cmd_execution_times(yaml_dicts, split_recon_all_stages=True)

    df = pd.DataFrame({'cmd_names': cmd_names, 'cmd_times': cmd_times, 'subject_id': subject_ids})

    filtered_df = df.copy()

    filtered_df = separate_hemis(filtered_df, sides_list)

    ## Apply filters:
    if args.top_x is not None:
        filtered_df = get_top_x_cmds(filtered_df, args.top_x)

    if args.select_cmds is not None:
        excluded_cmds = [cmd_name for cmd_name in filtered_df.cmd_names.values if cmd_name not in args.select_cmds]

        for excluded_cmd in excluded_cmds:
            filtered_df = filtered_df.drop(filtered_df[filtered_df.cmd_names == excluded_cmd].index)

        print('[INFO] Plotting only the desired commands:')
        print(' - ' + '\n - '.join(args.select_cmds))

    if args.time_threshold is not None:
        filtered_df = filtered_df.drop(filtered_df[filtered_df.cmd_times < args.time_threshold].index)

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

