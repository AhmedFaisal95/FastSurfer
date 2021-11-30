#!/usr/bin/env python3
import os
import sys
import argparse

import yaml
import seaborn as sns
import matplotlib.pyplot as plt


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--root_dir', type=str, default='.',
                        help='Root directory containing subject directories')
    parser.add_argument('-s','--subject_dirs', nargs='+',
                        help='Directories of subjects to plot for', required=True)
    parser.add_argument('-p','--plot_type', type=str, default='bar',
                        help='One of [\'bar\',\'box\']')
    parser.add_argument('--save_fig', dest='save_fig', action='store_true')
    parser.add_argument('--plot_recon_all_stages', dest='plot_recon_all_stages', action='store_true')
    parser.set_defaults(save_fig=False, plot_recon_all_stages=False)

    args = parser.parse_args()

    yaml_dicts = []

    print('[INFO] Processing data from the files:')
    for subject_dir in args.subject_dirs:
        file_path = os.path.join(args.root_dir, subject_dir, 'scripts/recon_surf_times.log'),
        print('  - {}'.format(file_path[0]))
        try:
            with open(os.path.join(subject_dir, 'scripts/recon_surf_times.log'), 'r') as stream:
                try:
                    yaml_dicts.append(yaml.safe_load(stream))
                except yaml.YAMLError as e:
                    print(e)

        except FileNotFoundError as e:
            print(e)
            print('[INFO] Skipping this file...')

    # print(len(yaml_dicts))
    # for yaml_dict in yaml_dicts:
    #     print(len(yaml_dict['recon-surf_commands']))

    if len(yaml_dicts) == 0:
        print('[ERROR] No data could be read for processing! Exiting')
        sys.exit()

    print('[INFO] Extracting command execution times...')
    cmd_names, cmd_times, recon_all_stage_names, recon_all_stage_times = get_nonunique_cmd_execution_times(yaml_dicts, 
                                                                                                           False, True)

    # sns.set_style('darkgrid')

    print('[INFO] Plotting results')
    plt.figure(figsize=(18, 8))
    if args.plot_type == 'bar':
        sns.barplot(x=cmd_names, y=cmd_times, ci='sd', capsize=.2)
    elif args.plot_type == 'box':
        sns.boxplot(x=cmd_names, y=cmd_times)
    else:
        print('[WARN] Invalid plot type: {}. Defaulting to bar plot...'.format(args.plot_type))
        sns.barplot(x=cmd_names, y=cmd_times, ci='sd', capsize=.2)

    plt.xticks(rotation='80', fontsize=13)
    plt.title('recon-surf Command Execution Times (Average over {} runs)'.format(len(yaml_dicts)), fontsize=20, pad=15)
    plt.ylabel('Time (minutes)', fontsize=15)
    plt.subplots_adjust(bottom=0.5) 
    plt.grid()
    fig = plt.gcf()
    if args.save_fig:
        fig.savefig('/tmp/recon-surf_times_plot.png')

    if args.plot_recon_all_stages:
        plt.figure(figsize=(18, 8))
        if args.plot_type == 'bar':
            sns.barplot(x=recon_all_stage_names, y=recon_all_stage_times, ci='sd', capsize=.2)
        elif args.plot_type == 'box':
            sns.boxplot(x=recon_all_stage_names, y=recon_all_stage_times)
        else:
            print('[WARN] Invalid plot type: {}. Defaulting to bar plot...'.format(args.plot_type))
            sns.barplot(x=recon_all_stage_names, y=recon_all_stage_times, ci='sd', capsize=.2)

        plt.xticks(rotation='80', fontsize=13)
        plt.title('recon_surf: recon-all Stage Times (Average over {} runs)'.format(len(yaml_dicts)), fontsize=20, pad=15)
        plt.ylabel('Time (minutes)', fontsize=15)
        plt.subplots_adjust(bottom=0.4) 
        plt.grid()
        fig = plt.gcf()
        if args.save_fig:
            fig.savefig('/tmp/recon-surf_recon_all_times_plot.png')

    plt.show()

