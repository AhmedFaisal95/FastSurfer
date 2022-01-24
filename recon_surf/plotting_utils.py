#!/usr/bin/env python3

import os
import yaml

import numpy as np
import pandas as pd


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

    excluded_cmds = [cmd_name for cmd_name in np.unique(means_df.cmd_names.values).tolist() if cmd_name not in top_unique_cmds]

    for cmd_name in excluded_cmds:
        plotting_df = plotting_df.drop(plotting_df[plotting_df.cmd_names == cmd_name].index)

    return plotting_df