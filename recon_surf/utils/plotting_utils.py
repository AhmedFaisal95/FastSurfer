#!/usr/bin/env python3

import os

import yaml
from yaml import CLoader
import numpy as np
import pandas as pd


def extract_cmd_runtime_data(yaml_dicts, split_recon_all_stages=True):
    """
    Extract recon-surf command run-time information from yaml dictionaries obtained from
    recon-surf_times.yaml files.

    :param list yaml_dicts: dicts extracted using get_yaml_data()
    :param bool split_recon_all_stages: whether to separate recon-all stages

    :return: list cmd_names: unique command names
    :return: list execution_times: corresponding command run-times
    :return: list hemis_list: corresponding command side (lh, rh, or both)
    :return: list subject_ids: corresponding subject IDs
    """
    cmd_names = []
    execution_times = []
    hemis_list = []
    subject_ids = []

    for subject_dir, yaml_dict in yaml_dicts.items():
        for stage_num in range(len(yaml_dict['recon-surf_commands'])):
            entry_list = list(yaml_dict['recon-surf_commands'][stage_num].values())[0]

            for cmd_entry in entry_list:
                hemi = None
                if any(lh_str in cmd_entry['cmd'] for lh_str in ['lh', '255 ']):
                    hemi = 'lh'
                elif any(rh_str in cmd_entry['cmd'] for rh_str in ['rh', '127 ']):
                    hemi = 'rh'
                else:
                    hemi = 'both'

                if cmd_entry['cmd'].split(' ')[0] == 'recon-all':
                    if not split_recon_all_stages:
                        cmd_names.append(cmd_entry['cmd'].split(' ')[0])
                        if 'duration_m' in cmd_entry.keys():
                            execution_times.append(cmd_entry['duration_m'])
                        elif 'duration_s' in cmd_entry.keys():
                            execution_times.append(cmd_entry['duration_s'])

                        hemis_list.append(hemi)
                        subject_ids.append(yaml_dict['subject_id'])

                    else:
                        for stage_dict in cmd_entry['stages']:
                            cmd_names.append('recon-all:'+stage_dict['stage_name'])
                            if 'duration_m' in stage_dict.keys():
                                execution_times.append(stage_dict['duration_m'])
                            elif 'duration_s' in stage_dict.keys():
                                execution_times.append(stage_dict['duration_s'])
                            hemis_list.append(hemi)
                            subject_ids.append(yaml_dict['subject_id'])

                else:
                    ## If python3 script, get script name:
                    if 'python3' in cmd_entry['cmd'].split(' ')[0]:
                        cmd_names.append(cmd_entry['cmd'].split(' ')[1].split('/')[-1])
                    else:
                        cmd_names.append(cmd_entry['cmd'].split(' ')[0])

                    if 'duration_m' in cmd_entry.keys():
                        execution_times.append(cmd_entry['duration_m'])
                    elif 'duration_s' in cmd_entry.keys():
                        execution_times.append(cmd_entry['duration_s'])

                    hemis_list.append(hemi)
                    subject_ids.append(yaml_dict['subject_id'])

    return cmd_names, execution_times, hemis_list, subject_ids

def separate_hemis(filtered_df, hemis_list):
    """
    Separate all commands in the given dataframe according to the hemisphere
    (lh, rh, or both), and return the dataframe with the additional grouping column.

    :param pd.DataFrame filtered_df: dataframe containing command run-time information
    :param list hemis_list: command sides (lh, rh, or both) corresponding to every entry in filtered_df

    :return: pd.DataFrame two_side_filtered_df: dataframe following hemisphere grouping
    """
    cols = filtered_df.columns.values.tolist()
    cols.append('hemi')
    rows_list = []
        
    for index, row in filtered_df.iterrows():
        cmd_name = row['cmd_name']
        cmd_time = row['execution_time']
        subject_id = row['subject_id']
        hemi = hemis_list[index]

        if 'recon-all' in cmd_name:
            if 'lh' in cmd_name:
                cmd_name = cmd_name.replace('lh', '')
                if cmd_name[-1] == ' ':
                    cmd_name = cmd_name[:-1]
            if 'rh' in cmd_name:
                cmd_name = cmd_name.replace('rh', '')
                if cmd_name[-1] == ' ':
                    cmd_name = cmd_name[:-1]

        rows_list.append([cmd_name, cmd_time, subject_id, hemi])
        
    two_sided_filtered_df = pd.DataFrame(rows_list, columns=cols)
    
    return two_sided_filtered_df 

def get_yaml_data(root_dir, subject_dirs):
    """
    Extracts data from all recon-surf_times.yaml files in the given subject directories
    within the given root diectory.

    :param str root_dir: full path to the root directory containing subject sub-directories
    :param list subject_dirs: the subject IDs for which runtime dats should be extracted

    :return: pd.DataFrame two_side_filtered_df: dataframe following hemisphere grouping
    """
    yaml_dicts = {}

    print('[INFO] Extracting data from the files:')
    for subject_dir in subject_dirs:
        if not os.path.isdir(os.path.join(root_dir, subject_dir)):
            continue
        file_path = os.path.join(root_dir, subject_dir, 'scripts/recon-surf_times.yaml')
        print('  - {}'.format(file_path))
        try:
            with open(file_path, 'r') as stream:
                try:
                    yaml_dicts[subject_dir] = yaml.load(stream, Loader=CLoader)
                except yaml.YAMLError as e:
                    print(e)

        except FileNotFoundError as e:
            print(e)
            print('[INFO] Skipping this file...')

    return yaml_dicts

def get_top_x_cmds(plotting_df, num_cmds):
    """
    Filter entries in the given dataframe to only include the x commands with the
    highest average run-times.
    Note that commands that run seperately on both hemispheres are ranked based on
    max(lh, rh).

    :param pd.DataFrame plotting_df: dataframe containing command run-time information
    :param int num_cmds: number of commands to plot

    :return: pd.DataFrame plotting_df: filtered dataframe
    """
    means_df = plotting_df.groupby(['cmd_name', 'hemi'], as_index=False).mean()
    means_df = means_df.loc[means_df['execution_time'] == means_df['execution_time']]   ## Remove NaN entries produced by groupby
    means_df['execution_time'] = means_df['execution_time'].apply(lambda x: np.abs(x))

    ordered_means_df = means_df.loc[reversed(means_df['execution_time'].sort_values().index)]
    top_unique_cmds = []
    counter = 0

    for index, row in ordered_means_df.iterrows():
        cmd_name = row['cmd_name']
        if cmd_name not in top_unique_cmds:
            top_unique_cmds.append(cmd_name)
            counter += 1

        if counter == num_cmds:
            break

    excluded_cmds = [cmd_name for cmd_name in np.unique(means_df.cmd_name.values).tolist() if cmd_name not in top_unique_cmds]

    for cmd_name in excluded_cmds:
        plotting_df = plotting_df.drop(plotting_df[plotting_df.cmd_name == cmd_name].index)

    return plotting_df, top_unique_cmds

def get_selected_cmds(plotting_df, selected_cmds):
    """
    Filter entries in the given dataframe to only include the given list of named
    commands.

    :param pd.DataFrame plotting_df: dataframe containing command run-time information
    :param list selected_cmds: the names of commands to be plotted

    :return: pd.DataFrame plotting_df: filtered dataframe
    """
    excluded_cmds = [cmd_name for cmd_name in plotting_df.cmd_name.values if cmd_name not in selected_cmds]

    for excluded_cmd in excluded_cmds:
        plotting_df = plotting_df.drop(plotting_df[plotting_df.cmd_name == excluded_cmd].index)

    return plotting_df

def get_runtimes_exceeding_threshold(plotting_df, time_threshold):
    """
    Filter entries in the given dataframe to only include commands whose run-times exceed
    the given time threshold.

    :param pd.DataFrame plotting_df: dataframe containing command run-time information
    :param float time_threshold: the lower bound on run-time values to include in plot

    :return: pd.DataFrame plotting_df: filtered dataframe
    """
    plotting_df = plotting_df.drop(plotting_df[np.abs(plotting_df.execution_time) < time_threshold].index)

    return plotting_df
