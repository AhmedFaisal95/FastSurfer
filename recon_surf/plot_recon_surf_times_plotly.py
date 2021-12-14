#!/usr/bin/env python3
import os
import sys
import argparse

import yaml
import numpy as np
import pandas as pd

import pandas as pd

import plotly
import plotly.express as px

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

plotly_colors = px.colors.qualitative.Plotly


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

def get_yaml_data(root_dir, subject_dirs):
    yaml_dicts = []

    print('[INFO] Extracting data from the files:')
    for subject_dir in subject_dirs:
        if not os.path.isdir(os.path.join(root_dir, subject_dir)):
            continue
        file_path = os.path.join(root_dir, subject_dir, 'scripts/recon_surf_times.log'),
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

    return yaml_dicts

def enforce_custom_side_order(df):
    '''
    Fixing order of within-group bars (plotly can not do this implictly
    as seaborn does through hue_order)
    '''
    df['Side'] = pd.Categorical(df['Side'], ['lh', 'full', 'rh'])
    df = df.sort_values('Side')

    return df

def compute_comparison(df, exemplary_df):
    '''
    NOTE: this functions forces averaging of duplicate rows,
    so std info is lost and would need to maintained elsewhere

    '''
    cols = df.columns.values.tolist()
    rows_list = []

    for index, row in df.iterrows():
        cmd_name = row[cols[0]]

        if cmd_name not in exemplary_df.cmd_names.values:
            continue

        cmd_time = row[cols[1]]
        exemplary_cmd_time = exemplary_df.loc[exemplary_df.cmd_names == cmd_name, 'cmd_times'].values[0]
        if exemplary_cmd_time == 0.0:
            continue
        cmd_time_norm = cmd_time / exemplary_cmd_time

        rows_list.append([cmd_name, cmd_time_norm])

    filtered_df = pd.DataFrame(rows_list, columns=cols)

    return filtered_df

def update_data(fig, df):
    temp = px.histogram(df, x='cmd_names', y='cmd_times',
                        color='Side', barmode='group', histfunc='avg',
                        color_discrete_map={'lh': plotly_colors[4],
                                            'full': plotly_colors[0],
                                            'rh': plotly_colors[2]},
                        )
    if len(fig.data) == 0:
        for i, trace_data in enumerate(temp.data):
            fig.add_trace(trace_data)
    else:
        for i, trace_data in enumerate(temp.data):
            fig.data[i].y = trace_data.y
            fig.data[i].x = trace_data.x

    return fig

def get_fig(df, exemplary_subject_selection, num_subjects):
    ##TODO: Add manual plotting of error bars
    fig = px.histogram(df, x='cmd_names', y='cmd_times',
                        color='Side', barmode='group', histfunc='avg',
                        color_discrete_map={'lh': plotly_colors[4],
                                            'full': plotly_colors[0],
                                            'rh': plotly_colors[2]},
                        )

    fig.update_layout(
        title_text='recon-surf Command Execution Times (Average over {} runs)'.format(num_subjects),
        bargap=0.1,
        bargroupgap=0.0,
        template='seaborn',
        width=1200, height=800,
        legend=dict(title=None, orientation="h", y=1,
                    yanchor="bottom", x=0.5, xanchor="center"
        )
    )
    fig.update_xaxes(categoryorder='mean ascending',
                     tickangle=280,
                     showgrid=True,
                     tickfont={'size': None},
                     title=None,
                     automargin=True)

    no_exemplary_subject_selection = exemplary_subject_selection == 'None' or exemplary_subject_selection is None
    fig.update_yaxes(title='Time (m)' if no_exemplary_subject_selection else '% of exemplary subject: {}'.format(exemplary_subject_selection))

    return fig

def get_box_fig(df, exemplary_subject_selection, num_subjects):
    ##TODO: debug issue of tiny bars (only here in script, not in ipynb)
    fig = px.box(df, x='cmd_names', y='cmd_times',
                  color='Side',
                  color_discrete_map={'lh': plotly_colors[4],
                                      'full': plotly_colors[0],
                                      'rh': plotly_colors[2]},
                  points='all'
                 )

    fig.update_layout(
        title_text='recon-surf Command Execution Times (Average over {} runs)'.format(num_subjects),
        template='seaborn',
        width=1200, height=800,
        legend=dict(title=None, orientation="h", y=1,
                    yanchor="bottom", x=0.5, xanchor="center"
        )
    )
    fig.update_xaxes(categoryorder='mean ascending',
                     tickangle=280,
                     showgrid=True,
                     tickfont={'size': None},
                     title=None)

    no_exemplary_subject_selection = exemplary_subject_selection == 'None' or exemplary_subject_selection is None
    fig.update_yaxes(title='Time (m)' if no_exemplary_subject_selection else '% of exemplary subject: {}'.format(exemplary_subject_selection))

    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--root_dir', type=str, default='.',    
                        help='Root directory containing subject directories')

    args = parser.parse_args()

    start_with_empty_plot = False

    ## Default Data Extraction:
    ## ---------------------------------------------------------------------------

    print('[INFO] Extracting initial data...')
    all_subject_dirs = os.listdir(args.root_dir)

    yaml_dicts = get_yaml_data(args.root_dir, all_subject_dirs)

    cmd_names, cmd_times, recon_all_stage_names, recon_all_stage_times = get_nonunique_cmd_execution_times(yaml_dicts,
                                                                                                           True, True)

    df = pd.DataFrame({'cmd_names': cmd_names, 'cmd_times': cmd_times})
    df = df.groupby('cmd_names', as_index=False).mean()

    base_df = df.copy()
    base_df = separate_hemis(base_df)
    base_df = enforce_custom_side_order(base_df)

    default_cmd_options = np.unique(base_df['cmd_names'].values).tolist()
    cmd_multi_dropdown_options = [{'label': cmd_name, 'value': cmd_name} for cmd_name in default_cmd_options]
    if start_with_empty_plot:
        default_cmd_options = []

    subject_options = [{'label': subject_id, 'value': subject_id} for subject_id in all_subject_dirs]
    # default_subject_dirs = ['382249']
    default_subject_dirs = all_subject_dirs

    exemplary_subject_options = subject_options.copy()
    exemplary_subject_options.append({'label': 'None', 'value': 'None'})

    # draw_debug_borders = True
    draw_debug_borders = False


    ##TODO: Add groove-style borders around main divs, with some padding around and maintaining
    ## current distances between labels and elements
    ## Use CSS: 'border-style': 'groove'

    ## Dash App Layout:
    ## ---------------------------------------------------------------------------

    app = dash.Dash(__name__)

    app.layout = html.Div(children=[
        html.H2(children='recon-surf Command Runtime Analysis',
                style={
                        'textAlign': 'center',
                        'color': None,
                    }),

        # html.Br(),

        html.Div([
                html.Div([
                          html.Label('Subjects:', style={'font-weight': 'bold'}),

                          html.Div([
                                    dcc.Dropdown(
                                        id='subject_selection',
                                        options=subject_options,
                                        value=default_subject_dirs,
                                        multi=True),
                                    ], style={'margin-top': '2%','border':'2px black solid' if draw_debug_borders else None}),

                          html.Br(),
                          html.Label('Exemplary Subject:', style={'font-weight': 'bold'}),

                          html.Div([
                                    dcc.Dropdown(id='exemplary_subject_selection',
                                                 options=exemplary_subject_options,
                                                 value='None'),
                                    ], style={'margin-top': '2%','border':'2px black solid' if draw_debug_borders else None}),

                          html.Br(),

                          html.Label('Parameters:', style={'font-weight': 'bold'}),
                          html.Div([
                                    html.Div([
                                              html.Div([
                                                        html.Div([
                                                                'Time threshold:',
                                                                dcc.Input(id='time_threshold',
                                                                          value=0, type='number'),
                                                                ], style={'width': '80%', 'border':'2px black solid' if draw_debug_borders else None}),
                                                        html.Div([
                                                                'Plot top commands: ',
                                                                dcc.Input(id='top_x',
                                                                          value=0, type='number'),
                                                                ], style={'margin-top': '2%','width': '80%', 'border':'2px black solid' if draw_debug_borders else None}),
                                                        ], style={'display': 'inline-block', 'flexWrap': 'wrap','width': '50%', 'border':'2px black solid' if draw_debug_borders else None}),

                                              html.Div([
                                                  'Plot Type:',
                                                       dcc.RadioItems(
                                                        id='plot_type',
                                                          options=[
                                                              {'label': 'Bar', 'value': 'Bar'},
                                                              {'label': 'Box', 'value': 'Box'},
                                                          ],
                                                          value='Bar')
                                                       ], style={'display': 'inline-block', 'flexWrap': 'wrap', 'verticalAlign': 'top', 'margin-left': '2%', 'border':'2px black solid' if draw_debug_borders else None}),


                                              ], style={'margin-top': '2%','width': '100%', 'border':'2px black solid' if draw_debug_borders else None}, className='row'),
                                  html.Br(),

                                  html.Label('Controls:', style={'font-weight': 'bold'}),
                                  html.Div([
                                            html.Div([
                                                    html.Button(id='reset_state', n_clicks=0, children='Reset', style={'width': '100%', 'height':'100%'}),
                                                    ], style={'display': 'inline-block', 'width': '120px', 'height':'30px' , 'border':'2px black solid' if draw_debug_borders else None}),
                                            html.Div([
                                                    html.Button(id='reload_cmd_state', n_clicks=0, children='Reload Cmds', style={'width': '100%', 'height':'100%'}),
                                                    ], style={'display': 'inline-block','width': '120px', 'margin-left': '2%', 'height':'30px' , 'border':'2px black solid' if draw_debug_borders else None}),
                                            html.Div([
                                                    html.Button(id='load_all_subjs_state', n_clicks=0, children='Load All Subjects', style={'width': '100%', 'height':'100%'}),
                                                    ], style={'display': 'inline-block','width': '120px',  'margin-left': '2%', 'border':'2px black solid' if draw_debug_borders else None, 'height':'30px'}),
                                            ], style={'width':'100%', 'flexWrap': 'wrap','display': 'inline-block', 'margin-top': '2%','border':'2px black solid' if draw_debug_borders else None}),
                                  ], style={'width': None, 'border':'2px black solid' if draw_debug_borders else None}),
                            ], style={'width': '45%', 'display': 'inline-block', 'margin-right': '2%', 'margin-left': '2%', 'border':'2px black solid' if draw_debug_borders else None}),

                html.Div(['Commands:',
                          dcc.Dropdown(
                              id='cmd_selection',
                              options=cmd_multi_dropdown_options,
                              value=default_cmd_options,
                              multi=True),
                          ], style={'width': '45%', 'display': 'inline-block',"verticalAlign": "top" , 'border':'2px black solid' if draw_debug_borders else None}),
                ], className='row'),

        html.Br(),

        html.Div([
                dcc.Graph(
                id='recon-surf-times-plot',
                figure=plotly.graph_objs.Figure()
                )], style={'display': 'flex', 'justify-content':'center', 'border':'2px black solid' if draw_debug_borders else None}),
        ])

    @app.callback(
        Output('recon-surf-times-plot', 'figure'),
        Output('cmd_selection', 'value'),
        Output('reset_state', 'n_clicks'),
        Output('reload_cmd_state', 'n_clicks'),
        Output('load_all_subjs_state', 'n_clicks'),
        Output('subject_selection', 'value'),
        Output('time_threshold', 'value'),
        Output('top_x', 'value'),
        Output('exemplary_subject_selection', 'value'),
        Output('time_threshold', 'disabled'),
        Input('subject_selection', 'value'),
        Input('cmd_selection', 'value'),
        Input('time_threshold', 'value'),
        Input('top_x', 'value'),
        Input('reset_state', 'n_clicks'),
        Input('reload_cmd_state', 'n_clicks'),
        Input('load_all_subjs_state', 'n_clicks'),
        Input('exemplary_subject_selection', 'value'),
        Input('plot_type', 'value'))
    def update_graph(subject_selection, cmd_selection,
                     time_threshold, top_x, reset_state, reload_cmd_state, load_all_subjs_state,
                     exemplary_subject_selection, plot_type):

        disable_time_threshold_option = False

        if reset_state == 1:
            subject_selection = default_subject_dirs
            cmd_selection = default_cmd_options
            time_threshold = 0
            top_x = 0
            exemplary_subject_selection = 'None'

            reset_state = 0

        if load_all_subjs_state == 1:
            subject_selection = all_subject_dirs
            load_all_subjs_state = 0

        yaml_dicts = get_yaml_data(args.root_dir, subject_selection)

        orig_cmd_names, orig_cmd_times, recon_all_stage_names, recon_all_stage_times = get_nonunique_cmd_execution_times(yaml_dicts,
                                                                                                           True, True)
        df = pd.DataFrame({'cmd_names': orig_cmd_names, 'cmd_times': orig_cmd_times})

        plotting_df = df.copy()

        if exemplary_subject_selection != 'None' and exemplary_subject_selection is not None:
            ## TODO: handle case for box plot, where computing means leads to loss of variance info
            ## but not doing so leads to non-unique cmd entries, which compute_comparison can not handle
            plotting_df = plotting_df.groupby('cmd_names', as_index=False).mean()

            exemplary_yaml_dicts = get_yaml_data(args.root_dir, [exemplary_subject_selection])

            cmd_names, cmd_times, recon_all_stage_names, recon_all_stage_times = get_nonunique_cmd_execution_times(exemplary_yaml_dicts,
                                                                                                                   True, True)
            exemplary_df = pd.DataFrame({'cmd_names': cmd_names, 'cmd_times': cmd_times})
            exemplary_df = exemplary_df.groupby('cmd_names', as_index=False).mean()

            plotting_df = compute_comparison(plotting_df, exemplary_df)

            disable_time_threshold_option = True

        plotting_df = separate_hemis(plotting_df)
        plotting_df = enforce_custom_side_order(plotting_df)

        ## Selected cmds:
        if reload_cmd_state == 1:
            excluded_cmds = []

            reload_cmd_state = 0
        else:
            excluded_cmds = [cmd_name for cmd_name in plotting_df.cmd_names.values if cmd_name not in cmd_selection]
        for excluded_cmd in excluded_cmds:
            plotting_df = plotting_df.drop(plotting_df[plotting_df.cmd_names == excluded_cmd].index)

        ## Top x cmds:
        if top_x is not None and top_x != 0:
            increasing_avgs_df = plotting_df.groupby('cmd_names').mean()['cmd_times'].sort_values()
            top_avgs_df = increasing_avgs_df[-top_x:]
            bottom_avgs_df = increasing_avgs_df[:-top_x]

            for cmd_name in bottom_avgs_df.keys():
                plotting_df = plotting_df.drop(plotting_df[plotting_df.cmd_names == cmd_name].index)

        ## Time threshold:
        plotting_df = plotting_df.drop(plotting_df[plotting_df.cmd_times < time_threshold].index)

        if plot_type == 'Bar':
            fig = get_fig(plotting_df, exemplary_subject_selection, len(yaml_dicts))
        if plot_type == 'Box':
            fig = get_box_fig(plotting_df, exemplary_subject_selection, len(yaml_dicts))

        cmd_options = np.unique(plotting_df['cmd_names'].values).tolist()
        new_cmd_multi_dropdown_options = [{'label': cmd_name, 'value': cmd_name} for cmd_name in cmd_options]

        return fig, cmd_options, reset_state, reload_cmd_state, load_all_subjs_state, subject_selection, time_threshold, top_x, exemplary_subject_selection, disable_time_threshold_option 


    app.run_server(debug=True)
