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

##TODO: Set relative import once integrated in package
from plotting_utils import extract_cmd_runtime_data, separate_hemis, get_yaml_data, get_top_x_cmds

plotly_colors = px.colors.qualitative.Plotly


def enforce_custom_side_order(df):
    '''
    Fixing order of within-group bars (plotly can not do this implictly
    as seaborn does through hue_order)
    '''
    df['hemi'] = pd.Categorical(df['hemi'], ['lh', 'both', 'rh'])
    df = df.sort_values('hemi')

    return df

def compute_comparison(df, exemplary_df):
    '''
    NOTE: this functions forces averaging of duplicate rows,
    so std info is lost and would need to maintained elsewhere

    '''
    cols = df.columns.values.tolist()
    rows_list = []

    for index, row in df.iterrows():
        cmd_name = row['cmd_name']

        if cmd_name not in exemplary_df.cmd_name.values:
            continue

        cmd_time = row['execution_time']
        hemi = row['hemi']
        subject_id = row['subject_id']

        try:
            exemplary_cmd_time = exemplary_df[(exemplary_df.cmd_name == cmd_name) & \
                                              (exemplary_df.hemi == hemi)]['execution_time'].item()
        except ValueError as e:
            print(e)
            print('In compute_comparison: Found more than one element satisfying the given condition ' \
                  'within exemplary subject dataframe!' \
                  'Command name = {}\nSide = {}\n'.format(cmd_name, hemi) + \
                  'This may indicate mistake in the input dataframes.')

        if exemplary_cmd_time == 0.0:
            continue
        cmd_time_diff = cmd_time - exemplary_cmd_time

        rows_list.append([cmd_name, cmd_time_diff, subject_id, hemi])

    comparison_df = pd.DataFrame(rows_list, columns=cols)

    return comparison_df

def update_data(fig, df):
    temp = px.histogram(df, x='cmd_name', y='execution_time',
                        color='hemi', barmode='group', histfunc='avg',
                        color_discrete_map={'lh': plotly_colors[4],
                                            'both': plotly_colors[0],
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
    # df.loc[(df['cmd_name'] == 'mris_anatomical_stats') & (df['hemi'] == 'lh'), 'execution_time'] *= 20    ## data manipulated to test max ordering
    fig = px.histogram(df, x='cmd_name', y='execution_time',
                       color='hemi', barmode='group', histfunc='avg',
                       color_discrete_map={'lh': plotly_colors[4],
                                           'both': plotly_colors[0],
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

    means_df = df.groupby(['cmd_name', 'hemi'], as_index=False).mean()   # only used to obtain desired order_array
    order_array = means_df.groupby(['cmd_name'], as_index=False).max().sort_values('execution_time')['cmd_name']
    fig.update_xaxes(categoryorder='array',
                     categoryarray=order_array,
                     tickangle=280,
                     showgrid=True,
                     tickfont={'size': None},
                     title=None,
                     automargin=True)

    no_exemplary_subject_selection = exemplary_subject_selection == 'None' or exemplary_subject_selection is None
    fig.update_yaxes(title='Time (m)' if no_exemplary_subject_selection else 'Difference to exemplary subject: {}'.format(exemplary_subject_selection))

    return fig

def get_bar_fig(df, exemplary_subject_selection, num_subjects):
    means_df = df.groupby(['cmd_name', 'hemi'], as_index=False).mean()
    stds_df = df.groupby(['cmd_name', 'hemi'], as_index=False).std()

    fig = px.bar(means_df, x='cmd_name', y='execution_time',
                 color='hemi', barmode='group',
                 color_discrete_map={'lh': plotly_colors[4],
                                     'both': plotly_colors[0],
                                     'rh': plotly_colors[2]},
                 error_y=stds_df['execution_time'].values,
                 )

    fig.update_layout(
        title_text='recon-surf Command Execution Times (Average over {} runs)'.format(num_subjects),
        template='seaborn',
        width=1200, height=800,
        legend=dict(title=None, orientation="h", y=1,
                    yanchor="bottom", x=0.5, xanchor="center"
        )
    )

    ## Previously, setting categoryorder to 'mean ascending' would correctly sort bars according to the mean,
    ## but as a side-effect caused an issue of xticks not getting removed once a cmd is unselected.
    ## Using 'array', and explicitly providing a sort order in order_array achieves the intended result,
    ## without this undesirable side-effect.
    ## TODO: debug FutureWarning due to max op
    order_array = means_df.groupby(['cmd_name'], as_index=False).max().sort_values('execution_time')['cmd_name']

    fig.update_xaxes(categoryorder='array',
                     categoryarray=order_array,
                     tickangle=280,
                     showgrid=True,
                     tickfont={'size': None},
                     title=None)

    no_exemplary_subject_selection = exemplary_subject_selection == 'None' or exemplary_subject_selection is None
    fig.update_yaxes(title='Time (m)' if no_exemplary_subject_selection else 'Difference to exemplary subject: {}'.format(exemplary_subject_selection))

    return fig

def get_box_fig(df, exemplary_subject_selection, num_subjects):
    ##TODO: debug issue of tiny bars (only here in script, not in ipynb)
    df.to_csv('/tmp/df.csv')
    fig = px.box(df, x='cmd_name', y='execution_time',
                 color='hemi',
                 color_discrete_map={'lh': plotly_colors[4],
                                     'both': plotly_colors[0],
                                     'rh': plotly_colors[2]},
                 points='all',
                 hover_data={'subject_id': True}
                 )

    fig.update_layout(
        title_text='recon-surf Command Execution Times (Average over {} runs)'.format(num_subjects),
        template='seaborn',
        width=1200, height=800,
        legend=dict(title=None, orientation="h", y=1,
                    yanchor="bottom", x=0.5, xanchor="center"
        )
    )

    order_array = df.groupby(['cmd_name'], as_index=False).max().sort_values('execution_time')['cmd_name']
    fig.update_xaxes(categoryorder='array',
                     categoryarray=order_array,
                     tickangle=280,
                     showgrid=True,
                     tickfont={'size': None},
                     title=None)

    no_exemplary_subject_selection = exemplary_subject_selection == 'None' or exemplary_subject_selection is None
    fig.update_yaxes(title='Time (m)' if no_exemplary_subject_selection else 'Difference to exemplary subject: {}'.format(exemplary_subject_selection))

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
    yaml_dicts, all_subject_dirs = get_yaml_data(args.root_dir, all_subject_dirs)
    if len(yaml_dicts) == 0:
        print('[ERROR] No data could be read for processing! Exiting')
        sys.exit()

    cmd_names, execution_time, hemis_list, subject_ids = extract_cmd_runtime_data(yaml_dicts, True)

    df = pd.DataFrame({'cmd_name': cmd_names, 'execution_time': execution_time, 'subject_id': subject_ids})
    base_df = separate_hemis(df, hemis_list)
    base_df = base_df.groupby(['cmd_name', 'hemi'], as_index=False).mean()
    base_df = enforce_custom_side_order(base_df)

    default_cmd_options = np.unique(base_df['cmd_name'].values).tolist()
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

        html.Div([
                html.Div([
                          html.Label('Subjects:', style={'font-weight': 'bold'},
                                     title='Command run-times are averaged over the subjects selected here'),

                          html.Div([
                                    dcc.Dropdown(
                                        id='subject_selection',
                                        options=subject_options,
                                        value=default_subject_dirs,
                                        multi=True),
                                    ], style={'margin-top': '2%','border':'2px black solid' if draw_debug_borders else None}),

                          html.Br(),
                          html.Label('Exemplary Subject:', style={'font-weight': 'bold'},
                                     title='If specified, the plot displays the differences between the selected '
                                            'subjects\' execution times and the exemplary subject\'s mean execution time, for each common command'),

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
                                                                'Time threshold: ',
                                                                dcc.Input(id='time_threshold',
                                                                          value=0, type='number'),
                                                                ], style={'width': '80%', 'border':'2px black solid' if draw_debug_borders else None},
                                                                title='The plot displays only commands whose execution times exceed this threshold'),
                                                        html.Div([
                                                                'Plot top commands: ',
                                                                dcc.Input(id='top_x',
                                                                          value=0, type='number'),
                                                                ], style={'margin-top': '2%','width': '80%', 'border':'2px black solid' if draw_debug_borders else None},
                                                                title='If specified, only the commands with the highest execution times are plotted. This field specifies how many'),
                                                        ], style={'display': 'inline-block', 'flexWrap': 'wrap','width': '50%', 'border':'2px black solid' if draw_debug_borders else None}),

                                              html.Div([
                                                  'Plot Type:',
                                                       dcc.RadioItems(
                                                        id='plot_type',
                                                          options=[
                                                              {'label': 'Bar', 'value': 'Bar'},
                                                              {'label': 'Box', 'value': 'Box'},
                                                              {'label': 'Bar (error bounds)', 'value': 'Bar_2'},
                                                          ],
                                                          value='Bar')
                                                       ], style={'display': 'inline-block', 'flexWrap': 'wrap', 'verticalAlign': 'top', 'margin-left': '2%', 'border':'2px black solid' if draw_debug_borders else None}),


                                              ], style={'margin-top': '2%','width': '100%', 'border':'2px black solid' if draw_debug_borders else None}, className='row'),
                                  html.Br(),

                                  html.Label('Controls:', style={'font-weight': 'bold'}),
                                  html.Div([
                                            html.Div([
                                                    html.Button(id='reset_state', n_clicks=0, children='Reset', style={'width': '100%', 'height':'100%'}),
                                                    ], style={'display': 'inline-block', 'width': '120px', 'height':'30px' , 'border':'2px black solid' if draw_debug_borders else None},
                                                                title='Resets to the initial state. Useful for clearing applying filters and reloading all data'),
                                            html.Div([
                                                    html.Button(id='reload_cmd_state', n_clicks=0, children='Reload Cmds', style={'width': '100%', 'height':'100%'}),
                                                    ], style={'display': 'inline-block','width': '120px', 'margin-left': '2%', 'height':'30px' , 'border':'2px black solid' if draw_debug_borders else None},
                                                    title='Reloads all commands applicable to the current selection of subjects and filters. Useful for loading all valid, unselected commands'),
                                            html.Div([
                                                    html.Button(id='load_all_subjs_state', n_clicks=0, children='Load All Subjects', style={'width': '100%', 'height':'100%'}),
                                                    ], style={'display': 'inline-block','width': '120px',  'margin-left': '2%', 'border':'2px black solid' if draw_debug_borders else None, 'height':'30px'}),
                                            ], style={'width':'100%', 'flexWrap': 'wrap','display': 'inline-block', 'margin-top': '2%','border':'2px black solid' if draw_debug_borders else None},
                                                                title='Loads all valid subject data found in the root directory'),
                                  ], style={'width': None, 'border':'2px black solid' if draw_debug_borders else None}),
                            ], style={'width': '45%', 'display': 'inline-block', 'margin-right': '2%', 'margin-left': '2%', 'verticalAlign': 'top', 'border':'2px black solid' if draw_debug_borders else None}),

                html.Div([
                    html.Label('Commands:', style={'font-weight': 'bold'},
                                title='Commands can be selected or removed from the plot using the drop-down menu. Command names can also be searched for'),
                         html.Div([
                              dcc.Dropdown(
                                  id='cmd_selection',
                                  options=cmd_multi_dropdown_options,
                                  value=default_cmd_options,
                                  multi=True),
                              ], style={'margin-top': '2%', 'width': None, 'display': 'inline-block', 'border':'2px black solid' if draw_debug_borders else None}),
                         ], style={'width': '45%', 'display': 'inline-block', 'margin-right': '2%', 'margin-left': '2%', 'border':'2px black solid' if draw_debug_borders else None}),
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

        yaml_dicts, subject_dirs = get_yaml_data(args.root_dir, subject_selection)

        orig_cmd_names, orig_execution_time, hemis_list, subject_ids = extract_cmd_runtime_data(yaml_dicts)

        df = pd.DataFrame({'cmd_name': orig_cmd_names, 'execution_time': orig_execution_time, 'subject_id': subject_ids})

        plotting_df = df.copy()
        plotting_df = separate_hemis(plotting_df, hemis_list)

        if exemplary_subject_selection != 'None' and exemplary_subject_selection is not None:
            exemplary_yaml_dicts, _ = get_yaml_data(args.root_dir, [exemplary_subject_selection])

            cmd_names, execution_time, hemis_list, subject_ids = extract_cmd_runtime_data(exemplary_yaml_dicts)
            exemplary_df = pd.DataFrame({'cmd_name': cmd_names, 'execution_time': execution_time, 'subject_id': subject_ids})
            exemplary_df = separate_hemis(exemplary_df, hemis_list)
            exemplary_df = exemplary_df.groupby(['cmd_name', 'hemi', 'subject_id'], as_index=False).mean()

            plotting_df = compute_comparison(plotting_df, exemplary_df)

            disable_time_threshold_option = True

        plotting_df = enforce_custom_side_order(plotting_df)

        ## Selected cmds:
        if reload_cmd_state == 1:
            excluded_cmds = []

            reload_cmd_state = 0
        else:
            excluded_cmds = [cmd_name for cmd_name in plotting_df.cmd_name.values if cmd_name not in cmd_selection]
        for excluded_cmd in excluded_cmds:
            plotting_df = plotting_df.drop(plotting_df[plotting_df.cmd_name == excluded_cmd].index)

        ## Top x cmds:
        if top_x is not None and top_x != 0:
            plotting_df = get_top_x_cmds(plotting_df, top_x)

        ## Time threshold:
        if not disable_time_threshold_option:
            plotting_df = plotting_df.drop(plotting_df[plotting_df.execution_time < time_threshold].index)

        if len(plotting_df) != 0:
            if plot_type == 'Bar':
                fig = get_fig(plotting_df, exemplary_subject_selection, len(yaml_dicts))
            elif plot_type == 'Box':
                fig = get_box_fig(plotting_df, exemplary_subject_selection, len(yaml_dicts))
            elif plot_type == 'Bar_2':
                fig = get_bar_fig(plotting_df, exemplary_subject_selection, len(yaml_dicts))
        else:
            fig = plotly.graph_objs.Figure()

        cmd_options = np.unique(plotting_df['cmd_name'].values).tolist()

        return fig, cmd_options, reset_state, reload_cmd_state, load_all_subjs_state, subject_selection, time_threshold, top_x, exemplary_subject_selection, disable_time_threshold_option 


    app.run_server(host='0.0.0.0', port='8060', debug=True)
