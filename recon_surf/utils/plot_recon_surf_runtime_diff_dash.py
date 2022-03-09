#!/usr/bin/env python3
import os
import sys
import argparse

import yaml
import numpy as np
import pandas as pd

import plotly
import plotly.express as px

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

##TODO: Set relative import once integrated in package
from plotting_utils import extract_cmd_runtime_data, separate_hemis, get_yaml_data, get_top_x_cmds, get_selected_cmds, get_runtimes_exceeding_threshold

plotly_colors = px.colors.qualitative.Plotly

all_subjects_yaml_dicts = None

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

def get_fig(df, exemplary_subject_selection, num_subjects, orient='horizontal'):
    if orient == 'vertical':
        x = 'cmd_name'
        y = 'execution_time'
    elif orient == 'horizontal':
        x = 'execution_time'
        y = 'cmd_name'

    fig = px.histogram(df, x=x, y=y,
                       color='hemi', barmode='group', histfunc='avg',
                       color_discrete_map={'lh': plotly_colors[4],
                                           'both': plotly_colors[0],
                                           'rh': plotly_colors[2]},
                       )

    fig.update_layout(
        title_text='recon-surf Command Execution Time Differences (Dir 2 - Dir 1; Average over {} runs)'.format(num_subjects),
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

    if orient == 'vertical':
        fig.update_xaxes(categoryorder='array',
                         categoryarray=order_array,
                         tickangle=280,
                         showgrid=True,
                         ticks='outside',
                         ticklen=7,
                         title=None,
                         automargin=True)
    elif orient == 'horizontal':
        fig.update_yaxes(categoryorder='array',
                         categoryarray=order_array,
                         showgrid=True,
                         ticks='outside',
                         ticklen=7,
                         title=None,
                         automargin=True)

    no_exemplary_subject_selection = exemplary_subject_selection == 'None' or exemplary_subject_selection is None
    if orient == 'vertical':
        fig.update_yaxes(title='Time Difference (m)' if no_exemplary_subject_selection else 'Difference to exemplary subject: {}'.format(exemplary_subject_selection))
    elif orient == 'horizontal':
        fig.update_xaxes(title='Time Difference (m)' if no_exemplary_subject_selection else 'Difference to exemplary subject: {}'.format(exemplary_subject_selection))

    return fig

def get_bar_fig(df, exemplary_subject_selection, num_subjects, orient='horizontal'):
    means_df = df.groupby(['cmd_name', 'hemi'], as_index=False).mean()
    stds_df = df.groupby(['cmd_name', 'hemi'], as_index=False).std()

    if orient == 'vertical':
        fig = px.bar(means_df, x='cmd_name', y='execution_time',
                     color='hemi', barmode='group',
                     color_discrete_map={'lh': plotly_colors[4],
                                         'both': plotly_colors[0],
                                         'rh': plotly_colors[2]},
                     error_y=stds_df['execution_time'].values,
                     )
    elif orient == 'horizontal':
        fig = px.bar(means_df, x='execution_time', y='cmd_name',
                     color='hemi', barmode='group',
                     color_discrete_map={'lh': plotly_colors[4],
                                         'both': plotly_colors[0],
                                         'rh': plotly_colors[2]},
                     error_x=stds_df['execution_time'].values,
                     )

    fig.update_layout(
        title_text='recon-surf Command Execution Time Differences (Dir 2 - Dir 1; Average over {} runs)'.format(num_subjects),
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

    if orient == 'vertical':
        fig.update_xaxes(categoryorder='array',
                         categoryarray=order_array,
                         tickangle=280,
                         showgrid=True,
                         ticks='outside',
                         ticklen=7,
                         title=None)
    elif orient == 'horizontal':
        fig.update_yaxes(categoryorder='array',
                         categoryarray=order_array,
                         showgrid=True,
                         ticks='outside',
                         ticklen=7,
                         title=None,
                         automargin=True)

    no_exemplary_subject_selection = exemplary_subject_selection == 'None' or exemplary_subject_selection is None
    if orient == 'vertical':
        fig.update_yaxes(title='Time Difference (m)' if no_exemplary_subject_selection else 'Difference to exemplary subject: {}'.format(exemplary_subject_selection))
    elif orient == 'horizontal':
        fig.update_xaxes(title='Time Difference (m)' if no_exemplary_subject_selection else 'Difference to exemplary subject: {}'.format(exemplary_subject_selection))

    return fig

def get_box_fig(df, exemplary_subject_selection, num_subjects, orient='horizontal'):
    if orient == 'vertical':
        x = 'cmd_name'
        y = 'execution_time'
    elif orient == 'horizontal':
        x = 'execution_time'
        y = 'cmd_name'

    fig = px.box(df, x=x, y=y,
                 color='hemi',
                 color_discrete_map={'lh': plotly_colors[4],
                                     'both': plotly_colors[0],
                                     'rh': plotly_colors[2]},
                 points='all',
                 hover_data={'subject_id': True},
                 boxmode='group'
                 )

    fig.update_layout(
        title_text='recon-surf Command Execution Time Differences (Dir 2 - Dir 1; Average over {} runs)'.format(num_subjects),
        template='seaborn',
        width=1200, height=800,
        legend=dict(title=None, orientation="h", y=1,
                    yanchor="bottom", x=0.5, xanchor="center"
        )
    )

    order_array = df.groupby(['cmd_name'], as_index=False).max().sort_values('execution_time')['cmd_name']
    if orient == 'vertical':
        fig.update_xaxes(categoryorder='array',
                         categoryarray=order_array,
                         tickangle=280,
                         showgrid=True,
                         ticks='outside',
                         ticklen=7,
                         title=None)
    elif orient == 'horizontal':
        fig.update_yaxes(categoryorder='array',
                         categoryarray=order_array,
                         showgrid=True,
                         ticks='outside',
                         ticklen=7,
                         title=None)

    no_exemplary_subject_selection = exemplary_subject_selection == 'None' or exemplary_subject_selection is None
    if orient == 'vertical':
        fig.update_yaxes(title='Time Difference (m)' if no_exemplary_subject_selection else 'Difference to exemplary subject: {}'.format(exemplary_subject_selection))
    elif orient == 'horizontal':
        fig.update_xaxes(title='Time Difference (m)' if no_exemplary_subject_selection else 'Difference to exemplary subject: {}'.format(exemplary_subject_selection))

    fig.update_layout(
        boxgap=0.0,
        boxgroupgap=0.0
    )

    return fig

def get_execution_time_diff_df(base_df_1, base_df_2, verbose=False):
    base_df_1['cmd_name'] = base_df_1['cmd_name'].apply(lambda x: x.split('/')[-1] if '/' in x else x)
    base_df_2['cmd_name'] = base_df_2['cmd_name'].apply(lambda x: x.split('/')[-1] if '/' in x else x)

    cols = base_df_2.columns.values.tolist()
    rows_list = []

    for index, row in base_df_2.iterrows():
        cmd_name = row['cmd_name']
        subject_id = row['subject_id']
        hemi = row['hemi']

        base_df_1_entry = base_df_1.loc[(base_df_1['cmd_name'] == cmd_name) & \
                                        (base_df_1['subject_id'] == subject_id) & \
                                        (base_df_1['hemi'] == hemi)]

        if len(base_df_1_entry) == 0:
            if verbose:
                print('[WARN] First dataframe does not have an entry with the following details:')
                print('cmd_name: {}, hemi: {}, subject_id: {}'.format(cmd_name, hemi, subject_id))
                print('Will skip this data point!\n')
        else:
            cmd_time = row['execution_time']
            diff = cmd_time - base_df_1_entry.execution_time.values.item()

            rows_list.append([cmd_name, hemi, subject_id, diff])

    temp_df = pd.DataFrame(rows_list, columns=cols)

    for index, row in base_df_1.iterrows():
        cmd_name = row['cmd_name']
        subject_id = row['subject_id']
        hemi = row['hemi']

        temp_df_entry = temp_df.loc[(temp_df['cmd_name'] == cmd_name) & \
                                    (temp_df['subject_id'] == subject_id) & \
                                    (temp_df['hemi'] == hemi)]

        if len(temp_df_entry) != 0:
            continue

        base_df_2_entry = base_df_2.loc[(base_df_2['cmd_name'] == cmd_name) & \
                                        (base_df_2['subject_id'] == subject_id) & \
                                        (base_df_2['hemi'] == hemi)]

        if len(base_df_2_entry) == 0:
            if verbose:
                print('[WARN] Second dataframe does not have an entry with the following details:')
                print('cmd_name: {}, hemi: {}, subject_id: {}'.format(cmd_name, hemi, subject_id))
                print('Will skip this data point!\n')

        else:
            cmd_time = row['execution_time']
            diff = base_df_2_entry.execution_time.values.item() - cmd_time

            rows_list.append([cmd_name, diff, subject_id, hemi])

    return pd.DataFrame(rows_list, columns=cols)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-r1','--root_dir_1', type=str, required=True,
                        help='Root directory containing subject directories')
    parser.add_argument('-r2','--root_dir_2', type=str, required=True,
                        help='Root directory containing subject directories')
    parser.add_argument('-o','--output_path', type=str, default='/tmp/plot_recon_surf_runtime_diff_dash_figure.png',
                        help='Path to the file in which plot images will be saved.')
    parser.add_argument('--save_fig_and_exit', dest='save_fig_and_exit', action='store_true')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true')

    args = parser.parse_args()

    start_with_empty_plot = False
    draw_debug_borders = False

    ## Default Data Extraction:
    ## ---------------------------------------------------------------------------

    print('[INFO] Extracting initial data...')
    all_subdirs_1 = os.listdir(args.root_dir_1)
    all_subjects_yaml_dicts_1 = get_yaml_data(args.root_dir_1, all_subdirs_1)
    if len(all_subjects_yaml_dicts_1) == 0:
        print('[ERROR] No data could be read for processing in root_dir_1! Exiting')
        sys.exit()

    all_subdirs_2 = os.listdir(args.root_dir_2)
    all_subjects_yaml_dicts_2 = get_yaml_data(args.root_dir_2, all_subdirs_2)
    if len(all_subjects_yaml_dicts_2) == 0:
        print('[ERROR] No data could be read for processing in root_dir_1! Exiting')
        sys.exit()

    all_subject_dirs_1 = list(all_subjects_yaml_dicts_1.keys())
    all_subject_dirs_2 = list(all_subjects_yaml_dicts_2.keys())

    cmd_names_1, execution_time_1, hemis_list_1, subject_ids_1 = extract_cmd_runtime_data(all_subjects_yaml_dicts_1, True)

    base_df_1 = pd.DataFrame({'cmd_name': cmd_names_1, 'execution_time': execution_time_1, 'subject_id': subject_ids_1})
    base_df_1 = separate_hemis(base_df_1, hemis_list_1)
    ## NOTE: here, if a subject has multiple calls of a given cmd, we average over the individual call values
    base_df_1 = base_df_1.groupby(['cmd_name', 'hemi', 'subject_id'], as_index=False).mean()
    base_df_1 = enforce_custom_side_order(base_df_1)

    cmd_names_2, execution_time_2, hemis_list_2, subject_ids_2 = extract_cmd_runtime_data(all_subjects_yaml_dicts_2, True)

    base_df_2 = pd.DataFrame({'cmd_name': cmd_names_2, 'execution_time': execution_time_2, 'subject_id': subject_ids_2})
    base_df_2 = separate_hemis(base_df_2, hemis_list_2)
    base_df_2 = base_df_2.groupby(['cmd_name', 'hemi', 'subject_id'], as_index=False).mean()
    base_df_2 = enforce_custom_side_order(base_df_2)

    print('[INFO] Computing run-time differences...')
    diff_df = get_execution_time_diff_df(base_df_1, base_df_2, args.verbose)

    all_subject_dirs = diff_df.subject_id.unique().tolist()

    if args.save_fig_and_exit:
        fig = get_box_fig(diff_df, None, len(all_subject_dirs), 'horizontal')

        print('[INFO] Saving figure and exiting...')
        fig.write_image(args.output_path)
        sys.exit()

    default_cmd_options = np.unique(diff_df['cmd_name'].values).tolist()
    cmd_multi_dropdown_options = [{'label': cmd_name, 'value': cmd_name} for cmd_name in default_cmd_options]
    if start_with_empty_plot:
        default_cmd_options = []

    subject_options = [{'label': subject_id, 'value': subject_id} for subject_id in all_subject_dirs]
    default_subject_dirs = all_subject_dirs

    exemplary_subject_options = subject_options.copy()
    exemplary_subject_options.append({'label': 'None', 'value': 'None'})

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
                                                 disabled=True,
                                                 value='None'),
                                    ], style={'margin-top': '2%','border':'2px black solid' if draw_debug_borders else None}),

                          html.Br(),

                          html.Label('Parameters:', style={'font-weight': 'bold'}),
                          html.Div([
                                    html.Div([
                                              html.Div([
                                                        html.Div([
                                                                'Diff. threshold: ',
                                                                dcc.Input(id='diff_threshold',
                                                                          value=0, type='number',
                                                                          style={'float':'right'}),
                                                                ], style={'margin-top': '2%','width': '80%', 'border':'2px black solid' if draw_debug_borders else None},
                                                                title='The plot displays only commands whose execution times exceed this threshold'),
                                                        html.Div([
                                                                'Plot top commands: ',
                                                                dcc.Input(id='top_x',
                                                                          value=0, type='number',
                                                                          style={'float':'right'}),
                                                                ], style={'margin-top': '2%','width': '80%', 'border':'2px black solid' if draw_debug_borders else None},
                                                                title='If specified, only the commands with the highest execution times are plotted. This field specifies how many.\nNOTE: Command execution times are ranked based on their absolute values'),
                                                        ], style={'display': 'inline-block', 'flexWrap': 'wrap','width': '50%', 'padding-right':'1%','border':'2px black solid' if draw_debug_borders else None}),

                                              html.Div([
                                                        html.Div([
                                                                 'Plot Type:',
                                                                      dcc.RadioItems(
                                                                       id='plot_type',
                                                                         options=[
                                                                             {'label': 'Bar', 'value': 'Bar'},
                                                                             {'label': 'Box', 'value': 'Box'},
                                                                             {'label': 'Bar (error bounds)', 'value': 'Bar_2'},
                                                                         ],
                                                                         value='Box'),
                                                                 ], style={'margin-top': '2%', 'border':'2px black solid' if draw_debug_borders else None}),

                                                        html.Div([
                                                                 'Plot Orientiation:',
                                                                      dcc.RadioItems(
                                                                       id='plot_orientation',
                                                                         options=[
                                                                             {'label': 'Horizontal', 'value': 'horizontal'},
                                                                             {'label': 'Vertical', 'value': 'vertical'},
                                                                         ],
                                                                         value='horizontal')
                                                                 ], style={'margin-top': '2%', 'border':'2px black solid' if draw_debug_borders else None}),
                                                       ], style={'display': 'inline-block', 'flexWrap': 'wrap', 'verticalAlign': 'top', 'border':'2px black solid' if draw_debug_borders else None}),


                                              ], style={'margin-top': '0%','width': '100%', 'border':'2px black solid' if draw_debug_borders else None}, className='row'),
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
                            ], style={'width': '45%', 'display': 'inline-block', 'margin-left': '2%', 'verticalAlign': 'top', 'border':'2px black solid' if draw_debug_borders else None,
                             'border-style':'groove', 'padding': '10px'}),

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
                         ], style={'width': '45%', 'display': 'inline-block', 'margin-left': '2%', 'border':'2px black solid' if draw_debug_borders else None,
                                   'border-style':'groove', 'padding': '10px'}),
                ], className='row'),

        html.Br(),

        html.Div([
                dcc.Graph(
                id='recon-surf-times-plot',
                figure=plotly.graph_objs.Figure()
                )], style={'display': 'flex', 'justify-content':'center', 'border':'2px black solid' if draw_debug_borders else None,
                           'border-style':'groove', 'margin-left':'2%', 'margin-right':'3%'}),
        ])

    @app.callback(
        Output('recon-surf-times-plot', 'figure'),
        Output('cmd_selection', 'value'),
        Output('reset_state', 'n_clicks'),
        Output('reload_cmd_state', 'n_clicks'),
        Output('load_all_subjs_state', 'n_clicks'),
        Output('subject_selection', 'value'),
        Output('diff_threshold', 'value'),
        Output('top_x', 'value'),
        Output('exemplary_subject_selection', 'value'),
        Output('diff_threshold', 'disabled'),
        Output('exemplary_subject_selection', 'disabled'),
        Input('subject_selection', 'value'),
        Input('cmd_selection', 'value'),
        Input('diff_threshold', 'value'),
        Input('top_x', 'value'),
        Input('reset_state', 'n_clicks'),
        Input('reload_cmd_state', 'n_clicks'),
        Input('load_all_subjs_state', 'n_clicks'),
        Input('exemplary_subject_selection', 'value'),
        Input('plot_type', 'value'),
        Input('plot_orientation', 'value'))
    def update_graph(subject_selection, cmd_selection,
                     diff_threshold, top_x, reset_state, reload_cmd_state, load_all_subjs_state,
                     exemplary_subject_selection, plot_type, plot_orientation):

        disable_diff_threshold_option = False

        if reset_state == 1:
            subject_selection = default_subject_dirs
            cmd_selection = default_cmd_options
            diff_threshold = 0
            top_x = 0
            exemplary_subject_selection = 'None'

            reset_state = 0

        if load_all_subjs_state == 1:
            subject_selection = all_subject_dirs
            load_all_subjs_state = 0

        yaml_dicts_1 = {subject_dir: all_subjects_yaml_dicts_1[subject_dir] for subject_dir in subject_selection}
        cmd_names_1, execution_time_1, hemis_list_1, subject_ids_1 = extract_cmd_runtime_data(yaml_dicts_1)

        df_1 = pd.DataFrame({'cmd_name': cmd_names_1, 'execution_time': execution_time_1, 'subject_id': subject_ids_1})
        df_1 = separate_hemis(df_1, hemis_list_1)
        ## NOTE: here, if a subject has multiple calls of a given cmd, we average over the individual call values
        df_1 = df_1.groupby(['cmd_name', 'hemi', 'subject_id'], as_index=False).mean()

        yaml_dicts_2 = {subject_dir: all_subjects_yaml_dicts_2[subject_dir] for subject_dir in subject_selection}
        cmd_names_2, execution_time_2, hemis_list_2, subject_ids_2 = extract_cmd_runtime_data(yaml_dicts_2)

        df_2 = pd.DataFrame({'cmd_name': cmd_names_2, 'execution_time': execution_time_2, 'subject_id': subject_ids_2})
        df_2 = separate_hemis(df_2, hemis_list_2)
        df_2 = df_2.groupby(['cmd_name', 'hemi', 'subject_id'], as_index=False).mean()

        ## NOTE: at the moment, always df_2 - df_1:
        plotting_df= get_execution_time_diff_df(df_1, df_2, args.verbose)

        ## TODO: incorporate or remove exemplary subject comaprison
        # if exemplary_subject_selection != 'None' and exemplary_subject_selection is not None:
        #     exemplary_yaml_dicts = all_subjects_yaml_dicts[exemplary_subject_selection]

        #     cmd_names, execution_time, hemis_list, subject_ids = extract_cmd_runtime_data(exemplary_yaml_dicts)
        #     exemplary_df = pd.DataFrame({'cmd_name': cmd_names, 'execution_time': execution_time, 'subject_id': subject_ids})
        #     exemplary_df = separate_hemis(exemplary_df, hemis_list)
        #     exemplary_df = exemplary_df.groupby(['cmd_name', 'hemi', 'subject_id'], as_index=False).mean()

        #     plotting_df = compute_comparison(plotting_df, exemplary_df)

        #     disable_diff_threshold_option = True

        plotting_df = enforce_custom_side_order(plotting_df)

        ## Apply filters:
        if reload_cmd_state == 1:
            reload_cmd_state = 0
        else:
            plotting_df = get_selected_cmds(plotting_df, cmd_selection)

        if top_x is not None and top_x != 0:
            plotting_df, _ = get_top_x_cmds(plotting_df, top_x)

        if not disable_diff_threshold_option and diff_threshold != 0:
            plotting_df = get_runtimes_exceeding_threshold(plotting_df, diff_threshold)

        if len(plotting_df) != 0:
            if plot_type == 'Bar':
                fig = get_fig(plotting_df, exemplary_subject_selection, plotting_df.subject_id.unique().size, plot_orientation)
            elif plot_type == 'Box':
                fig = get_box_fig(plotting_df, exemplary_subject_selection, plotting_df.subject_id.unique().size, plot_orientation)
            elif plot_type == 'Bar_2':
                fig = get_bar_fig(plotting_df, exemplary_subject_selection, plotting_df.subject_id.unique().size, plot_orientation)
        else:
            fig = plotly.graph_objs.Figure()

        fig.write_image(args.output_path)

        cmd_options = np.unique(plotting_df['cmd_name'].values).tolist()

        return fig, cmd_options, reset_state, reload_cmd_state, load_all_subjs_state, subject_selection, diff_threshold, top_x, exemplary_subject_selection, disable_diff_threshold_option, True


    app.run_server(host='0.0.0.0', port='8060', debug=True)
