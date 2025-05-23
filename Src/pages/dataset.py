# pages/dataset.py
import dash
from dash import html, dcc
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import os

import assets.dash_styles as dash_styles

# Dont removed even if it looks unused.
import callbacks.datasetCallbacks

import diskcache
cache = diskcache.Cache("/cache")

cpu_cores = os.cpu_count()

dash.register_page(
    __name__,
    path="/",
    name="Dataset",
)  # The default/home page

tab1 = (
    html.Div(
        [
            dcc.Store(id="shared-data"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4(
                                        "Input CSV information and your prefered thresholds:"
                                    ),
                                    dbc.Row(
                                        [
                                            html.Div(
                                                [
                                                    dcc.Input(
                                                        id="delim-input",
                                                        type="text",
                                                        value="",
                                                        placeholder="Input csv delimiter",
                                                        style={"margin": "5px"},
                                                    ),
                                                    dcc.Input(
                                                        id="null-value-input",
                                                        type="text",
                                                        value="",
                                                        placeholder="Input csv Null Placeholder",
                                                        style={"margin": "5px"},
                                                    ),
                                                    html.Span(
                                                        "❓",
                                                        id="csv-setting-tooltip",
                                                        style={
                                                            "cursor": "pointer",
                                                            "fontSize": "20px",
                                                            "marginLeft": "5px",
                                                        },
                                                    ),
                                                    dbc.Tooltip(
                                                        [
                                                            html.P(
                                                                "Delimiters used in CSV files can vary, default set to ','. After initiation it is set to ',' for the rest of this program."
                                                            ),
                                                            html.P(
                                                                "Values representing Null values in CSV can vary, default set to '' (empty)"
                                                            ),
                                                        ],
                                                        target="csv-setting-tooltip",  # Matches the icon ID
                                                        placement="right",  # Position of the tooltip
                                                    ),
                                                ]
                                            ),
                                            html.Div(
                                                [
                                                    dcc.Input(
                                                        id="null-threshold-input",
                                                        type="text",
                                                        value="",
                                                        placeholder="NaN Threshold %",
                                                        style={"margin": "5px"},
                                                    ),
                                                    dcc.Input(
                                                        id="repetitive-threshold-input",
                                                        type="text",
                                                        value="",
                                                        placeholder="Repetitive value %",
                                                        style={"margin": "5px"},
                                                    ),
                                                    dcc.Input(
                                                        id="categorical-threshold-input",
                                                        type="text",
                                                        value="",
                                                        placeholder="Unique Categorical threshold",
                                                        style={"margin": "5px"},
                                                    ),
                                                    html.Span(
                                                        "❓",
                                                        id="thresholds-tooltip",
                                                        style={
                                                            "cursor": "pointer",
                                                            "fontSize": "20px",
                                                            "marginLeft": "5px",
                                                        },
                                                    ),
                                                    dbc.Tooltip(
                                                        [
                                                            html.P(
                                                                "NaN Value Threshold: Minimum number of NaN values in a column (%) to be treated as a error. (WARNING: Setting the treshold above 40 is NOT recommended)."
                                                            ),
                                                            html.P(
                                                                "Repetive Value: The maximum number (%) of a single unique value in a column. All columns concisting of a single value higher than this percent are treated as an error."
                                                            ),
                                                            html.P(
                                                                "Categorical Threshold: If a Categorical column contains more than this amount of unique Values it will be treated as an error. Reason: columns with many unique categorical values will bloat the amount of columns when performing one-hot encoding."
                                                            ),
                                                        ],
                                                        target="thresholds-tooltip",  # Matches the icon ID
                                                        placement="right",  # Position of the tooltip
                                                    ),
                                                ],
                                                className="mb-3",
                                            ),
                                            # html.H3("Press Analyze to Search the csv file based on your thresholds:"),
                                            dbc.Button(
                                                "Analyze",
                                                id="analyze-btn",
                                                n_clicks=0,
                                                style={"width": "20%", "margin": "5px"},
                                            ),
                                            dbc.InputGroup(
                                                [
                                                    html.H4(
                                                        "Perform actions on selected:"
                                                    ),
                                                    dbc.Col(
                                                        html.Span(
                                                            "❓",
                                                            id="analyze-tooltip",
                                                            style={
                                                                "cursor": "pointer",
                                                                "fontSize": "20px",
                                                                "marginLeft": "5px",
                                                            },
                                                        ),
                                                        width="auto",
                                                    ),
                                                    dbc.Tooltip(
                                                        "It is impossible to cover all possible potential errors that a CSV could have. View the list and see potential errors, \
                                select and use buttons to treat the most common issues. But sometimes you have to resolve them yourself in the csv file.",
                                                        # Tooltip text
                                                        target="analyze-tooltip",  # Matches the icon ID
                                                        placement="right",  # Position of the tooltip
                                                    ),
                                                ],
                                                className="mt-3",  # Adds margin below for spacing
                                            ),
                                            dbc.Row(
                                                [
                                                    dbc.Button(
                                                        "Exclude",
                                                        id="exclude-btn",
                                                        n_clicks=0,
                                                        style={
                                                            "width": "20%",
                                                            "margin": "5px",
                                                        },
                                                    ),
                                                    dbc.Button(
                                                        "Fix Commas",
                                                        id="decimal-btn",
                                                        n_clicks=0,
                                                        style={
                                                            "width": "20%",
                                                            "margin": "5px",
                                                        },
                                                    ),
                                                    dbc.Button(
                                                        "Fix Unicode",
                                                        id="unicode-btn",
                                                        n_clicks=0,
                                                        style={
                                                            "width": "20%",
                                                            "margin": "5px",
                                                        },
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                ],
                                # style={"width": "50%"},
                            ),
                            className="shadow-sm",
                            style=dash_styles.card_style_P1,
                        ),
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div(
                                        [
                                            dcc.Loading(
                                                id="loading-error-bar-chart",
                                                children=dcc.Graph(
                                                    id="error-bar-chart",
                                                    figure={
                                                        "data": [],
                                                        "layout": {
                                                            "title": {
                                                                "text": "Amount of Columns with stated errors",
                                                                "x": 0.5,  # Center title
                                                                "xanchor": "center",
                                                                "yanchor": "top",
                                                                "font": {
                                                                    "size": 16
                                                                },  # Reduce title size
                                                            },
                                                            "xaxis": {
                                                                "title": {
                                                                    "text": "Error Type",
                                                                    "font": {
                                                                        "size": 12
                                                                    },
                                                                },
                                                                "showgrid": False,
                                                                "zeroline": False,
                                                                "showticklabels": False,
                                                                "automargin": True,
                                                                "tickangle": -45,  # Rotate labels if needed
                                                            },
                                                            "yaxis": {
                                                                "title": {
                                                                    "text": "Count",
                                                                    "font": {
                                                                        "size": 12
                                                                    },
                                                                },
                                                                "showgrid": False,
                                                                "zeroline": False,
                                                                "showticklabels": False,
                                                                "automargin": True,
                                                            },
                                                            "plot_bgcolor": "rgba(0,0,0,0)",
                                                            "paper_bgcolor": "rgba(0,0,0,0)",
                                                            "margin": {
                                                                "l": 40,
                                                                "r": 20,
                                                                "t": 40,
                                                                "b": 40,
                                                            },  # Reduce excess space
                                                            "annotations": [
                                                                {
                                                                    "text": "Waiting for analyze...",
                                                                    "xref": "paper",
                                                                    "yref": "paper",
                                                                    "showarrow": False,
                                                                    "font": {
                                                                        "size": 16
                                                                    },
                                                                    "x": 0.5,
                                                                    "y": 0.5,
                                                                    "xanchor": "center",
                                                                    "yanchor": "middle",
                                                                }
                                                            ],
                                                        },
                                                    },
                                                    selectedData={},
                                                    config={"displayModeBar": False},
                                                    clickData=None,
                                                    style={
                                                        "height": "300px"
                                                    }, 
                                                ),
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                            className="shadow-sm",
                            style=dash_styles.card_style_P1,
                        ),
                    ),
                ],
                style={"height": "auto", "marginTop": "20px"},
            ),
            html.Div(
                [
                    dcc.Loading(
                        id="loading-aggrid",
                        children=dag.AgGrid(
                            id="health-issues-table",
                            columnDefs=[
                                {
                                    "field": "Column",
                                    "headerName": "Column",
                                    "sortable": True,
                                    "filter": True,
                                },
                                {
                                    "field": "Errors Found",
                                    "headerName": "Errors Found",
                                    "sortable": True,
                                    "filter": True,
                                    "cellRenderer": "markdown",
                                    "autoHeight": True,
                                    "wrapText": True,
                                },
                            ],
                            selectedRows=[None],
                            dashGridOptions={
                                "rowSelection": "multiple",
                                "animateRows": True,
                                "pagination": False,
                                "rowMultiSelectWithClick": True,
                                "suppressMenuHide": True,
                            },
                            defaultColDef={
                                "flex": 1,
                                "minWidth": 150,
                                "minHeight": 350,
                                "autoHeight": True,
                                "wrapText": True,
                            },
                            filterModel={
                                "Errors Found": {
                                    "filterType": "text",
                                    "type": "Contains",
                                    "filter": "NaN",  # ensure clicked_error is the correct string value
                                },
                            },
                            style={"height": "500px"},
                        ),
                    ),
                ]
            ),
        ],
        style={"marginBottom": "50px"},
    ),
)
#
################################ Outliers
#
tab2 = (
    html.Div(
        [
            dcc.Store(id="shared-data"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Search and Select:"),
                                    dbc.Button(
                                        "Search",
                                        id="outliers-btn",
                                        n_clicks=0,
                                        style={"margin": "5px"},
                                    ),
                                    dag.AgGrid(
                                        id="outliers-table",
                                        columnDefs=[
                                            {
                                                "field": "Column",
                                                "headerName": "Column",
                                                "sortable": False,
                                                "filter": False,
                                            },
                                            # {'field': 'Row', 'headerName': 'Row', 'sortable': False, 'filter': False},
                                            {
                                                "field": "Value",
                                                "headerName": "Value",
                                                "sortable": False,
                                                "filter": False,
                                            },
                                        ],
                                        dashGridOptions={
                                            "rowSelection": "multiple",
                                            "animateRows": True,
                                            "pagination": False,
                                            "rowMultiSelectWithClick": False,
                                        },
                                        defaultColDef={
                                            "flex": 1,
                                            "minWidth": 150,
                                            "minHeight": 350,
                                            "autoHeight": True,
                                            "wrapText": True,
                                        },
                                    ),
                                ]
                            ),
                            className="shadow-sm",
                            style=dash_styles.card_style_P1,
                        ),
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Select and Delete:"),
                                    dbc.Button(
                                        "Delete row",
                                        id="row-del-btn",
                                        n_clicks=0,
                                        style={"margin": "5px"},
                                    ),
                                    dag.AgGrid(
                                        id="outliers-target-table",
                                        columnDefs=[
                                            {
                                                "field": "Row",
                                                "headerName": "Row",
                                                "sortable": False,
                                                "filter": False,
                                            },
                                            {
                                                "field": "Value",
                                                "headerName": "Value",
                                                "sortable": False,
                                                "filter": False,
                                            },
                                        ],
                                        dashGridOptions={
                                            "rowSelection": "multiple",
                                            "animateRows": True,
                                            "pagination": False,
                                            "rowMultiSelectWithClick": True,
                                        },
                                        defaultColDef={
                                            "flex": 1,
                                            "minWidth": 150,
                                            "minHeight": 350,
                                            "autoHeight": True,
                                            "wrapText": True,
                                        },
                                        selectedRows=[None],
                                    ),
                                ]
                            ),
                            className="shadow-sm",
                            style=dash_styles.card_style_P1,
                        ),
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Selected Column Distribution"),
                                    dcc.Graph(
                                        id="outlier-boxplot",
                                        figure={},
                                        style={"height": "auto"},
                                        config={
                                            "modeBarButtonsToRemove": [
                                                "lasso2d",
                                                "select2d",
                                            ]
                                        },
                                    ),
                                ]
                            ),
                            className="shadow-sm",
                            style=dash_styles.card_style_P1,
                        ),
                    ),
                ]
            ),
        ],
        style={"marginBottom": "50px"},
    ),
)
#
################################ IMPUTATION
#
tab3 = (
    html.Div(
        [
            dcc.Store(id="shared-data"),
            dcc.Store(id="shared-data-imputate"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody(
                                                [
                                                    html.Div(
                                                        html.H4(
                                                            "Imputation Settings",
                                                            style={
                                                                "marginBottom": "25px"
                                                            },
                                                        ),
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            html.H4("CPU Cores:"),
                                                            dcc.Dropdown(
                                                                id="cpu-dropdown",
                                                                options=[
                                                                    {
                                                                        "label": str(i),
                                                                        "value": i,
                                                                    }
                                                                    for i in range(
                                                                        cpu_cores
                                                                    )
                                                                ],
                                                                value=cpu_cores - 1,
                                                                clearable=False,
                                                                style=dash_styles.dropdown_style,
                                                            ),
                                                        ],
                                                        className="mb-3",  # Adds margin below for spacing
                                                    ),
                                                    dbc.InputGroup(
                                                        [
                                                            html.H4("Fast run:"),
                                                            dcc.Dropdown(
                                                                id="imputation-fastrun-dropdown",
                                                                options=[
                                                                    "Yes",
                                                                    "No",
                                                                ],  # 'optimize_for_deployment',
                                                                placeholder="Select",
                                                                clearable=False,
                                                                style=dash_styles.dropdown_style,
                                                            ),
                                                            dbc.Col(
                                                                html.Span(
                                                                    "❓",
                                                                    id="fast-tooltip-icon",
                                                                    style={
                                                                        "cursor": "pointer",
                                                                        "fontSize": "20px",
                                                                        "marginLeft": "5px",
                                                                    },
                                                                ),
                                                                width="auto",
                                                            ),
                                                            dbc.Tooltip(
                                                                "If Yes, then the imputation model will only run and choose between KNN and Randomforrest. If No, then the full range of models will be applied.",
                                                                # Tooltip text
                                                                target="fast-tooltip-icon",  # Matches the icon ID
                                                                placement="right",  # Position of the tooltip
                                                            ),
                                                        ],
                                                        className="mb-3",  # Adds margin below for spacing
                                                    ),
                                                    html.Div(
                                                        [
                                                            dbc.Button(
                                                                "Perform imputation",
                                                                id="imputate-btn",
                                                                n_clicks=0,
                                                                style={"margin": "5px"},
                                                            ),
                                                        ]
                                                    ),
                                                    html.Div(
                                                        [
                                                            dcc.Loading(
                                                                id="loading-imputation",
                                                                children=html.H4(
                                                                    "",
                                                                    id="imputation-text",
                                                                    style={
                                                                        "margin": "10px"
                                                                    },
                                                                ),
                                                            )
                                                        ],
                                                        style={"marginTop": "100px"},
                                                    ),
                                                ]
                                            ),
                                            className="shadow-sm",
                                            style=dash_styles.card_style_P1,
                                        ),
                                    ),
                                    dbc.Col(
                                        dbc.Card(
                                            dbc.CardBody(
                                                [
                                                    dbc.InputGroup(
                                                        [
                                                            html.H4(
                                                                "Replace null value"
                                                            ),
                                                            dbc.Col(
                                                                html.Span(
                                                                    "❓",
                                                                    id="null-replace-tooltip-icon",
                                                                    style={
                                                                        "cursor": "pointer",
                                                                        "fontSize": "20px",
                                                                        "marginLeft": "5px",
                                                                    },
                                                                ),
                                                                width="auto",
                                                            ),
                                                            dbc.Tooltip(
                                                                children=[
                                                                    html.P(
                                                                        "If the nullvalue is a placeholder in your csv for 'no value' etc. you might want to change it to an actual value so that it is not imputated."
                                                                    ),  # Tooltip text
                                                                    html.P(
                                                                        "Simply select the column that this applies for, enter a placeholder value, such as 'no value', then press replace."
                                                                    ),
                                                                ],  # Tooltip text
                                                                target="null-replace-tooltip-icon",
                                                                # Matches the icon ID
                                                                placement="right",  # Position of the tooltip
                                                            ),
                                                        ],
                                                        className="mb-3",  # Adds margin below for spacing
                                                    ),
                                                    html.H4("Choose Column"),
                                                    dcc.Dropdown(
                                                        id="null-col-dropdown",
                                                        options=[],
                                                        style=dash_styles.dropdown_style,
                                                    ),
                                                    dbc.Input(
                                                        id="null-replace-input",
                                                        placeholder="Input replacement.",
                                                        style={
                                                            "width": "200px",
                                                            "marginTop": "5px",
                                                            "marginBottom": "5px",
                                                        },
                                                    ),
                                                    dbc.Button(
                                                        "Replace",
                                                        id="replace-nan-btn",
                                                        n_clicks=0,
                                                        style={"marginTop": "5px"},
                                                    ),
                                                ]
                                            ),
                                            className="shadow-sm",
                                            style=dash_styles.card_style_P1,
                                        ),
                                    ),
                                ]
                            ),
                        ]
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [dcc.Graph(id="missing-values-bar")],
                            ),
                            className="shadow-sm",
                            style=dash_styles.card_style_P1,
                        ),
                    ),
                ]
            ),
        ],
        style={"marginBottom": "50px"},
    ),
)
#
######################### END IMPUTATION
#
tab4 = html.Div(
    [
        dcc.Store(id="ordinal-data-store"),
        dcc.Store(id="shared-data"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H3("Convert to numerical and normalize"),
                                dcc.RadioItems(
                                    ["One-Hot Only", "One-Hot + Ordinal"],
                                    "One-Hot Only",
                                    inline=False,
                                    id="normalize-choice-radio",
                                ),
                                dbc.Button(
                                    "Transform to numerical and normalize",
                                    id="transform-numerical-btn",
                                    n_clicks=0,
                                    style={"marginTop": "75px"},
                                ),
                                dcc.Loading(
                                    id="loading-encoding",
                                    children=html.H3("", id="encoding-status"),
                                ),
                            ]
                        ),
                        className="shadow-sm",
                        style=dash_styles.card_style_P1,
                    ),
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dbc.InputGroup(
                                    [
                                        html.H4("Select columns for ordinal encoding"),
                                        html.Span(
                                            "❓",
                                            id="select-ordinal-icon",
                                            style={
                                                "cursor": "pointer",
                                                "fontSize": "20px",
                                                "marginLeft": "5px",
                                            },
                                        ),
                                        dbc.Tooltip(
                                            children=[
                                                html.P(
                                                    "Start with selecting the column you consider to have ordinal values. Then press SELECT COLUMNS button."
                                                ),  # Tooltip text
                                                html.P(
                                                    "Enter weights on all values in the columns, 1 is lowest importance, 2 is second lowest and so on."
                                                ),
                                                html.P(
                                                    "When you have assigned numbers to all values in the columns you have choosen, press CONFIRM RANKING. Then press TRANSFORM TO NUMERICAL AND NORMALIZE."
                                                ),
                                            ],  # Tooltip text
                                            target="select-ordinal-icon",  # Matches the icon ID
                                            placement="right",  # Position of the tooltip
                                        ),
                                    ]
                                ),
                                dcc.Dropdown(
                                    id="column-selector",
                                    options=[],
                                    multi=True,
                                    style=dash_styles.dropdown_style,
                                ),
                                dbc.Button(
                                    "Select columns",
                                    id="select-ordinal-btn",
                                    n_clicks=0,
                                    style={"marginTop": "10px"},
                                ),
                                html.Div(id="ordinal-ranking-container"),
                                dbc.Button(
                                    "Confirm ranking",
                                    id="confirm-ordinal-btn",
                                    n_clicks=0,
                                    style={"marginTop": "10px"},
                                ),
                                html.H3(children="", id="confirm-ordinal-text"),
                            ],
                            id="second-normalize-column",
                        ),
                    ),
                    className="shadow-sm",
                    style=dash_styles.card_style_P1,
                ),
            ]
        ),
    ]
)

tab5 = dbc.Card(
    dbc.CardBody(
        [
            html.Div(
                [
                    dbc.Button("Save CSV", id="btn_csv", style={"margin": "5px"}),
                    dcc.Download(id="download-dataframe-csv"),
                    dcc.Link(
                        dbc.Button("Go to Data Processing", color="primary"),
                        href="/processing",
                    ),
                ]
            ),
        ]
    ),
    className="shadow-sm",
    style=dash_styles.card_style_P1,
)


layout = (
    html.Div(
        [
            dcc.Store(id="filename-data", storage_type="memory"),
            html.Header(
                className="dataset-header",
                children=[
                    html.P(
                        "Upload your CSV below to start.", className="header-subtitle"
                    ),
                    # Upload Section
                    html.Div(
                        className="dataset-upload-section",
                        children=[
                            dcc.Upload(
                                id="upload-data",
                                children=html.Div(
                                    ["Drag and Drop or ", html.A("Select CSV File")]
                                ),
                                style={
                                    "width": "100%",
                                    "height": "100px",
                                    "lineHeight": "100px",
                                    "borderWidth": "2px",
                                    "borderStyle": "dashed",
                                    "borderRadius": "5px",
                                    "textAlign": "center",
                                    "margin": "10px auto",
                                },
                                multiple=False,
                                accept=".csv",  # CSV only
                            ),
                            html.Div(id="output-data-upload"),
                        ],
                    ),
                ],
            ),
            dcc.Tabs(
                id="tabs-example",
                value="tab-1",
                children=[
                    dcc.Tab(label="HealthCheck", value="tab-1"),
                    dcc.Tab(label="Outliers", value="tab-2"),
                    dcc.Tab(label="Imputation", value="tab-3"),
                    dcc.Tab(label="Encode and Normalize", value="tab-4"),
                    dcc.Tab(label="Save and Continue", value="tab-5"),
                ],
            ),
            html.Div(
                id="tabs-content",
                children=[
                    html.Div(tab1, id="content-tab1", style={"display": "block"}),
                    html.Div(tab2, id="content-tab2", style={"display": "none"}),
                    html.Div(tab3, id="content-tab3", style={"display": "none"}),
                    html.Div(tab4, id="content-tab4", style={"display": "none"}),
                    html.Div(tab5, id="content-tab5", style={"display": "none"}),
                ],
            ),
        ]
    ),
)
