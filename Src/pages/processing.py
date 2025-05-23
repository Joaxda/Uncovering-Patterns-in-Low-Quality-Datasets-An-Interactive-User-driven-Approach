# processing/data_processing.py
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
import assets.dash_styles as dash_styles

from scripts.processing.clustering import clustering_cpu_only
from scripts.processing.dr import dr_cpu_only

# Dont remove this import even if it looks unused.
import callbacks.processingCallbacks

import diskcache
cache = diskcache.Cache("/cache")


def update_output_error_handler(err):
    print("Dash error: ", err)


dash.register_page(
    __name__,
    path="/processing",
    name="data processing",
    on_error=update_output_error_handler,
)
dr_options = [
    {"label": v["description"], "value": k}
    for k, v in dr_cpu_only.DR_METHODS_CPU.items()
]
clustering_options = [
    {"label": v["description"], "value": k}
    for k, v in clustering_cpu_only.CLUSTERS_CPU.items()
]

empty_fig = {
    "data": [
        {
            "x": [],
            "y": [],
            "mode": "markers",
            "marker": {"opacity": 0},  # invisible trace
        }
    ],
    "layout": {
        "template": "plotly_dark",  # this applies the dark theme
        "xaxis": {"visible": False, "showgrid": False, "zeroline": False},
        "yaxis": {"visible": False, "showgrid": False, "zeroline": False},
        "annotations": [
            {
                "text": "No data available",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 10, "color": "gray"},
            }
        ],
    },
}

# df_something = "temp/best_fnox_test.csv"
layout = html.Div(
    [
        # dcc.Store(id="filename-data"),
        dcc.Store(
            id="filename-data",
            # data=df_something,
            storage_type="memory",
        ),
        dbc.Container(
            fluid=True,
            children=[  # Page Header & Global Stores
                html.H2("Processing Page", style={"fontSize": "20px"}),
                html.P(
                    "Select target variable, DR technique, and clustering.",
                    style={"fontSize": "14px"},
                ),
                # Global Stores
                dcc.Store(
                    id="selected-columns-feature-relevance", storage_type="memory"
                ),
                dcc.Store(id="selected-columns", storage_type="memory"),
                dcc.Store(id="reduced-data", storage_type="memory"),
                dcc.Store(id="clustering-data", storage_type="memory"),
                dcc.Store(id="dr-plot-axis", storage_type="memory"),
                dcc.Store(id="cluster-selected-indices", storage_type="memory"),
                # -------------------- TOP ROW --------------------
                dbc.Row(
                    [  # TOP LEFT: Feature Selection Stuff wrapped in a Card
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [  # Row for Target Variable & Feature Relevance Method Selection
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.H4(
                                                        "Feature Selection",
                                                        style={"fontSize": "14px"},
                                                    ),
                                                    width=12,
                                                )
                                            ],
                                            className="mb-4",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Select target variable:",
                                                            style={
                                                                "marginBottom": "5px",
                                                                "fontSize": "12px",
                                                            },
                                                        ),
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    children=[
                                                                        html.Div(
                                                                            style={
                                                                                "display": "flex",
                                                                                "alignItems": "center",
                                                                            },  # Align items horizontally
                                                                            children=[
                                                                                dcc.Dropdown(
                                                                                    id="target-variable-dropdown",
                                                                                    style=dash_styles.dropdown_style,
                                                                                    value="filename-data",
                                                                                    clearable=False,
                                                                                    searchable=True,
                                                                                ),
                                                                                html.Span(
                                                                                    "❓",
                                                                                    id="target-variable-tooltip",
                                                                                    style={
                                                                                        "cursor": "pointer",
                                                                                        "fontSize": "16px",
                                                                                        "color": "#6c757d",
                                                                                    },
                                                                                ),
                                                                                dbc.Tooltip(
                                                                                    "Choose the column that represents the target variable for analysis. "
                                                                                    "The target variable is the variable you want to learn to predict from the other features in the dataset.",
                                                                                    target="target-variable-tooltip",
                                                                                    placement="right",
                                                                                ),
                                                                            ],
                                                                        ),
                                                                    ],
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label(
                                                            "Select feature relevance method:",
                                                            style={
                                                                "marginBottom": "5px",
                                                                "fontSize": "12px",
                                                            },
                                                        ),
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    children=[
                                                                        html.Div(
                                                                            style={
                                                                                "display": "flex",
                                                                                "alignItems": "center",
                                                                            },  # Align items horizontally
                                                                            children=[
                                                                                dcc.Dropdown(
                                                                                    id="feature-relevance-method-dropdown",
                                                                                    options=[
                                                                                        {
                                                                                            "label": "Correlation Analysis",
                                                                                            "value": "Correlation Analysis",
                                                                                        },
                                                                                        {
                                                                                            "label": "Gradient Boosting",
                                                                                            "value": "Gradient Boosting",
                                                                                        },
                                                                                        {
                                                                                            "label": "Mutual Information",
                                                                                            "value": "Mutual Information",
                                                                                        },
                                                                                    ],
                                                                                    placeholder="Select a Feature Relevance Method",
                                                                                    value="Correlation Analysis",
                                                                                    multi=False,
                                                                                    style=dash_styles.dropdown_style,
                                                                                    clearable=False,
                                                                                ),
                                                                                html.Span(
                                                                                    "❓",
                                                                                    id="feature-relevance-method-dropdown-tooltip",
                                                                                    style=dash_styles.tooltip_style,
                                                                                ),
                                                                                dbc.Tooltip(
                                                                                    [
                                                                                        html.P(
                                                                                            "The feature relevance method identifies the most important features in the dataset.",
                                                                                            style={
                                                                                                "fontSize": "12px"
                                                                                            },
                                                                                        ),
                                                                                        html.P(
                                                                                            " - Correlation Analysis: Measures the linear relationship between features and the target.",
                                                                                            style={
                                                                                                "fontSize": "12px"
                                                                                            },
                                                                                        ),
                                                                                        html.P(
                                                                                            " - Gradient Boosting: Uses a model-based approach to determine feature importance.",
                                                                                            style={
                                                                                                "fontSize": "12px"
                                                                                            },
                                                                                        ),
                                                                                        html.P(
                                                                                            " - Mutual Information: Captures nonlinear relationships between variables.",
                                                                                            style={
                                                                                                "fontSize": "12px"
                                                                                            },
                                                                                        ),
                                                                                    ],
                                                                                    target="feature-relevance-method-dropdown-tooltip",
                                                                                    placement="right",
                                                                                ),
                                                                            ],
                                                                        ),
                                                                    ],
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ],
                                            className="mb-3",
                                        ),
                                        # Row for Feature Relevance Graph and MultiSelect
                                        dbc.Row(
                                            dbc.Col(
                                                [
                                                    # Feature Relevance Graph Title + Tooltip
                                                    dbc.Row(
                                                        dbc.Col(
                                                            [
                                                                html.Label(
                                                                    "Feature Relevance Graph",
                                                                    style={
                                                                        "fontWeight": "bold",
                                                                        "fontSize": "14px",
                                                                    },
                                                                ),
                                                                html.Span(
                                                                    "❓",
                                                                    id="feature-relevance-graph-tooltip",
                                                                    style=dash_styles.tooltip_style,
                                                                ),
                                                            ],
                                                            style={
                                                                "display": "flex",
                                                                "alignItems": "center",
                                                            },
                                                        ),
                                                        className="mb-2",
                                                    ),
                                                    # Feature Relevance Graph with Loading Indicator
                                                    dcc.Loading(
                                                        id="loading-feature-relevance-graph",
                                                        type="cube",
                                                        children=dcc.Graph(
                                                            id="feature-relevance-graph",
                                                            figure=empty_fig,
                                                        ),
                                                    ),
                                                    # Tooltip for the Feature Relevance Graph
                                                    dbc.Tooltip(
                                                        [
                                                            html.P(
                                                                "This graph visualizes the relevance of each feature in the dataset.",
                                                                style={
                                                                    "fontSize": "12px"
                                                                },
                                                            ),
                                                            html.P(
                                                                "- Higher values indicate stronger relevance to the target variable.",
                                                                style={
                                                                    "fontSize": "12px"
                                                                },
                                                            ),
                                                            html.P(
                                                                "- Used for feature selection and model optimization.",
                                                                style={
                                                                    "fontSize": "12px"
                                                                },
                                                            ),
                                                            html.P(
                                                                "- You need to select at least **3 features**.",
                                                                style={
                                                                    "fontSize": "12px"
                                                                },
                                                            ),
                                                            html.P(
                                                                "The selected features will be used for Dimensionality Reduction, Clustering, and EBM.",
                                                                style={
                                                                    "fontSize": "12px"
                                                                },
                                                            ),
                                                        ],
                                                        target="feature-relevance-graph-tooltip",
                                                        placement="right",
                                                        style={
                                                            "fontSize": "12px",
                                                            "borderRadius": "8px",
                                                            "padding": "8px",
                                                        },
                                                    ),
                                                    # Feature Selection Dropdown
                                                    html.Label(
                                                        "Select Features:",
                                                        style={
                                                            "fontWeight": "bold",
                                                            "fontSize": "12px",
                                                            "marginTop": "10px",
                                                        },
                                                    ),
                                                    html.Div(
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    html.Div(
                                                                        dcc.Dropdown(
                                                                            id="feature-multi-select",
                                                                            options=[],  # Populated dynamically
                                                                            multi=True,
                                                                            placeholder="Select at least 3 features",
                                                                            searchable=True,
                                                                            clearable=True,
                                                                            maxHeight=180,
                                                                            className="mb-2",
                                                                            style={
                                                                                "maxWidth": "100%",
                                                                                "maxHeight": 180,
                                                                                "width": "100%",
                                                                            },
                                                                        ),
                                                                    ),
                                                                    width=None,
                                                                    style={
                                                                        "padding": "0px",
                                                                        "flex": "0 0 90%",
                                                                        "maxWidth": "99%",
                                                                        "width": "90%",
                                                                    },
                                                                ),
                                                            ],
                                                            align="center",
                                                            className="g-0",
                                                            style={
                                                                "width": "100%",
                                                                "display": "flex",
                                                                "flexWrap": "nowrap",
                                                            },
                                                        ),
                                                        style={
                                                            "width": "100%",
                                                            "position": "relative",
                                                        },
                                                    ),
                                                ],
                                                width=12,
                                            ),
                                            className="mb-4",
                                        ),
                                    ]
                                ),
                                className="shadow-sm",
                                style=dash_styles.card_style_P2,
                            ),
                            width=6,
                        ),
                        # TOP RIGHT: Dimensionality Reduction (DR) Stuff wrapped in a Card
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.H4(
                                                        "Dimensionality Reduction",
                                                        style={"fontSize": "14px"},
                                                    ),
                                                    width=12,
                                                )
                                            ],
                                            className="mb-4",
                                        ),
                                        # ------------------ NEW: Single Row for Dropdown + Params + Tooltip ------------------ #
                                        dbc.Row(
                                            [  # Left column (Dropdown ~30%)
                                                dbc.Col(
                                                    children=[
                                                        html.Div(
                                                            style={
                                                                "display": "flex",
                                                                "alignItems": "center",
                                                            },  # Align items horizontally
                                                            children=[
                                                                dcc.Dropdown(
                                                                    id="dr-methods-dropdown",
                                                                    options=dr_options,
                                                                    multi=False,
                                                                    placeholder="Select a DR method",
                                                                    style={
                                                                        **dash_styles.dropdown_style,
                                                                    },  # Make dropdown take available space
                                                                ),
                                                                html.Span(
                                                                    "❓",
                                                                    id="dr-method-dropdown-tooltip",
                                                                    style=dash_styles.tooltip_style,
                                                                ),
                                                                dbc.Tooltip(
                                                                    "Choose a DR method to reduce the dimensionality of the dataset. Then you will choose the parameters for the selected DR method.",
                                                                    target="dr-method-dropdown-tooltip",
                                                                    placement="right",
                                                                ),
                                                            ],
                                                        ),
                                                        dbc.Button(
                                                            "Compute DR",
                                                            id="compute-dr-btn",
                                                            color="primary",
                                                            size="sm",
                                                            style={
                                                                "marginTop": "10px"
                                                            },  # Add spacing below the dropdown
                                                        ),
                                                    ]
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            id="dr-params-container",
                                                            style={
                                                                "marginBottom": "0.25rem"
                                                            },
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        # ------------------ Remaining Rows for RadioItems, Axes, and Compute Button ------------------ #
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.RadioItems(
                                                        id="dr-dimension",
                                                        inputStyle=dash_styles.radio_input_style,
                                                        labelStyle=dash_styles.radio_label_style,
                                                    ),
                                                ),
                                            ],
                                            style={"align": "right"},
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.Div(
                                                        dcc.Loading(
                                                            id="loading-dr-plot",
                                                            type="cube",
                                                            children=dcc.Graph(
                                                                id="dr-plot",
                                                                figure=empty_fig,
                                                                clear_on_unhover=True,
                                                            ),
                                                        ),
                                                        id="dr-plot-container",
                                                    ),
                                                )
                                            ],
                                            className="mb-4",
                                        ),
                                    ]
                                ),
                                className="shadow-sm",
                                style=dash_styles.card_style_P2,
                            ),
                        ),
                    ]
                ),
                # -------------------- BOTTOM ROW --------------------
                dbc.Row(
                    [  # BOTTOM LEFT: Feature Importance / Explainable Boosting Machine (EBM) wrapped in a Card
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.H4(
                                                        "Explainable Boosting Machine ",
                                                        style={"fontSize": "14px"},
                                                    ),
                                                    width=12,
                                                )
                                            ],
                                            className="mb-4",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Loading(
                                                        id="loading-cluster-to-use-dropdown",
                                                        type="cube",
                                                        children=dcc.Dropdown(
                                                            id="cluster-to-use",
                                                            placeholder="Select the cluster of interest",
                                                            style={
                                                                **dash_styles.dropdown_style,
                                                                "marginBottom": "1rem",
                                                                "visibility": "hidden",
                                                                "width": "200px",
                                                            },
                                                        ),
                                                    ),
                                                )
                                            ],
                                            className="mb-4",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.Button(
                                                        "Compute EBM",
                                                        id="compute-em-btn",
                                                        color="primary",
                                                        size="sm",
                                                    ),
                                                    className="d-flex justify-content-center",
                                                    width=12,
                                                )
                                            ],
                                            className="mb-4",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Loading(
                                                        id="loading-ebm-graph",
                                                        type="cube",
                                                        children=dcc.Graph(
                                                            id="ebm-graph", figure={}
                                                        ),
                                                    ),
                                                    width=12,
                                                )
                                            ],
                                            className="mb-4",
                                        ),
                                    ]
                                ),
                                id="ebm_card",
                                className="shadow-sm",
                                style=dash_styles.card_style_P2_hidden,
                            ),
                            width=6,
                        ),
                        # BOTTOM RIGHT: Clustering Stuff wrapped in a Card
                        dbc.Col(
                            dbc.Card(
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    html.H4(
                                                        "Clustering",
                                                        style={"fontSize": "14px"},
                                                    ),
                                                    width=12,
                                                )
                                            ],
                                            className="mb-4",
                                        ),
                                        dbc.Row(
                                            [  # Left column (Dropdown ~30%)
                                                dbc.Col(
                                                    [
                                                        html.Div(
                                                            style={
                                                                "display": "flex",
                                                                "alignItems": "left",
                                                            },
                                                            children=[
                                                                dcc.Dropdown(
                                                                    id="clustering-methods-dropdown",
                                                                    options=clustering_options,
                                                                    multi=False,
                                                                    placeholder="Select a clustering technique",
                                                                    style={
                                                                        **dash_styles.dropdown_style,
                                                                        "marginBottom": "0.25rem",
                                                                    },
                                                                ),
                                                                html.Span(
                                                                    "❓",
                                                                    id="clustering-method-tooltip",
                                                                    style=dash_styles.tooltip_style,
                                                                ),
                                                                dbc.Tooltip(
                                                                    "Choose a clustering method to group your data. Then you will choose the parameters for the selected clustering method.",
                                                                    target="clustering-method-tooltip",
                                                                    placement="right",
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                dbc.Col(
                                                    html.Div(
                                                        id="clustering-params-container",
                                                    ),
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    children=[
                                                        html.Div(
                                                            style={
                                                                "display": "flex",
                                                                "alignItems": "center",
                                                            },
                                                            children=[
                                                                dcc.RadioItems(
                                                                    id="clustering-param-mode",
                                                                    options=[
                                                                        {
                                                                            "label": "Manual Parameter Input ",
                                                                            "value": "manual",
                                                                        },
                                                                        {
                                                                            "label": "Auto-tune parameters",
                                                                            "value": "auto",
                                                                        },
                                                                    ],
                                                                    value="manual",
                                                                    inline=True,
                                                                    labelStyle=dash_styles.radio_input_style,
                                                                    inputStyle=dash_styles.radio_label_style,
                                                                ),
                                                                html.Span(
                                                                    "❓",
                                                                    id="clustering-param-mode-tooltip",
                                                                    style=dash_styles.tooltip_style,
                                                                ),
                                                                dbc.Tooltip(
                                                                    [
                                                                        html.P(
                                                                            "Choose how clustering parameters are selected.",
                                                                            style={
                                                                                "fontSize": "12px"
                                                                            },
                                                                        ),
                                                                        html.P(
                                                                            "- Manual: You define the parameters.",
                                                                            style={
                                                                                "fontSize": "12px"
                                                                            },
                                                                        ),
                                                                        html.P(
                                                                            "- Auto: The system optimizes parameters automatically. (This will take longer)",
                                                                            style={
                                                                                "fontSize": "12px"
                                                                            },
                                                                        ),
                                                                    ],
                                                                    target="clustering-param-mode-tooltip",
                                                                    placement="right",
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    children=[
                                                        html.Div(
                                                            style={
                                                                "display": "flex",
                                                                "alignItems": "center",
                                                            },
                                                            children=[
                                                                dcc.Checklist(
                                                                    id="use-whole-data-check",
                                                                    options=[
                                                                        {
                                                                            "label": "Run on the complete data without DR.",
                                                                            "value": "whole",
                                                                        }
                                                                    ],
                                                                    value=["whole"],
                                                                    inline=True,
                                                                    inputStyle={
                                                                        "marginRight": "5px"
                                                                    },
                                                                    labelStyle={
                                                                        "fontSize": "12px"
                                                                    },
                                                                ),
                                                                html.Span(
                                                                    "❓",
                                                                    id="use-whole-data-tooltip",
                                                                    style=dash_styles.tooltip_style,
                                                                ),
                                                                dbc.Tooltip(
                                                                    [
                                                                        html.P(
                                                                            "Decide whether to apply clustering to the entire dataset or the dimensionality reduction data (Checked = original data, Unchecked = DR data)",
                                                                            style={
                                                                                "fontSize": "12px"
                                                                            },
                                                                        ),
                                                                        html.P(
                                                                            "Checking this option will use the columns selected in the feature relevance graph.",
                                                                            style={
                                                                                "fontSize": "12px"
                                                                            },
                                                                        ),
                                                                        html.P(
                                                                            "Unchecking this option may speed up processing for large datasets.",
                                                                            style={
                                                                                "fontSize": "12px"
                                                                            },
                                                                        ),
                                                                    ],
                                                                    target="use-whole-data-tooltip",
                                                                    placement="right",
                                                                ),
                                                            ],
                                                        ),
                                                        dbc.Row(
                                                            [
                                                                dbc.Col(
                                                                    dbc.Button(
                                                                        "Compute Clustering",
                                                                        id="compute-clustering-btn",
                                                                        color="primary",
                                                                        size="sm",
                                                                    ),
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        dbc.Row(
                                            [
                                                html.Div(
                                                    style={
                                                        "display": "flex",
                                                        "alignItems": "center",
                                                        "marginTop": "0.25rem",
                                                    },
                                                    children=[
                                                        dbc.Col(
                                                            dcc.RadioItems(
                                                                id="clustering-dimension",
                                                                inputStyle=dash_styles.radio_input_style,
                                                                labelStyle=dash_styles.radio_label_style,
                                                            ),
                                                        ),
                                                        dbc.Row(
                                                            [
                                                                # X Column
                                                                dbc.Col(
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Col(
                                                                                html.Span(
                                                                                    "x:"
                                                                                )
                                                                            ),
                                                                            dbc.Col(
                                                                                dcc.Dropdown(
                                                                                    id="clustering-x-dropdown",
                                                                                    placeholder="Select X axis",
                                                                                    style={
                                                                                        "fontSize": "14px",
                                                                                        "minHeight": "20px",
                                                                                        "width": "150px",
                                                                                        # Increased width here
                                                                                    },
                                                                                )
                                                                            ),
                                                                        ],
                                                                        align="center",
                                                                    ),
                                                                    id="clustering-x-col",
                                                                    style={
                                                                        "display": "none"
                                                                    },
                                                                ),
                                                                # Y Column
                                                                dbc.Col(
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Col(
                                                                                html.Span(
                                                                                    "y:"
                                                                                )
                                                                            ),
                                                                            dbc.Col(
                                                                                dcc.Dropdown(
                                                                                    id="clustering-y-dropdown",
                                                                                    placeholder="Select Y axis",
                                                                                    style={
                                                                                        "fontSize": "14px",
                                                                                        "minHeight": "20px",
                                                                                        "width": "150px",
                                                                                        # Increased width here
                                                                                    },
                                                                                )
                                                                            ),
                                                                        ],
                                                                        align="center",
                                                                    ),
                                                                    id="clustering-y-col",
                                                                    style={
                                                                        "display": "none"
                                                                    },
                                                                ),
                                                                # Z Column
                                                                dbc.Col(
                                                                    dbc.Row(
                                                                        [
                                                                            dbc.Col(
                                                                                html.Span(
                                                                                    "z:"
                                                                                )
                                                                            ),
                                                                            dbc.Col(
                                                                                dcc.Dropdown(
                                                                                    id="clustering-z-dropdown",
                                                                                    placeholder="Select Z axis",
                                                                                    style={
                                                                                        "fontSize": "14px",
                                                                                        "minHeight": "20px",
                                                                                        "width": "150px",
                                                                                        # Increased width here
                                                                                    },
                                                                                )
                                                                            ),
                                                                        ],
                                                                        align="center",
                                                                    ),
                                                                    id="clustering-z-col",
                                                                    style={
                                                                        "display": "none"
                                                                    },
                                                                ),
                                                            ],
                                                            justify="between",
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                        dbc.Row(
                                            dbc.Col(
                                                html.Div(
                                                    dcc.Loading(
                                                        id="loading-clustering-plot",
                                                        type="cube",
                                                        children=dcc.Graph(
                                                            id="clustering-plot",
                                                            figure=empty_fig,
                                                            clear_on_unhover=False,
                                                            style={
                                                                "height": "100%",
                                                                "width": "100%",
                                                            },
                                                        ),
                                                    ),
                                                    id="clustering-plot-container",
                                                    style={
                                                        "flex": "1",
                                                        "minHeight": "0",
                                                    },
                                                    # Allow container to shrink properly
                                                ),
                                                width=12,
                                            ),
                                            className="mb-4",
                                            style={"flex": "1", "minHeight": "0"},
                                            # Ensure the row doesn't force extra height
                                        ),
                                    ]
                                ),
                                className="shadow-sm",
                                style=dash_styles.card_style_P2,
                            ),
                            width=6,
                        ),
                    ]
                ),
                # Hidden Stores for DR
                dcc.Store(id="dr-transformed-data", storage_type="memory"),
                dcc.Store(id="dr-selected-indices"),
            ],
        ),
    ]
)
