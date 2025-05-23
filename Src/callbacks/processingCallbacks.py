import dash
import dash_bootstrap_components as dbc
import numpy as np
from dash import html
from dash.dependencies import ALL
from dash.dependencies import Input, Output, State
import assets.dash_styles as dash_styles
import pandas as pd
import plotly.express as px

from pages.processing import update_output_error_handler
from scripts.processing.EBM import EBM
from scripts.processing.FeatureRelevance import FeatureRelevance

from scripts.processing.clustering import clustering_cpu_only
from scripts.processing.dr import dr_cpu_only
from scripts.processing.clustering.param_tuning import get_best_params

import diskcache
cache = diskcache.Cache("/cache")



#
#
# CALLBACKS FOR FEATURE RELEVANCE
#
#


@dash.callback(
    Output("target-variable-dropdown", "value"),
    Output("target-variable-dropdown", "options"),
    Input("filename-data", "data"),
    timeout=600000,
)
def update_variable_dropdown(clean_data):
    if clean_data == None:
        return dash.no_update
    return pd.read_csv(clean_data).columns[-1], [
        {"label": col, "value": col} for col in pd.read_csv(clean_data).columns
    ]


def shorten_labels(labels, max_len=15):
    """
    Takes a list of feature names and returns a new list of truncated labels,
    ensuring each truncated label is unique. If two labels collide, a suffix
    like (2), (3), etc. is appended to differentiate them.
    """
    used = set()
    shortened_list = []
    for label in labels:
        # Truncate if needed
        truncated = label[:max_len] + "..." if len(label) > max_len else label

        # If this truncated label is already used, append a suffix
        i = 2
        new_label = truncated
        while new_label in used:
            new_label = f"{truncated}({i})"
            i += 1

        used.add(new_label)
        shortened_list.append(new_label)
    return shortened_list


def style_figure(fig):
    """
    Applies consistent styling to any Plotly figure so that:
    - The plot area gets generous margins.
    - The colorbar (continuous legend) is made thinner.
    - Axis tick fonts are reduced.
    """
    fig.update_layout(margin=dict(l=120, r=20, t=60, b=60))
    fig.update_layout(coloraxis_colorbar=dict(thickness=15))
    fig.update_yaxes(tickfont=dict(size=10))
    fig.update_xaxes(tickfont=dict(size=10))
    return fig


@dash.callback(
    Output("feature-relevance-graph", "figure"),
    State("filename-data", "data"),
    Input("target-variable-dropdown", "value"),
    Input("feature-relevance-method-dropdown", "value"),
    timeout=600000,
)
def update_feature_relevance_graph(clean_data, target, method):
    # Read the dataset and instantiate your FeatureRelevance class.
    if clean_data == None:
        return dash.no_update
    df = pd.read_csv(clean_data)
    print(f"The length of the dataset is {len(df)}")
    fr = FeatureRelevance(dataset=df, targets=[target])
    method = method or "Correlation Analysis"

    if method == "Correlation Analysis":
        corr_matrix, target_corr = fr.correlation_analysis()
        target_corr = target_corr.reset_index().rename(
            columns={"index": "Feature", target: "Correlation"}
        )
        target_corr = target_corr.sort_values(by="Correlation", ascending=True)

        # Create a unique shortened label for each Feature
        features = target_corr["Feature"].tolist()
        short_labels = shorten_labels(features)
        target_corr["ShortFeature"] = short_labels

        fig = px.bar(
            target_corr,
            x="Correlation",
            y="ShortFeature",
            orientation="h",
            color="Correlation",
            color_continuous_scale="Viridis",
            labels={
                "Correlation": "Correlation Coefficient",
                "ShortFeature": "Feature",
            },
            title=f"Correlation Analysis with {target}",
            custom_data=["Feature"],
            template="plotly",
        )
        # Show full feature name on hover
        fig.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>Correlation: %{x}<extra></extra>"
        )
        # Force all short labels to appear on the y-axis
        fig.update_layout(
            yaxis=dict(
                automargin=True,
            )
        )

        fig.update_yaxes(tickfont=dict(size=8))
        return style_figure(fig)

    elif method == "Gradient Boosting":
        fi_df = fr.gradient_boosting(esti=100, plot=False)
        fi_df = fi_df.sort_values(by="Importance", key=lambda x: abs(x), ascending=True)

        # Unique short labels
        features = fi_df["Feature"].tolist()
        short_labels = shorten_labels(features)
        fi_df["ShortFeature"] = short_labels

        fig = px.bar(
            fi_df,
            x="Importance",
            y="ShortFeature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Viridis",
            labels={"Importance": "Feature Importance", "ShortFeature": "Feature"},
            title=f"Gradient Boosting Feature Importance for {target}",
            custom_data=["Feature"],
        )
        fig.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>Importance: %{x}<extra></extra>"
        )
        fig.update_layout(
            yaxis=dict(
                automargin=True,
            )
        )

        fig.update_yaxes(tickfont=dict(size=8))
        return style_figure(fig)

    elif method == "Mutual Information":
        mi = fr.mutual_information(plot=False)[target].sort_values(ascending=True)
        mi_df = pd.DataFrame({"Feature": mi.index, "MI": mi.values})

        # Unique short labels
        features = mi_df["Feature"].tolist()
        short_labels = shorten_labels(features)
        mi_df["ShortFeature"] = short_labels

        fig = px.bar(
            mi_df,
            x="MI",
            y="ShortFeature",
            orientation="h",
            color="MI",
            color_continuous_scale="Viridis",
            labels={"MI": "Mutual Information Score", "ShortFeature": "Feature"},
            title=f"Mutual Information for {target}",
            custom_data=["Feature"],
        )
        fig.update_traces(
            hovertemplate="<b>%{customdata[0]}</b><br>MI: %{x}<extra></extra>"
        )
        fig.update_layout(
            yaxis=dict(
                automargin=True,
            )
        )

        fig.update_yaxes(tickfont=dict(size=8))
        return style_figure(fig)

    else:
        return {}


@dash.callback(
    Output("selected-columns", "data"),
    Input("feature-relevance-graph", "selectedData"),
    # prevent_initial_call=True,
    timeout=600000,
)
def update_selected_features(selectedData):
    if selectedData is None or "points" not in selectedData:
        return []
    return [point.get("customdata")[0] for point in selectedData["points"]]


@dash.callback(
    [
        Output("feature-multi-select", "options"),
        Output("feature-multi-select", "value"),
        Output("selected-columns-feature-relevance", "data"),
    ],
    [
        Input("filename-data", "data"),
        Input("selected-columns", "data"),  # Box/lasso selection
        Input("feature-multi-select", "value"),
    ],  # Manual dropdown selection
    prevent_initial_call=False,
    timeout=600000,
)
def update_feature_dropdown_options(clean_data, selected_data, dropdown_selected):
    # Build dropdown options from the column names in the clean_data
    if clean_data == None:
        return dash.no_update
    column_names = pd.read_csv(clean_data).columns.tolist()
    options = [{"value": col, "label": col} for col in column_names]
    # Determine which input triggered the callback
    triggered_input = (
        dash.ctx.triggered[0]["prop_id"].split(".")[0] if dash.ctx.triggered else None
    )
    # If box selection was triggered and there is a selection, use it (overriding previous dropdown selection)
    if (
            triggered_input == "selected-columns"
            and selected_data
            and len(selected_data) > 0
    ):
        # Assuming box_selected is a list of dicts, extract feature keys from the first element
        new_selection = selected_data
    # Otherwise, if the dropdown triggered the callback, use its current value (or empty if None)
    elif triggered_input == "feature-multi-select" and dropdown_selected:
        new_selection = dropdown_selected
    else:
        new_selection = dropdown_selected or []

    return options, new_selection, new_selection


@dash.callback(
    Output("dr-params-container", "children"),
    Input("dr-methods-dropdown", "value"),
    State("filename-data", "data"),
    timeout=600000,
    # prevent_initial_call=True,
    # timeout=600000,
    # on_error=update_output_error_handler
)
def render_dr_params(selected_method, clean_data):
    # Exit early if no method is selected.
    print("Inside: render_dr_params")
    if clean_data == None:
        return dash.no_update
    if not selected_method:
        return []

    # Get method metadata from CPU or GPU implementations.
    meta = dr_cpu_only.DR_METHODS_CPU.get(selected_method)
    if not meta:
        return []

    # Use a smaller heading for the parameters title.
    title = html.H5(
        f"{selected_method} Parameters", style={"margin": "0", "fontSize": "10px"}
    )

    def create_component(p):
        param_name = p["name"]
        param_type = p["type"]
        comp_id = {
            "type": "dr-param-input",
            "method": selected_method,
            "param": param_name,
        }
        # Smaller label styling
        label = html.Label(
            param_name,
            style={"marginRight": "0.25rem", "minWidth": "50px", "fontSize": "10px"},
        )

        # Create the appropriate input based on parameter type, using smaller widths and font sizes.
        if param_type == "int":
            min_val = (
                p.get("min", 2) if param_name == "n_components" else p.get("min", 0)
            )
            if param_name == "degree":
                max_val = 10
            elif param_name == "n_neighbors":
                max_val = 200
            elif param_name == "n_components":
                max_val = 3
            elif param_name == "perplexity":
                max_val = 100
            elif param_name == "learning_rate":
                max_val = 1000
            else:
                print("No match with param_name = ", param_name)

            input_component = dbc.Input(
                id=comp_id,
                type="number",
                min=min_val,
                max=max_val,
                step=1,
                value=p["default"],
                style={"width": "80px", "fontSize": "10px", "padding": "2px"},
            )
        elif param_type == "float":
            input_component = dbc.Input(
                id=comp_id,
                type="number",
                min=p.get("min", 0.0),
                max=p.get("max", 9999.0),
                step=0.1,
                value=p["default"],
                style={"width": "80px", "fontSize": "10px", "padding": "2px"},
            )
        elif param_type == "bool":
            input_component = dbc.Checklist(
                id=comp_id,
                options=[{"label": "", "value": True}],
                value=[True] if p["default"] else [],
                inputStyle={"width": "15px", "height": "15px"},
                labelStyle={"fontSize": "10px"},
            )
        elif param_type == "list":
            input_component = dbc.Select(
                id=comp_id,
                options=[{"label": option, "value": option} for option in p["options"]],
                value=p["default"],
                style={"width": "80px", "fontSize": "10px", "padding": "2px"},
            )
        else:
            input_component = dbc.Input(
                id=comp_id,
                type="text",
                value=p.get("default", ""),
                style={"width": "80px", "fontSize": "10px", "padding": "2px"},
            )

        return html.Div(
            [label, input_component],
            style={
                "display": "flex",
                "alignItems": "center",
                "marginRight": "0.25rem",
                "marginBottom": "0.25rem",
            },
        )

    # Build all parameter components with reduced size.
    param_groups = [create_component(p) for p in meta["parameters"]]

    container = html.Div(
        param_groups,
        style={"display": "flex", "flexWrap": "wrap", "alignItems": "center"},
    )

    return html.Div(
        [title, container], style={"border": "1px solid #ccc", "padding": "0.5rem"}
    )


@dash.callback(
    [
        Output("dr-transformed-data", "data"),  # Store computed DR data
        Output("dr-dimension", "options"),
        Output("dr-dimension", "value"),
        Output("dr-dimension", "style"),
        Output("dr-plot", "figure"),
    ],
    [
        Input("compute-dr-btn", "n_clicks"),  # Trigger DR computation
        Input("dr-dimension", "value"),  # 2D/3D switch
        Input(
            "feature-relevance-graph", "clickData"
        ),  # click on the feature relevance graph
    ],
    [
        State("target-variable-dropdown", "value"),  # Target variable for coloring
        State("dr-methods-dropdown", "value"),  # DR method
        State(
            {"type": "dr-param-input", "method": ALL, "param": ALL}, "value"
        ),  # DR parameters
        State(
            {"type": "dr-param-input", "method": ALL, "param": ALL}, "id"
        ),  # DR parameter IDs
        State("selected-columns-feature-relevance", "data"),  # Selected features
        State("filename-data", "data"),  # Clean data
        State("dr-transformed-data", "data"),
    ],
    prevent_initial_call=True,
    timeout=600000,
    on_error=update_output_error_handler,
)
def merged_DR_callback(
        n_clicks,
        dimension,
        clicked_data,
        target_variable,
        selected_method,
        param_values,
        param_ids,
        feature_relevance_selected_data,
        clean_data,
        dr_data,
):
    print("Inside: merged_DR_callback")
    if clean_data is None:
        return dash.no_update

    # Define base styles
    hidden_style = {"marginBottom": "1rem", "visibility": "hidden"}
    visible_style = {"marginBottom": "1rem", "visibility": "visible"}

    # Initialize defaults for outputs
    radio_options = []
    radio_value = 2
    radio_style = hidden_style
    figure = {}

    # Get the triggered input's ID
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    # Read the clean data only once
    df_clean = pd.read_csv(clean_data)
    df_clean = df_clean.reset_index().rename(columns={"index": "orig_index"})

    # Process based on what triggered the callback:
    if triggered_id == "feature-relevance-graph":
        print("Feature relevance graph clicked")
        if clicked_data is None or dr_data is None:
            return dash.no_update

        # Get the feature selected from the click data
        selected_feature = clicked_data["points"][0]["customdata"][0]
        selected_feature_data = df_clean[selected_feature]

        # Compute marker sizes using a logarithmic scale
        min_size, max_size = 1, 7
        sdata = selected_feature_data - selected_feature_data.min() + 1e-9
        log_scaled = np.log1p(sdata)
        marker_sizes = min_size + (max_size - min_size) * (
                log_scaled - log_scaled.min()
        ) / (log_scaled.max() - log_scaled.min())

        # Convert dr_data to a DataFrame for further manipulation
        dr_df = pd.DataFrame(dr_data)
    elif triggered_id == "dr-dimension":
        # Ensure DR data exists
        if not dr_data or len(dr_data) == 0:
            raise dash.exceptions.PreventUpdate
        dr_df = pd.DataFrame(dr_data)
    elif triggered_id == "compute-dr-btn":
        print("Compute DR button clicked")
        if not selected_method or not clean_data:
            raise dash.exceptions.PreventUpdate

        # Read the clean data and preserve original index
        df_clean = pd.read_csv(clean_data)
        df_clean = df_clean.reset_index().rename(columns={"index": "orig_index"})

        # Use feature relevance selection if available, ensuring orig_index is kept
        if feature_relevance_selected_data:
            df_features = df_clean[feature_relevance_selected_data + ["orig_index"]]
        else:
            df_features = df_clean

        # Build DR parameters dictionary
        params = {
            pid["param"]: (val[0] if isinstance(val, list) else val)
            for val, pid in zip(param_values, param_ids)
        }
        try:
            print("Running DR method:", selected_method, params, df_features.shape)
            dr_result = dr_cpu_only.run_cpu_dr(selected_method, df_features, params)
        except Exception as e:
            print("Error running DR method:", e)
            raise dash.exceptions.PreventUpdate
        try:
            dr_data = dr_result.to_dict("records")
        except Exception as e:
            print("Error converting DR data:", e)
            raise dash.exceptions.PreventUpdate

        if not dr_data or len(dr_data) == 0:
            raise dash.exceptions.PreventUpdate

        dr_df = pd.DataFrame.from_dict(dr_data)
        # Ensure the DR result includes the original index
        if "orig_index" not in dr_df.columns:
            dr_df["orig_index"] = df_features["orig_index"].values

    else:
        raise dash.exceptions.PreventUpdate

    # Compute available dimensions and default axes from dr_df
    available_dims = list(dr_df.columns)
    default_x = available_dims[0]
    default_y = available_dims[1]
    default_z = available_dims[2] if len(available_dims) >= 3 else None

    # Set radio options and value based on the number of dimensions available
    if len(available_dims) >= 3:
        radio_options = [{"label": "2D", "value": 2}, {"label": "3D", "value": 3}]
        try:
            d = int(dimension)
        except (ValueError, TypeError):
            d = 2
        radio_value = d if d in [2, 3] else 2
        radio_style = visible_style
    else:
        radio_options = []
        radio_value = 2
        radio_style = visible_style

    # Build the figure
    if target_variable:
        # Get the color values from the clean data
        colors = df_clean[target_variable].tolist()

        # Different handling for feature relevance branch versus others:
        if triggered_id == "feature-relevance-graph":
            col_name = f"Size: {selected_feature}/Color: {target_variable}"
            colorbar_title = target_variable
            dr_df["marker_size"] = marker_sizes
            dr_df[col_name] = colors
        else:
            col_name = f"Color: {target_variable}"
            # In compute-dr-btn branch, shorten the title if needed
            if triggered_id == "compute-dr-btn":
                colorbar_title = (
                    target_variable
                    if len(target_variable) <= 10
                    else target_variable[:10] + "..."
                )
            else:
                colorbar_title = target_variable
            dr_df[col_name] = colors

        if radio_value == 2:
            if triggered_id == "feature-relevance-graph":
                fig = px.scatter(
                    dr_df,
                    x=default_x,
                    y=default_y,
                    size=marker_sizes,
                    custom_data=["orig_index"],
                    color=col_name,
                    color_continuous_scale="Viridis",
                    labels={default_x: default_x, default_y: default_y},
                    hover_data={"orig_index": True},
                )
                fig.update_layout(
                    plot_bgcolor='#cfcccc'
                )
                fig.update_traces(
                    marker_colorbar=dict(title=colorbar_title), marker_line_width=0
                )
            else:
                fig = px.scatter(
                    dr_df,
                    x=default_x,
                    y=default_y,
                    custom_data=["orig_index"],
                    color=col_name,
                    color_continuous_scale="Viridis",
                    labels={default_x: default_x, default_y: default_y},
                    hover_data={"orig_index": True},
                )
                fig.update_traces(
                    marker=dict(size=4),
                    marker_colorbar=dict(title=colorbar_title),
                    marker_line_width=0,
                )
            fig.update_layout(margin=dict(l=40, b=40, t=10, r=0), hovermode="closest", plot_bgcolor='#cfcccc')
        else:  # 3D case
            if triggered_id == "feature-relevance-graph":
                fig = px.scatter_3d(
                    dr_df,
                    x=default_x,
                    y=default_y,
                    z=default_z,
                    custom_data=["orig_index"],
                    size="marker_size",
                    color=col_name,
                    color_continuous_scale="Viridis",
                    labels={
                        default_x: default_x,
                        default_y: default_y,
                        default_z: default_z,
                    },
                    hover_data={"orig_index": True},
                )
                fig.update_layout(
                    plot_bgcolor='#cfcccc'
                )
                fig.update_traces(
                    marker_colorbar=dict(title=colorbar_title), marker_line_width=0
                )
            else:
                fig = px.scatter_3d(
                    dr_df,
                    x=default_x,
                    y=default_y,
                    z=default_z,
                    color=col_name,
                    custom_data=["orig_index"],
                    color_continuous_scale="Viridis",
                    labels={
                        default_x: default_x,
                        default_y: default_y,
                        default_z: default_z,
                    },
                    hover_data={"orig_index": True},
                )
                fig.update_traces(
                    marker=dict(size=4),
                    marker_colorbar=dict(title=colorbar_title),
                    marker_line_width=0,
                )
            fig.update_layout(
                margin=dict(l=40, b=40, t=10, r=0),
                hovermode="closest",
                scene=dict(aspectmode="cube"),
                plot_bgcolor='#cfcccc'
            )
        figure = fig
    else:
        # Fallback: create a basic figure if target_variable is not provided
        if radio_value == 2:
            figure = px.scatter(
                dr_df,
                x=default_x,
                y=default_y,
                labels={default_x: default_x, default_y: default_y},
            )
            figure.update_layout(
                plot_bgcolor='#cfcccc'
            )
        else:
            figure = px.scatter_3d(
                dr_df,
                x=default_x,
                y=default_y,
                z=default_z,
                labels={
                    default_x: default_x,
                    default_y: default_y,
                    default_z: default_z,
                },
            )
            figure.update_layout(
                plot_bgcolor='#cfcccc'
            )

    if triggered_id == "feature-relevance-graph":
        dr_df = dr_df.drop(columns=["marker_size"])
        dr_df = dr_df.drop(columns=[col_name])
    elif triggered_id == "compute-dr-btn":
        dr_df = dr_df.drop(columns=[col_name])

    # Always return the DR data as a list of dicts along with radio settings and the figure.
    return (
        dr_df.to_dict("records"),  # DR-transformed data
        radio_options,  # DR-dimension options
        radio_value,  # DR-dimension value
        radio_style,  # DR-dimension style
        figure,  # DR-plot figure
    )


@dash.callback(
    Output("dr-selected-indices", "data"),
    Input("dr-plot", "selectedData"),
    State("dr-transformed-data", "data"),
    prevent_initial_call=True,
    timeout=600000,
)
def store_dr_selection(selected_data, dr_data):
    print("Inside: store_dr_selection")
    # If no selection is made, return the indexes of the full dataset.
    if not selected_data or not selected_data.get("points"):
        df = pd.DataFrame(dr_data)
        return df["orig_index"].tolist()

    # Extract the original index from each selected point's customdata.
    orig_indices = [p["customdata"][0] for p in selected_data["points"]]
    return orig_indices


#
#
# CALLBACKS FOR Clustering
#
#


@dash.callback(
    Output("clustering-params-container", "children"),
    [
        Input("clustering-methods-dropdown", "value"),
        Input("clustering-param-mode", "value"),
    ],
    # prevent_initial_call=True,
    timeout=600000,
)
def render_clustering_params(selected_method, param_mode):
    if not selected_method:
        return []
    meta = clustering_cpu_only.CLUSTERS_CPU.get(selected_method)
    if not meta:
        return []

    # Smaller title styling
    title = html.H5(
        f"{selected_method} Parameters", style={"margin": "0", "fontSize": "10px"}
    )

    param_groups = []
    for p in meta["parameters"]:
        comp_id = {
            "type": "cluster-param-input",
            "method": selected_method,
            "param": p["name"],
        }

        # Smaller label styling
        label = html.Label(
            p["name"],
            style={"marginRight": "0.25rem", "minWidth": "50px", "fontSize": "10px"},
        )

        # Common style for small inputs
        input_style = {"width": "70px", "fontSize": "10px", "padding": "2px"}

        if p["type"] == "int":
            input_component = dbc.Input(
                id=comp_id,
                type="number",
                min=p.get("min", 0),
                max=p.get("max", 9999),
                step=1,
                value=p["default"],
                style=input_style,
            )
        elif p["type"] == "float":
            input_component = dbc.Input(
                id=comp_id,
                type="number",
                min=p.get("min", 0.0),
                max=p.get("max", 9999.0),
                step=0.1,
                value=p["default"],
                style=input_style,
            )
        elif p["type"] == "bool":
            # Smaller checklist styling
            input_component = dbc.Checklist(
                id=comp_id,
                options=[{"label": "", "value": True}],
                value=[True] if p["default"] else [],
                inputStyle={"width": "15px", "height": "15px"},
                labelStyle={"fontSize": "10px"},
            )
        elif p["type"] == "list":
            print("The options are", p)
            input_component = dbc.Select(
                id=comp_id,
                options=[{"label": option, "value": option} for option in p["options"]],
                value=(
                    p["default"][0] if isinstance(p["default"], list) else p["default"]
                ),
                style=input_style,
            )
        else:
            # Fallback: text input
            input_component = dbc.Input(
                id=comp_id, type="text", value=p.get("default", ""), style=input_style
            )

        group = html.Div(
            [label, input_component],
            style={
                "display": "flex",
                "alignItems": "center",
                "marginRight": "0.25rem",
                "marginBottom": "0.25rem",
            },
        )
        param_groups.append(group)

    # Container for manual parameters
    manual_container = html.Div(
        param_groups,
        style={"display": "flex", "flexWrap": "wrap", "alignItems": "center"},
    )

    if param_mode == "manual":
        return html.Div(
            [title, manual_container],
            style={"border": "1px solid #ccc", "padding": "0.5rem"},
        )
    else:
        # For auto-tune mode, return empty
        return html.Div([])


@dash.callback(
    [
        Output("clustering-data", "data"),  # Store computed clustering data
        Output("clustering-x-dropdown", "options"),
        Output("clustering-x-dropdown", "value"),
        Output("clustering-x-col", "style"),
        Output("clustering-y-dropdown", "options"),
        Output("clustering-y-dropdown", "value"),
        Output("clustering-y-col", "style"),
        Output("clustering-z-dropdown", "options"),
        Output("clustering-z-dropdown", "value"),
        Output("clustering-z-col", "style"),
        Output("clustering-dimension", "options"),
        Output("clustering-dimension", "value"),
        Output("clustering-dimension", "style"),
        Output("clustering-plot", "figure"),
        Output("ebm_card", "style"),
    ],
    [
        Input("compute-clustering-btn", "n_clicks"),  # Trigger clustering computation
        Input("clustering-dimension", "value"),  # 2D/3D switch
        Input("clustering-x-dropdown", "value"),  # Selected x-axis
        Input("clustering-y-dropdown", "value"),  # Selected y-axis
        Input("clustering-z-dropdown", "value"),
    ],  # Selected z-axis
    [
        State("clustering-param-mode", "value"),  # Auto/manual mode
        State("clustering-methods-dropdown", "value"),  # Clustering method
        State(
            {"type": "cluster-param-input", "method": ALL, "param": ALL}, "value"
        ),  # Parameters
        State(
            {"type": "cluster-param-input", "method": ALL, "param": ALL}, "id"
        ),  # Parameter IDs
        State("dr-transformed-data", "data"),  # DR data
        State("dr-selected-indices", "data"),  # DR selection
        State("filename-data", "data"),  # Original data
        State("use-whole-data-check", "value"),  # Data source toggle
        State("selected-columns-feature-relevance", "data"),  # Selected features
        Input("clustering-data", "data"),
        State("target-variable-dropdown", "value"),
    ],
    prevent_initial_call=True,
)
def merged_clustering_callback(
        n_clicks,
        dimension,
        x_axis,
        y_axis,
        z_axis,
        mode,
        selected_method,
        param_values,
        param_ids,
        dr_data,
        dr_selection,
        clean_data,
        use_whole_data,
        selected_columns,
        cluster_data,
        target_variable,
):
    # Define base styles
    hidden_style = {"display": "none"}
    visible_style = {"display": "block"}
    ebm_card_style_P2_default = {
        "height": "750px",  # Fixed height for uniformity
        "boxShadow": "0 4px 8px rgba(0,0,0,0.1)",
        "padding": "10px",
        "marginBottom": "10px",
        "borderRadius": "5px",
        "backgroundColor": "#fff",
    }

    # Initialize default values
    axis_options = []
    default_x = None
    default_y = None
    default_z = None
    radio_options = []
    radio_value = 2
    radio_style = hidden_style
    x_column_style = visible_style
    y_column_style = visible_style
    z_column_style = hidden_style
    figure = {}
    ebm_card_style_P2 = dash_styles.card_style_P2_hidden

    # Determine which input triggered the callback
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

    # If compute button was clicked, perform clustering
    if triggered_id == "compute-clustering-btn":
        if not selected_method or (not dr_data and not ("whole" in use_whole_data)):
            print("No clustering method or data available")
            return dash.no_update

        # Data selection logic
        if "whole" in use_whole_data:
            print("Using whole data for clustering")
            df = pd.read_csv(clean_data)
            df = df.reset_index().rename(columns={"index": "orig_index"})
            if len(selected_columns) > 0:
                # Ensure 'orig_index' is kept along with selected columns
                sel_cols = selected_columns.copy()
                if "orig_index" not in sel_cols:
                    sel_cols.append("orig_index")
                if target_variable not in sel_cols:
                    sel_cols.append(target_variable)
                df = df[sel_cols]
        else:
            # Read the original data and preserve the original index
            df = pd.read_csv(clean_data)
            df = df.reset_index().rename(columns={"index": "orig_index"})
            # Filter the dataframe using the preserved original indexes
            if dr_selection:
                if len(selected_columns) > 0:
                    sel_cols = selected_columns.copy()
                    if "orig_index" not in sel_cols:
                        sel_cols.append("orig_index")
                    df = df[df["orig_index"].isin(dr_selection)][sel_cols]
                else:
                    df = df[df["orig_index"].isin(dr_selection)]
            print(
                f"Using {'selected' if dr_selection else 'all'} DR data for clustering"
            )

        # Reset the index to ensure it's contiguous (this does not remove the 'orig_index' column)
        df = df.reset_index(drop=True)

        # Parameter handling
        if mode == "manual":
            params = {}
            for val, pid in zip(param_values, param_ids):
                if pid["method"] == selected_method:
                    params[pid["param"]] = val[0] if isinstance(val, list) else val
        else:
            print("Auto-tuning parameters")
            params = get_best_params(selected_method, df)

        print(f"The shape of the dataframe that is being clustered is {df.shape}")
        # Run clustering on the original data (with preserved orig_index)
        labels = clustering_cpu_only.run_cpu_clustering(selected_method, df, params)
        print(f"The number of labels returned: {len(labels)}")
        df["Cluster"] = labels
        print(f"The unique cluster labels before merging: {df['Cluster'].unique()}")
        cluster_data = df.to_dict("records")

        # Merge clustering results onto the DR data by matching orig_index
        if "whole" not in use_whole_data:
            dr_df = pd.DataFrame(dr_data)
            if dr_selection:
                dr_df = dr_df[dr_df["orig_index"].isin(dr_selection)]
                print(
                    f"The shape of the DR dataframe that is being clustered is {dr_df.shape}"
                )
            dr_df = dr_df.merge(
                df[["orig_index", "Cluster"]], on="orig_index", how="inner"
            )
            dr_df = dr_df.dropna(subset=["Cluster"])
            print(
                f"The unique cluster labels after merging: {dr_df['Cluster'].unique()}"
            )
            cluster_data = dr_df.to_dict("records")

    # Use computed or existing clustering data
    current_data = cluster_data if cluster_data else dash.no_update
    df = (
        pd.DataFrame(current_data)
        if current_data and current_data is not dash.no_update
        else pd.DataFrame()
    )

    if not df.empty:
        available_dims = [
            col for col in df.columns if col not in ("Cluster", "orig_index")
        ]
        axis_options = [{"label": col, "value": col} for col in available_dims]

        # Set default axis selections
        default_x = x_axis if x_axis in available_dims else available_dims[0]
        default_y = (
            y_axis
            if y_axis in available_dims
            else (available_dims[1] if len(available_dims) > 1 else available_dims[0])
        )

        # Handle 3D case if enough dimensions are available
        if len(available_dims) >= 3:
            default_z = z_axis if z_axis in available_dims else available_dims[2]
            radio_options = [{"label": "2D", "value": 2}, {"label": "3D", "value": 3}]
            radio_value = dimension if dimension in [2, 3] else 2
            radio_style = visible_style
            x_column_style = visible_style
            y_column_style = visible_style
            z_column_style = visible_style if radio_value == 3 else hidden_style
        else:
            radio_options = [{"label": "2D", "value": 2}]
            radio_value = 2
            radio_style = hidden_style
            x_column_style = visible_style
            y_column_style = visible_style
            z_column_style = hidden_style

        # Update plot
        if default_x and default_y:
            if radio_value == 3 and default_z:
                # 3D plot with cube aspect
                fig = px.scatter_3d(
                    df,
                    x=default_x,
                    y=default_y,
                    z=default_z,
                    color=df["Cluster"].astype(str),
                    custom_data=["orig_index"],
                    color_discrete_sequence=px.colors.qualitative.Set1,
                    hover_data={"orig_index": True},
                )
                fig.update_layout(
                    margin={"l": 40, "b": 40, "t": 10, "r": 0},
                    autosize=True,
                    scene={
                        "xaxis": {"title": default_x},
                        "yaxis": {"title": default_y},
                        "zaxis": {"title": default_z},
                        "aspectmode": "cube",
                    },
                    dragmode="lasso",
                    legend_title_text="Cluster",  # <- here’s the correct bit
                )
                fig.update_traces(marker=dict(size=2))
                figure = fig.to_dict()
            else:
                # 2D plot with matched axes.
                # x_vals = df[default_x].values
                # y_vals = df[default_y].values
                # common_min = min(x_vals.min(), y_vals.min())
                # common_max = max(x_vals.max(), y_vals.max())
                # common_range = [common_min, common_max]

                fig = px.scatter(
                    df,
                    x=default_x,
                    y=default_y,
                    color=df["Cluster"].astype(str),
                    custom_data=["orig_index"],
                    color_discrete_sequence=px.colors.qualitative.Set1,
                    hover_data={"orig_index": True},
                )

                fig.update_layout(
                    margin={"l": 40, "b": 40, "t": 10, "r": 0},
                    autosize=True,
                    dragmode="lasso",
                    legend_title_text="Cluster",  # <- here’s the correct bit
                )
                fig.update_traces(marker=dict(size=2))
                figure = fig.to_dict()

    # We check if select whole data is checked and if there is more than one cluster
    if len(df["Cluster"].unique()) > 1:
        ebm_card_style_P2 = ebm_card_style_P2_default

    return (
        (
            cluster_data if triggered_id == "compute-clustering-btn" else dash.no_update
        ),  # clustering-data
        axis_options,
        default_x,
        x_column_style,  # clustering-x-dropdown options and value
        axis_options,
        default_y,
        y_column_style,  # clustering-y-dropdown options and value
        axis_options,
        default_z,
        z_column_style,  # clustering-z-dropdown options, value, and style
        radio_options,
        radio_value,
        radio_style,  # clustering-dimension options, value, and style
        figure,  # clustering-plot figure
        ebm_card_style_P2,
    )


@dash.callback(
    Output("cluster-selected-indices", "data"),
    Input("clustering-plot", "selectedData"),
    State("clustering-data", "data"),
    prevent_initial_call=True,
)
def store_cluster_selection(selected_cluster_data, clustered_data):
    if not selected_cluster_data or not selected_cluster_data.get("points"):
        return pd.DataFrame(clustered_data).to_dict("records")

    # Extract the original indexes from custom_data
    orig_indices = [p["customdata"][0] for p in selected_cluster_data["points"]]

    df = pd.DataFrame(clustered_data)
    # Use boolean indexing to select rows where 'orig_index' is in orig_indices
    selected_df = df[df["orig_index"].isin(orig_indices)]
    print(
        f"Selected data shape: {selected_df.shape}, unique clusters: {selected_df['Cluster'].unique()}"
    )
    return selected_df.to_dict("records")


#
#
# CALLBACKS FOR EBM
#
#


@dash.callback(
    Output("cluster-to-use", "style"),
    Output("cluster-to-use", "options"),
    Output("cluster-to-use", "value"),
    Input("clustering-data", "data"),
    Input("cluster-selected-indices", "data"),
    prevent_initial_call=True,
    timeout=600000,
)
def update_cluster_dropdown(clustered_data, selected_cluster_data):
    if not selected_cluster_data and not clustered_data:
        return {"display": "none"}, [], None
    if not selected_cluster_data:
        clustered_clean_data = pd.DataFrame(clustered_data)
    else:
        clustered_clean_data = pd.DataFrame(selected_cluster_data)
    cluster_options = [
        {"label": f"Cluster {i}", "value": i}
        for i in clustered_clean_data["Cluster"].unique()
    ]
    # we make the cluster dropdown visible and we set the default value to the first cluster
    return {"display": "block"}, cluster_options, cluster_options[0]["value"]


@dash.callback(
    Output("ebm-graph", "figure"),
    Input("compute-em-btn", "n_clicks"),
    State("clustering-data", "data"),
    State("filename-data", "data"),
    State("cluster-to-use", "value"),
    State("selected-columns-feature-relevance", "data"),
    State("cluster-selected-indices", "data"),
    State("use-whole-data-check", "value"),
)
def update_ebm_graph(
        n_clicks,
        clustered_data,
        full_data,
        cluster_to_use,
        selected_columns,
        selected_cluster_indices,
        use_whole_data,
):
    if not clustered_data:
        return {}

    clustered_df = pd.DataFrame(
        clustered_data
    )  # Data with cluster labels and orig_index
    clean_df = pd.read_csv(full_data)
    print("1",clean_df.head(2))
    # Use selected columns if available
    if selected_columns:
        print("selected cols")
        clean_df = clean_df[selected_columns]

    # Ensure orig_index is present in clean_df
    clean_df = clean_df.reset_index().rename(columns={"index": "orig_index"})
    print("2",clean_df.head(2))
    # Case 1: Using whole data for clustering
    if "whole" in use_whole_data:
        print("whole")
        df = clustered_df
        # If cluster points are selected, filter them
        if selected_cluster_indices:
            actual_indices = [d["orig_index"] for d in selected_cluster_indices]
            df = df[df["orig_index"].isin(actual_indices)]
            print(
                f"Selected data shape: {df.shape}, unique clusters: {df['Cluster'].unique()}"
            )

    # Case 2: Using DR-transformed data
    else:
        print("not whole")
        # Merge just to get the Cluster labels; discard DR columns explicitly after merging.
        df = clean_df.merge(
            clustered_df[["orig_index", "Cluster"]], on="orig_index", how="inner"
        )
        print(f"Shape of merged data: {df.shape}")

        # Explicitly discard DR columns if they exist
        dr_columns = clustered_df.columns.difference(["orig_index", "Cluster"])
        print(f"DR columns to drop: {dr_columns.tolist()}")
        df = df.drop(columns=dr_columns, errors="ignore")
        print(f"Shape of merged data after dropping DR columns: {df.shape}")

        # Filter if cluster points are selected
        if selected_cluster_indices:
            actual_indices = [d["orig_index"] for d in selected_cluster_indices]
            df = df[df["orig_index"].isin(actual_indices)]
    # Validate all necessary columns are present
    if "Cluster" not in df.columns:
        print("Cluster column missing after merge.")
        return {}

    X = df.drop(columns=["Cluster", "orig_index"], errors="ignore")
    y = df["Cluster"]

    if X.empty or y.empty:
        print("Empty data after processing.")
        return {}

    # Instantiate and run EBM analysis
    ebm_instance = EBM(X_cluster=X, y_cluster=y, CLUSTER_LABEL_NO=cluster_to_use)
    accuracy, importances_df = ebm_instance.main()

    # Check if any feature has a positive importance
    if (importances_df["Importance"] > 0).any():
        # Sort the DataFrame in descending order by importance and take at most 10 rows.
        sorted_df = importances_df.sort_values(by="Importance", ascending=False).copy()
        filtered_df = sorted_df.head(15)
        title_text = "EBM Feature Importances"

    else:
        # All features have zero importance:
        # Create an empty scatter plot with a centered annotation.
        fig = px.scatter()
        fig.update_layout(
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                {
                    "text": "No importance data available (all features have 0 importance)",
                    "xref": "paper",
                    "yref": "paper",
                    "showarrow": False,
                    "x": 0.5,
                    "y": 0.5,
                    "font": {"size": 14},
                }
            ],
        )

        return fig

    # If we are plotting (high importance or nonzero importance), build the bar chart:
    filtered_df["y_numeric"] = range(len(filtered_df))
    fig = px.bar(
        filtered_df,
        x="Importance",
        y="Feature",  # Use feature names directly
        orientation="h",
        title=f"{title_text} (Accuracy: {accuracy:.2f})",
        color="Importance",
        color_continuous_scale="Viridis",
        custom_data=["Feature"],
        labels={"Feature": "Feature", "Importance": "Feature Importance"},
        # Optionally, preserve the desired order if filtered_df is pre-sorted:
        category_orders={"Feature": filtered_df["Feature"].tolist()[::]},
    )

    # Update hover to show full feature names
    fig.update_traces(
        hovertemplate="<b>%{customdata[0]}</b><br>Importance: %{x}<extra></extra>"
    )

    # Let Plotly automatically choose the tick labels (no forced tick settings)
    fig.update_layout(
        yaxis=dict(
            automargin=True,
        )
    )

    return fig
