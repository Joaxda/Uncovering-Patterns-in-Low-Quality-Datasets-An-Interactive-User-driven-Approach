# pages/dataset.py
import dash
from dash import callback_context as ctx
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import base64
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import assets.dash_styles as dash_styles
from components.dataset_util import analyze_dataset, clean_csv_cell, read_csv
from scripts.dataset.imputation import Imputation
from scripts.dataset.outliers import find_outliers
from scripts.dataset.FeatureConverter import SimpleFeatureConverter

import diskcache
cache = diskcache.Cache("/cache")

#
# CALLBACKS
#
#
@dash.callback(
    Output("second-normalize-column", "style"),
    Output("column-selector", "options"),
    Output("confirm-ordinal-btn", "style"),  # Add this output
    Input("normalize-choice-radio", "value"),
    Input("select-ordinal-btn", "n_clicks"),
    State("upload-data", "filename"),
    timeout=600000,
)
def update_dropdown_visibility(choice, n_clicks, file_path):
    # Default style for confirm button (hidden)
    confirm_btn_style = {"display": "none", "marginTop": "10px"}

    # Check if select-ordinal-btn was clicked
    ctx = dash.callback_context
    if (
        ctx.triggered
        and ctx.triggered[0]["prop_id"] == "select-ordinal-btn.n_clicks"
        and n_clicks
    ):
        confirm_btn_style = {"display": "block", "marginTop": "10px"}

    if choice != "One-Hot + Ordinal" or not file_path:
        return {"display": "none"}, [], confirm_btn_style

    try:
        df = pd.read_csv("temp/" + file_path)
        categorical_columns = df.select_dtypes(include=["object"]).columns
        column_options = [{"label": col, "value": col} for col in categorical_columns]
        return (
            {"display": "block"},
            column_options,
            confirm_btn_style
        )
    except Exception as e:
        print("Error in update_dropdown_visibility:", e)
        return {"display": "none"}, [], confirm_btn_style


@dash.callback(
    Output("ordinal-data-store", "data"),  # Store ordinal data in this store
    Output("confirm-ordinal-text", "children"),  # Store ordinal data in this store
    Input("confirm-ordinal-btn", "n_clicks"),
    State("column-selector", "value"),
    State({"type": "rank-input", "col": dash.ALL, "val": dash.ALL}, "value"),
    State({"type": "rank-input", "col": dash.ALL, "val": dash.ALL}, "id"),
    prevent_initial_call=True,
    timeout=600000,
)
def store_ordinal_data(n_clicks, selected_columns, selected_values, input_ids):
    if not n_clicks or not selected_columns:
        return dash.no_update

    # Create a dictionary with column → {value: rank}
    ordinal_data = {col: {} for col in selected_columns}

    # Filter out input_ids and selected_values for columns that are no longer selected
    filtered_values = []
    filtered_ids = []
    for val, input_id in zip(selected_values, input_ids):
        if input_id["col"] in selected_columns:
            filtered_values.append(val)
            filtered_ids.append(input_id)

    # Now use the filtered lists
    for val, input_id in zip(filtered_values, filtered_ids):
        col = input_id["col"]
        unique_val = input_id["val"]
        if (
            val is not None and unique_val is not None
        ):  # Add this check to avoid None values
            ordinal_data[col][unique_val] = val  # Map value → rank

    print(ordinal_data)
    return ordinal_data, "Rankings confirmed."


# Generate ranking UI when columns are confirmed
@dash.callback(
    Output("ordinal-ranking-container", "children"),
    Input("select-ordinal-btn", "n_clicks"),
    State("column-selector", "value"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
    timeout=600000,
)
def update_ranking_ui(n_clicks, selected_columns, file_path):
    if not n_clicks or not selected_columns or not file_path:
        return ""

    try:
        df = pd.read_csv("temp/" + file_path)
    except Exception:
        return "Error loading data."
    ranking_elements = []
    row_elements = []

    for col in selected_columns:
        unique_values = df[col].unique()
        dropdowns = [
            html.Div(
                [
                    html.Label(f"{val}:"),
                    dcc.Dropdown(
                        id={"type": "rank-input", "col": col, "val": val},
                        options=[
                            {"label": i, "value": i}
                            for i in range(1, len(unique_values) + 1)
                        ],
                        clearable=True,
                        style=dash_styles.dropdown_style,
                    ),
                ],
                style={"marginBottom": "5px"},
            )
            for val in unique_values
        ]
        row_elements.append(
            dbc.Col([html.H5(col)] + dropdowns, width=6)
        )  # Two columns per row

        if len(row_elements) == 2:
            ranking_elements.append(
                dbc.Row(row_elements, style={"marginBottom": "15px"})
            )
            row_elements = []

    if row_elements:
        ranking_elements.append(dbc.Row(row_elements, style={"marginBottom": "15px"}))
    return ranking_elements


# Ensure unique ranking within each column
@dash.callback(
    Output({"type": "rank-input", "col": dash.ALL, "val": dash.ALL}, "options"),
    Input({"type": "rank-input", "col": dash.ALL, "val": dash.ALL}, "value"),
    State({"type": "rank-input", "col": dash.ALL, "val": dash.ALL}, "id"),
    State("column-selector", "value"),  # Add this to get selected columns
    prevent_initial_call=True,
    timeout=600000,
)
def enforce_unique_ranks(selected_values, input_ids, selected_columns):
    updated_options = []
    selected_values_per_col = {}
    unique_values_count_per_col = {}

    # Count unique values per column and track selections
    for input_id in input_ids:
        col = input_id["col"]
        if col not in selected_values_per_col:
            selected_values_per_col[col] = set()
            # Get the count of unique values for this column
            unique_values_count_per_col[col] = sum(
                1 for id in input_ids if id["col"] == col
            )

    # Track selected values by column
    for val, input_id in zip(selected_values, input_ids):
        col = input_id["col"]
        if val:
            selected_values_per_col[col].add(val)

    # Update dropdown options to disable selected values in each column
    for input_id in input_ids:
        col = input_id["col"]
        # Use the column-specific count for options
        total_values = unique_values_count_per_col[col]

        options = [
            {
                "label": i,
                "value": i,
                "disabled": i in selected_values_per_col[col]
                and i != input_id.get("value"),
            }
            for i in range(1, total_values + 1)
        ]

        updated_options.append(options)

    return updated_options


@dash.callback(
    Output("shared-data", "data"),
    Input("delim-input", "value"),
    Input("null-value-input", "value"),
    timeout=600000,
)
def store_input_values(delim_input=",", null_value_input=""):
    return {
        "delimiter": delim_input if delim_input is not None else ",",
        "null_value": null_value_input if null_value_input is not None else "",
    }


@dash.callback(
    [
        Output("content-tab1", "style"),
        Output("content-tab2", "style"),
        Output("content-tab3", "style"),
        Output("content-tab4", "style"),
        Output("content-tab5", "style"),
    ],
    Input("tabs-example", "value"),
)
def display_tab(selected_tab):
    return [
        {"display": "block"} if selected_tab == "tab-1" else {"display": "none"},
        {"display": "block"} if selected_tab == "tab-2" else {"display": "none"},
        {"display": "block"} if selected_tab == "tab-3" else {"display": "none"},
        {"display": "block"} if selected_tab == "tab-4" else {"display": "none"},
        {"display": "block"} if selected_tab == "tab-5" else {"display": "none"},
    ]


@dash.callback(
    Output("filename-data", "data"),
    Input("upload-data", "filename"),
    prevent_initial_call=True,
    timeout=600000,
)
def put_filename_in_store(filename):
    return "temp/" + filename


@dash.callback(
    Output("upload-data", "children"),
    Input("upload-data", "filename"),
    prevent_initial_call=True,
    timeout=600000,
)
def update_upload_text(filename):
    if filename is None:
        filename = ""
    if filename:
        return html.Div(
            ["Selected file: ", html.Span(filename, style={"fontWeight": "bold"})]
        )
    return html.Div(["Drag and Drop or ", html.A("Select CSV File")])


@dash.callback(
    Output("shared-data-imputate", "data"),
    Input("null-col-dropdown", "value"),
    Input("null-replace-input", "value"),
    Input("replace-nan-btn", "n_clicks"),
    State("upload-data", "filename"),
    State("shared-data", "data"),
    prevent_initial_call=True,
    timeout=600000,
)
def update_nans(dropdown, input_value, n_clicks, filename, shared_data):
    triggered_id = ctx.triggered_id
    # if triggered_id not in {'null-col-dropdown', 'null-replace-input', 'replace-nan-val', 'upload-data', 'shared-data'}:
    #     raise dash.exceptions.PreventUpdate
    try:
        df = read_csv(
            "temp/" + filename, shared_data["delimiter"], shared_data["null_value"]
        )
        # print(df[dropdown].head(5))
        if triggered_id == "replace-nan-btn":
            df[dropdown] = df[dropdown].replace(np.nan, input_value)
            df.to_csv("temp/" + filename, index=False)
    except Exception as e:
        print("Error in update_nans", e)


@dash.callback(
    Output("health-issues-table", "filterModel"),  # Update the grid options!
    Input("error-bar-chart", "clickData"),
    prevent_initial_call=True,
    timeout=600000,
)
def filter_aggrid(clickData):

    filterModel = {
        "Errors Found": {
            "filterType": "text",
            "type": "Contains",
            "filter": "",
        }
    }
    # If there is clickData, add a filter for the "Errors Found" column
    if clickData:
        clicked_error = clickData["points"][0].get("label")
        if clicked_error:
            filterModel = {
                "Errors Found": {
                    "filterType": "text",
                    "type": "Contains",
                    "filter": clicked_error,
                }
            }

    return filterModel

@dash.callback(
    Output("output-data-upload", "children"),
    [Input("upload-data", "contents")],
    [State("upload-data", "filename")],
    timeout=600000,
)
def save_file(contents, filename):
    if contents is not None:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)

        # Define the directory to save the file
        directory = "temp"
        # Create the directory if it does not exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            # Delete all previous files in the folder
            for f in os.listdir(directory):
                file_path = os.path.join(directory, f)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        # Save the new file in the directory
        file_path = os.path.join(directory, filename)
        with open(file_path, "wb") as file:
            file.write(decoded)
        return []


@dash.callback(
    Output("encoding-status", "children"),
    Input("transform-numerical-btn", "n_clicks"),
    Input("normalize-choice-radio", "value"),
    State("upload-data", "filename"),
    State("shared-data", "data"),
    State("ordinal-data-store", "data"),
    prevent_initial_call=True,
    timeout=600000,
)
def transform(n_clicks_convert, ordinal_bool, filename, shared_data, ordinal_data):
    triggered_id = ctx.triggered_id
    if triggered_id not in {"transform-numerical-btn"}:
        raise dash.exceptions.PreventUpdate

    if n_clicks_convert and filename:
        df = read_csv(
            "temp/" + filename,
            shared_data["delimiter"],
            shared_data["null_value"],
        )
        perform_ordinal_bool = (
            True if (ordinal_bool == "One-Hot + Ordinal" and ordinal_data) else False
        )

        fcs = SimpleFeatureConverter(
            df, ordinal_data, perform_ordinal=perform_ordinal_bool
        )

        results = fcs.encode_and_normalize()

        results.to_csv("temp/" + filename, index=False)
        return "Encoding Complete."


@dash.callback(
    Output("outliers-table", "rowData"),
    Output("outliers-target-table", "rowData"),
    Input("outliers-table", "selectedRows"),
    Input("outliers-target-table", "selectedRows"),
    Input("outliers-btn", "n_clicks"),
    Input("row-del-btn", "n_clicks"),
    [
        State("outliers-table", "rowData"),
        State("outliers-target-table", "rowData"),
        State("upload-data", "filename"),
        State("shared-data", "data"),
    ],
    prevent_initial_call=True,
    timeout=600000,
)
def outliers(
    selected_rows_outliers,
    selected_rows_outliers_delete,
    search_clicks,
    row_del_clicks,
    outliers_table_data,
    outliers_table_data_delete,
    filename,
    shared_data,
):
    global df_outliers
    if "df_outliers" not in globals():
        df_outliers = pd.DataFrame()  # Initialize as empty DataFrame

    ctx = dash.callback_context
    triggered_id = ctx.triggered_id
    if triggered_id not in {"outliers-btn", "row-del-btn", "outliers-table"}:
        raise dash.exceptions.PreventUpdate

    # --------------------------------------------------
    # (1) Search Button: update outliers-table only
    if triggered_id == "outliers-btn" and search_clicks:
        try:
            df_outliers = read_csv(
                "temp/" + filename, shared_data["delimiter"], shared_data["null_value"]
            )
            df_outliers = find_outliers(df_outliers)
            # Update the outliers-table and clear the target table.
            return df_outliers.to_dict("records"), []
        except Exception as e:
            print(f"Error on outliers-btn: {e}")
            return [], []

    # --------------------------------------------------
    # (2) Row Selection in outliers-table: update target table only
    if triggered_id == "outliers-table" and selected_rows_outliers:
        try:
            df = read_csv(
                "temp/" + filename, shared_data["delimiter"], shared_data["null_value"]
            )
            selection = selected_rows_outliers[0]
            column = selection["Column"]
            high_or_low = selection["Value"]

            if high_or_low == "Possibly High Outlier(s)":
                table_data = df[column].nlargest(8)
            elif high_or_low == "Possibly Low Outlier(s)":
                table_data = df[df[column] != 0][column].nsmallest(8)
            else:
                # Return no update for the outliers table and clear target table
                return dash.no_update, []

            eighth_highest_df = pd.DataFrame(
                {"Value": table_data.values, "Row": table_data.index}
            )
            # Do not change outliers-table (first output) when a row is selected,
            # only update the target table.
            return dash.no_update, eighth_highest_df.to_dict("records")
        except Exception as e:
            print(f"Error on row selection in outliers-table: {e}")
            return dash.no_update, []

    # --------------------------------------------------
    # (3) Delete Button: remove selected rows from target table then refresh target table.
    if triggered_id == "row-del-btn" and row_del_clicks:
        # To refresh the target table after deletion we need an active selection from the outliers-table.
        if not selected_rows_outliers or not selected_rows_outliers_delete:
            raise dash.exceptions.PreventUpdate
        try:
            # Get the row identifiers from outliers-target-table selection
            rows_to_drop = [
                row["Row"] for row in selected_rows_outliers_delete if "Row" in row
            ]
            if not rows_to_drop:
                raise ValueError("No valid rows selected for deletion.")

            # Load the full CSV data and drop the rows
            df = read_csv(
                "temp/" + filename, shared_data["delimiter"], shared_data["null_value"]
            )
            df = df.drop(rows_to_drop)
            df.to_csv("temp/" + filename, sep=shared_data["delimiter"], index=False)

            # Now re-calculate target table values using the currently selected outlier
            selection = selected_rows_outliers[0]
            column = selection["Column"]
            high_or_low = selection["Value"]

            if high_or_low == "Possibly High Outlier(s)":
                table_data = df[column].nlargest(8)
            elif high_or_low == "Possibly Low Outlier(s)":
                table_data = df[df[column] != 0][column].nsmallest(8)
            else:
                return dash.no_update, []

            eighth_highest_df = pd.DataFrame(
                {"Value": table_data.values, "Row": table_data.index}
            )

            # For deletion, we return no update for outliers-table and a refreshed target table.
            return dash.no_update, eighth_highest_df.to_dict("records")
        except Exception as e:
            print(f"Error deleting rows in outliers callback: {e}")
            return dash.no_update, dash.no_update

    # --------------------------------------------------
    # Fallback: if no condition met but df_outliers exists, keep previous outliers-table value.
    if not df_outliers.empty:
        return df_outliers.to_dict("records"), dash.no_update
    return [], dash.no_update


# Update the boxplot callback to set all points to red and reflect selections
@dash.callback(
    Output("outlier-boxplot", "figure"),
    [
        Input("outliers-target-table", "rowData"),
        Input("outliers-target-table", "selectedRows"),
        Input("outliers-table", "selectedRows"),
    ],
    [State("upload-data", "filename"), State("shared-data", "data")],
    prevent_initial_call=True,
)
def update_boxplot(
    outliers_target_data,
    selected_target_rows,
    selected_rows_outliers,
    filename,
    shared_data,
):
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Select a column to view distribution",
        xaxis_title="",
        yaxis_title="Value",
        showlegend=False,
    )

    if not selected_rows_outliers or not outliers_target_data or not filename:
        return empty_fig

    try:
        column = selected_rows_outliers[0]["Column"]
        high_or_low = selected_rows_outliers[0]["Value"]

        df = pd.read_csv(
            "temp/" + filename,
            sep=shared_data["delimiter"],
            na_values=shared_data["null_value"],
            usecols=[column],
        )

        fig = go.Figure()
        fig.add_trace(
            go.Box(
                y=df[column].dropna(),
                name=column,
                boxpoints=False,
                marker_color="rgba(70, 130, 180, 0.7)",
                line_color="rgba(70, 130, 180, 1)",
            )
        )

        values = [row["Value"] for row in outliers_target_data]
        indices = [row["Row"] for row in outliers_target_data]

        # Default all points to red
        colors = ["rgba(255, 0, 0, 0.8)"] * len(values)
        sizes = [8] * len(values)

        # Update color for selected points
        if selected_target_rows:
            selected_indices = {
                row["Row"] for row in selected_target_rows
            }  # Convert to set for fast lookup
            for i, row in enumerate(outliers_target_data):
                if row["Row"] in selected_indices:
                    colors[i] = "rgba(0, 0, 255, 0.8)"  # Blue for selected points
                    sizes[i] = 10

        fig.add_trace(
            go.Scatter(
                y=values,
                x=[0] * len(values),
                mode="markers",
                name="Outlier Points",
                marker=dict(color=colors, size=sizes, symbol="circle"),
                hovertemplate="Value: %{y}<br>Index: %{customdata}<extra></extra>",
                customdata=indices,
            )
        )

        fig.update_layout(
            title=f"Distribution of {column} - {high_or_low}",
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            dragmode="lasso",
        )

        return fig

    except Exception as e:
        print(f"Error in boxplot callback: {e}")
        return empty_fig


@dash.callback(
    Output("outliers-target-table", "selectedRows"),
    Input("outlier-boxplot", "selectedData"),
    State("outliers-target-table", "rowData"),
    State("outliers-target-table", "selectedRows"),
    prevent_initial_call=True,
)
def sync_boxplot_selection_to_table(selected_data, table_data, current_selections):
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    # If no selections, persist the current selection instead of resetting
    if not selected_data or not table_data:
        return current_selections or []  # Keep previous selections

    try:
        selected_indices = [
            point["customdata"]
            for point in selected_data["points"]
            if "customdata" in point
        ]

        # Find corresponding rows in the table based on the "Row" key
        new_selected_rows = [
            row for row in table_data if row["Row"] in selected_indices
        ]

        # If selections haven't changed, prevent unnecessary updates
        if new_selected_rows == current_selections:
            raise dash.exceptions.PreventUpdate

        return new_selected_rows

    except Exception as e:
        print(f"Error in selection sync callback: {e}")
        return current_selections or []


@dash.callback(
    Output("imputation-text", "children"),
    Input("cpu-dropdown", "value"),
    Input("imputation-fastrun-dropdown", "value"),
    Input("imputate-btn", "n_clicks"),
    [State("upload-data", "filename"), State("shared-data", "data")],
    prevent_initial_call=True,
    running=[(Output("imputate-btn", "disabled"), True, False)],
    timeout=600000,  # 5 minutes in milliseconds
)
def imputation(
    cpu_value, fast_run, imputation_clicks, filename, shared_data
):  
    triggered_id = ctx.triggered_id
    fast_run = True if fast_run == "Yes" else False
    if triggered_id not in {"imputate-btn"}:
        raise dash.exceptions.PreventUpdate
    if triggered_id == "imputate-btn":
        try:
            if fast_run is not None:
                df = read_csv(
                    "temp/" + filename,
                    shared_data["delimiter"],
                    shared_data["null_value"],
                )
                df_sample = df.sample(frac=0.2, random_state=42)
                null_columns = df.columns[df.isnull().any()].tolist()
                imputation = Imputation(
                    df,
                    df_sample,
                    null_columns,
                    filename,
                    cpu_value,
                    fast_run,
                    True,
                )  # imputation_quality, delete_old_models
                imputated_dataset = imputation.perform_imputation()
                imputated_dataset.to_csv("temp/" + filename, index=False)
                return "Imputation complete."
        except Exception as e:
            print(f"Error in imputation: {e}")

@dash.callback(
    Output("null-col-dropdown", "value"),
    Input("missing-values-bar", "clickData"),
    State("null-col-dropdown", "options"),
)
def update_dropdown_value(clickData, options):
    if clickData:
        clicked_column = clickData["points"][0]["x"]
        return clicked_column if clicked_column in options else dash.no_update
    return dash.no_update

@dash.callback(
    Output("missing-values-bar", "figure"),
    Output("null-col-dropdown", "options"),
    Input("shared-data-imputate", "data"),
    Input("tabs-example", "value"),
    Input("missing-values-bar", "clickData"),  # Capture clicks on bars
    Input("imputation-text", "children"),
    State("upload-data", "filename"),
    State("shared-data", "data"),
    timeout=600000,
)
def update_imputation_text(_, selected_tab, clickData, imputate_btn, filename, shared_data):
    if selected_tab == "tab-3" and filename is not None:
        triggered_id = ctx.triggered_id
        if triggered_id == 'imputation-text':
            # Create an empty DataFrame with the right columns but no rows
            df = pd.DataFrame(columns=[
                "Column Name",
                "Number of Missing Values",
                "Color Label",
                "Percentage (%)"
            ])

            return px.bar(
                df,
                x="Column Name",
                y="Number of Missing Values",
                color="Color Label",
                color_discrete_map={"Above 40%": "red", "40% or Below": "green"},
                hover_data={"Percentage (%)": True},
                labels={
                    "Column Name": "Column Name",
                    "Number of Missing Values": "Missing Values",
                },
                title="Missing Values per Column",
                text="Number of Missing Values",
            ), [""]
        else:
            try:
                df = read_csv(
                    "temp/" + filename,
                    shared_data["delimiter"],
                    shared_data["null_value"],
                )

                null_counts = df.isnull().sum()
                null_counts = null_counts[null_counts > 0]

                if null_counts.empty:
                    return px.bar(), [""]

                total_rows = len(df)
                percentages = (null_counts / total_rows * 100).round(2)

                missing_data = pd.DataFrame(
                    {
                        "Column Name": null_counts.index,
                        "Number of Missing Values": null_counts.values,
                        "Percentage (%)": percentages.values,
                    }
                )

                missing_data["Missing Data Condition"] = missing_data["Percentage (%)"] > 40
                missing_data["Color Label"] = missing_data["Missing Data Condition"].map(
                    {True: "Above 40%", False: "40% or Below"}
                )

                missing_data = missing_data.sort_values(
                    by="Percentage (%)", ascending=False
                )

                top_n = 20
                if len(missing_data) > top_n:
                    missing_data = missing_data.iloc[:top_n]

                fig = px.bar(
                    missing_data,
                    x="Column Name",
                    y="Number of Missing Values",
                    color="Color Label",
                    color_discrete_map={"Above 40%": "red", "40% or Below": "green"},
                    hover_data={"Percentage (%)": True},
                    labels={
                        "x": "Column Name",
                        "Number of Missing Values": "Missing Values",
                    },
                    title="Missing Values per Column",
                    text="Number of Missing Values",
                )

                fig.update_layout(
                    xaxis_tickangle=-45,
                    margin=dict(l=40, r=40, t=50, b=150),
                    height=600,
                    legend_title_text="Missing Value Percentage",
                )

                fig.update_traces(opacity=0.8)

                return fig, missing_data["Column Name"]
            except Exception as e:
                print("Error in update_imputation_text", e)
                return px.bar(), [""]
    return px.bar(), [""]

@dash.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    State("upload-data", "filename"),
    prevent_initial_call=True,
    timeout=600000,
)
def save_modified(n_clicks, filename):
    df = pd.read_csv("temp/" + filename)
    return dcc.send_data_frame(
        df.to_csv, filename.strip(".csv") + "_modified.csv", index=False
    )


@dash.callback(
    Output("health-issues-table", "rowData"),
    Output("health-issues-table", "selectedRows"),
    Output("error-bar-chart", "figure"),
    Output("delim-input", "value"),
    Input("analyze-btn", "n_clicks"),
    Input("exclude-btn", "n_clicks"),
    Input("decimal-btn", "n_clicks"),
    Input("unicode-btn", "n_clicks"),
    [
        State("delim-input", "value"),
        State("null-value-input", "value"),
        State("null-threshold-input", "value"),
        State("repetitive-threshold-input", "value"),
        State("categorical-threshold-input", "value"),
        State("health-issues-table", "selectedRows"),
        State("health-issues-table", "rowData"),
        State("upload-data", "filename"),
        State("error-bar-chart", "clickData"),
    ],
    prevent_initial_call=True,
    timeout=600000,
)
def update_table(
    analyze_clicks,
    exclude_clicks,
    decimal_clicks,
    unicode_clicks,
    delimiter,
    nan_sign,
    nan_thresh,
    rep_thresh,
    cat_thresh,
    selected_rows,
    existing_table_data,
    filename,
    clickData,
):
    # Initialize variables
    triggered_id = ctx.triggered_id
    if triggered_id not in {
        "analyze-btn",
        "exclude-btn",
        "decimal-btn",
        "unicode-btn",
    }:
        raise dash.exceptions.PreventUpdate

    df = None
    table_data = existing_table_data if existing_table_data else []
    empty_fig = {
        "data": [],
        "layout": {
            "title": "Error Type Counts",
            "xaxis": {"title": "Error Type"},
            "yaxis": {"title": "Count"},
        },
    }

    # Load data if filename exists
    if filename is not None:
        na_values = nan_sign if nan_sign != "" else None
        delimiter_val = delimiter if delimiter != "" else ","
        try:
            df = read_csv("temp/" + filename, delimiter_val, na_values)
            if delimiter_val != ",":
                df.to_csv("temp/" + filename, index=False, sep=",")
        except Exception as e:
            print(f"Error reading file: {e}")
            return table_data, selected_rows, empty_fig, ","

    # Handle analyze button click
    if triggered_id == "analyze-btn" and analyze_clicks and filename and df is not None:
        try:
            table_data, fig = analyze_dataset(
                df, nan_thresh, rep_thresh, cat_thresh, empty_fig
            )
            return table_data, [], fig, ","
        except Exception as e:
            print(f"Error analyzing data: {e}")
            return table_data, selected_rows, empty_fig, ","

    # Handle exclude button click
    elif triggered_id == "exclude-btn" and exclude_clicks:
        if selected_rows and len(selected_rows) > 0:
            # Get columns to drop
            columns_to_drop = [row["Column"] for row in selected_rows]

            # Remove selected rows from table data
            table_data = [
                row
                for row in existing_table_data
                if row["Column"] not in columns_to_drop
            ]

            if filename and df is not None:
                try:
                    # Drop the columns and save
                    df = df.drop(columns=columns_to_drop)
                    df.to_csv("temp/" + filename, index=False)

                    # Update the chart
                    _, fig = analyze_dataset(
                        df, nan_thresh, rep_thresh, cat_thresh, empty_fig
                    )
                except Exception as e:
                    print(f"Error updating DataFrame: {e}")
                    fig = empty_fig
            else:
                fig = empty_fig

            return table_data, [], fig, ","

    # Handle decimal button click
    elif triggered_id == "decimal-btn" and decimal_clicks:
        if df is not None and selected_rows and len(selected_rows) > 0:
            try:
                selected_columns = [row["Column"] for row in selected_rows]
                df = clean_csv_cell(df, selected_columns)
                df.to_csv("temp/" + filename, index=False)  # Save the changes

                # Re-analyze the dataset
                table_data, fig = analyze_dataset(
                    df, nan_thresh, rep_thresh, cat_thresh, empty_fig
                )

                return table_data, [], fig, ","
            except Exception as e:
                print(f"Error processing decimal conversion: {e}")
                # If error, just return current state
                return table_data, selected_rows, empty_fig, ","

    elif triggered_id == "unicode-btn" and unicode_clicks and filename is not None:
        colnames = [row["Column"] for row in selected_rows]
        problematic_chars = {
            "−": ("-", "Unicode: EN DASH (U+2212)"),
            "–": ("-", "Unicode: EN DASH (U+2013)"),
            "—": ("-", "Unicode: EM DASH (U+2014)"),
            "‐": ("-", "Unicode: HYPHEN (U+2010)"),
            "“": ('"', "Unicode: CURVED QUOTE (U+201C)"),
            "”": ('"', "Unicode: CURVED QUOTE (U+201D)"),
            "'": ("", "Unicode: CURVED QUOTE (U+2019)"),
            "‘": ("'", "Unicode: CURVED QUOTE (U+2018)"),
            "’": ("'", "Unicode: CURVED QUOTE (U+2019)"),
            "…": ("", "Unicode: ELLIUnicode: PSIS (U+2026)"),
            "\xa0": (" ", "Unicode: NON-BREAKING SPACE (U+00A0)"),
        }

        for col in colnames:
            df[col] = df[col].copy()
            for key, (replacement, description) in problematic_chars.items():
                df[col] = df[col].str.replace(key, replacement, regex=False)

        df.to_csv("temp/" + filename, index=False)
        # Update the chart
        table_data, fig = analyze_dataset(
            df, nan_thresh, rep_thresh, cat_thresh, empty_fig
        )

        return table_data, [], fig, ","
    # Default return if no specific condition matched
    return table_data, selected_rows, empty_fig, ","
