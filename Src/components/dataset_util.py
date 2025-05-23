import re
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from scripts.dataset.checkDataset import CheckDataset

def clean_csv_cell(df, selected_columns):
    # Iterate over selected columns
    for colname in selected_columns:
        # Replace decimal commas with dots and remove quotation marks
        df[colname] = df[colname].apply(
            lambda cell: re.sub(r"(?<=\d),(?=\d)", ".", str(cell)).replace('"', "")
        )
        df[colname] = pd.to_numeric(df[colname], errors='coerce')
    return df


def read_csv(path, _delimiter, _nan_sign):
    df = pd.read_csv(path, delimiter=_delimiter, na_values=_nan_sign, low_memory=False)
    return df

def analyze_dataset(dataframe, nan_thresh, rep_thresh, cat_thresh, empty_fig):
    nan_thresh_val = 0 if nan_thresh == "" else float(nan_thresh)
    rep_thresh_val = 70 if rep_thresh == "" else float(rep_thresh)
    cat_thresh_val = 5 if cat_thresh == "" else int(cat_thresh)

    checker = CheckDataset(
        dataframe, nan_thresh_val, rep_thresh_val, cat_thresh_val
    )
    column_results, error_counter = checker.analyze_columns()

    # Create updated bar chart - only if we have data
    if error_counter:
        categories = list(error_counter.keys())
        values = list(error_counter.values())

        # Define color mapping
        color_map = {
            "Comma": "red",
            "Unicode": "red",
            "Numeric": "orange",
            "NaN": "orange",
            "Majority": "orange",
            "Categorical": "orange",
            "No Issue": "green",
        }

        # Assign colors based on categories
        colors = [color_map.get(category, "blue") for category in categories]

        # Create bar plot with specific colors
        updated_fig = px.bar(
            x=categories,
            y=values,
            labels={"x": "Error Type", "y": "Count"},
            title="Error Type Counts",
        )

        # Apply the colors
        updated_fig.update_traces(marker=dict(color=colors))

        # Add legend manually using dummy traces
        legend_items = {"Critical": "red", "Potential": "orange", "Ok": "green"}

        for label, color in legend_items.items():
            updated_fig.add_trace(
                go.Bar(
                    x=[None],
                    y=[None],  # Invisible data points
                    marker=dict(color=color),
                    name=label,
                )
            )

    else:
        updated_fig = empty_fig

    # Generate table data
    updated_table_data = []
    for column in dataframe.columns:
        errors_list = column_results.get(column, [])
        # Format each issue dictionary into a readable string
        formatted_errors = []
        for error in errors_list:
            error_str = f"- {error['Issue']}"
            if error["Count"] is not None:
                try:
                    value = float(error["Count"])
                    if value.is_integer():
                        error_str += f": {int(value)}"
                    else:
                        error_str += f": {value:.1f}"
                except (ValueError, TypeError):
                    error_str += f": {error['Count']}"
            if error["Example"]:
                error_str += f" ({error['Example']})"
            formatted_errors.append(error_str)

        updated_table_data.append(
            {
                "Column": column,
                "Errors Found": (
                    "\n".join(formatted_errors)
                    if formatted_errors
                    else "* No issues found"
                ),
            }
        )

    return updated_table_data, updated_fig