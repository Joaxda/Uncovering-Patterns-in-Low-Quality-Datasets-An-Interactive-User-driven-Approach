import time
import re
import pandas as pd
from collections import defaultdict

class CheckDataset:
    def __init__(self, df, nan_thresh, rep_thresh, cat_thresh):
        self.df = df.copy()  # work on a copy so we don’t alter caller’s DataFrame
        self.nan_thresh = float(nan_thresh)
        self.rep_thresh = float(rep_thresh)
        self.cat_thresh = float(cat_thresh)
        self.col_error_counter = {
            'Comma': 0,
            'Unicode': 0,
            'Numeric': 0,
            'NaN': 0,
            'Majority': 0,
            'Categorical': 0,
            'No Issue': 0,
        }

    def test_numeric_conversion(self, value):
        """
        Tests if a value can be properly converted to a number.
        Returns a tuple (is_convertible, error_message, suggested_fix)
        """
        if pd.isna(value):
            return True, None, None

        try:
            if isinstance(value, (int, float)):
                return True, None, None

            # Try to convert string to float
            value_str = str(value).strip()
            float(value_str)
            return True, None, None
        except ValueError:
            # Check for known problematic characters
            if "−" in value_str:  # Unicode minus (U+2212)
                suggested = value_str.replace("−", "-")
                try:
                    float(suggested)
                    return False, "Contains Unicode minus sign (U+2212)", suggested
                except ValueError:
                    pass

            if "," in value_str and "." in value_str:
                # Try European number format (1.234,56 -> 1234.56)
                suggested = value_str.replace(".", "").replace(",", ".")
                try:
                    float(suggested)
                    return False, "Uses European number format (comma as decimal)", suggested
                except ValueError:
                    pass
            elif "," in value_str:
                # Try replacing comma with dot
                suggested = value_str.replace(",", ".")
                try:
                    float(suggested)
                    return False, "Uses comma as decimal separator", suggested
                except ValueError:
                    pass

            return False, "Numeric Error: Cannot be converted to number", None

    def analyze_columns(self):
        """
        Analyzes DataFrame columns and returns a dictionary with column names
        as keys and lists of error dictionaries as values.
        Uses vectorized operations where possible but keeps output identical.
        """
        print("Processing Analysis...")
        start = time.time()
        results = defaultdict(list)

        # Define problematic characters.
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

        # Precompile regex patterns for problematic character check.
        problematic_re = re.compile(r'([−–—‐"\'…\xa0])')
        non_ascii_re = re.compile(r'([^\x00-\x7F])')
        allowed_chars = set(['å', 'ä', 'ö', 'Å', 'Ä', 'Ö'])

        # Clean column names (identical to original).
        self.df.columns = [
            col if not col.startswith('Unnamed: ')
            else f'Column_{col.split(": ")[1]}' for col in self.df.columns
        ]

        for column in self.df.columns:
            TEMP_PROBLEMATIC_CHARS_BOOL = False
            col_data = self.df[column]
            col_dtype = col_data.dtypes
            total_rows = len(col_data)
            char_issues = defaultdict(list)

            # 1. Check NaN values using vectorized operations.
            nan_count = col_data.isna().sum()
            nan_percentage = (nan_count / total_rows) * 100
            if nan_percentage > self.nan_thresh:
                results[column].append({
                    "Issue": "Contains NaN values (%)",
                    "Count": nan_percentage,
                    "Example": None
                })
                self.col_error_counter['NaN'] += 1

            # 2. Check if the column appears to be numeric.
            try:
                # If this works without error, assume all values convert.
                pd.to_numeric(col_data, errors="raise")
                is_numeric = True
            except Exception:
                non_na = col_data.dropna().astype(str)
                replaced = non_na.str.replace(',', '.', regex=False)\
                                 .str.replace('-', '', regex=False)\
                                 .str.replace('.', '', regex=False)
                numeric_values = replaced.str.isdigit().sum()
                is_numeric = (numeric_values / len(non_na)) > 0.5 if len(non_na) > 0 else False

            # If column appears numeric, check for conversion issues.
            if is_numeric:
                conv_results = col_data.dropna().apply(self.test_numeric_conversion)
                numeric_issues = defaultdict(list)
                for idx, res_tuple in conv_results.items():
                    convertible, error, fix = res_tuple
                    if not convertible and error:
                        numeric_issues[error].append((col_data.loc[idx], fix))
                for error, examples in numeric_issues.items():
                    count = len(examples)
                    example_val, example_fix = examples[0]
                    results[column].append({
                        "Issue": error,
                        "Count": count,
                        "Example": f"'{example_val}' → '{example_fix}'"
                    })
                    if "comma" in error.lower():
                        self.col_error_counter['Comma'] += 1
                    else:
                        self.col_error_counter['Numeric'] += 1

            # 3. Check for value majority using pandas value_counts.
            value_counts = col_data.value_counts()
            if not value_counts.empty:
                most_common_value = value_counts.index[0]
                most_common_count = value_counts.iloc[0]
                most_common_percentage = (most_common_count / total_rows) * 100
                if most_common_percentage >= self.rep_thresh:
                    results[column].append({
                        "Issue": "Value majority (%)",
                        "Count": "",
                        "Example": f"'{most_common_value}' appears in {most_common_percentage:.1f}% of rows"
                    })
                    self.col_error_counter['Majority'] += 1

            # 4. For object columns with a large number of unique values, check for problematic characters.
            if col_dtype == 'object' and len(pd.unique(col_data)) > self.cat_thresh:
                unique_count = int(len(pd.unique(col_data)))
                results[column].append({
                    "Issue": "Large amount of unique categorical variables",
                    "Count": unique_count,
                    "Example": ""
                })
                self.col_error_counter['Categorical'] += 1

                # Instead of looping character by character, use regex on each non-null value.
                for original_value in col_data.dropna():
                    # Convert the value to a string to avoid type errors.
                    value_str = str(original_value)
                    # Process known problematic characters.
                    for match in problematic_re.finditer(value_str):
                        char = match.group(1)
                        replacement, char_name = problematic_chars[char]
                        char_issues[char_name].append((char, replacement, value_str))
                        TEMP_PROBLEMATIC_CHARS_BOOL = True
                    # Process any non-ascii characters (excluding allowed ones and those already handled).
                    for match in non_ascii_re.finditer(value_str):
                        char = match.group(1)
                        if char not in allowed_chars and char not in problematic_chars:
                            char_name = f'Unicode character (U+{ord(char):04X})'
                            char_issues[char_name].append((char, None, value_str))
                            TEMP_PROBLEMATIC_CHARS_BOOL = True
                    for sign in value_str:
                        if sign in problematic_chars:
                            TEMP_PROBLEMATIC_CHARS_BOOL = True

            # 5. Append problematic character issues if column is of string type.
            if pd.api.types.is_string_dtype(col_data):
                for char_name, examples in char_issues.items():
                    count = len(examples)
                    char, replacement, example_val = examples[0]
                    if replacement:
                        results[column].append({
                            "Issue": char_name,
                            "Count": count,
                            "Example": f"'{example_val}' ('{char}' → '{replacement}')"
                        })
                    else:
                        results[column].append({
                            "Issue": char_name,
                            "Count": count,
                            "Example": f"'{example_val}' (character: '{char}')"
                        })
            if TEMP_PROBLEMATIC_CHARS_BOOL:
                self.col_error_counter['Unicode'] += 1

            # 6. If no issues were found for this column, record that.
            if not results[column]:
                results[column].append({
                    "Issue": "No issues found",
                    "Count": None,
                    "Example": None
                })
                self.col_error_counter['No Issue'] += 1

        print("Done analyzing, execution time: ", (time.time() - start))
        return dict(results), self.col_error_counter
