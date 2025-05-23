import pandas as pd
import numpy as np

THRESHOLD_MULTIPLIER = 5

def check_high(df, col):
    POSSIBLE_OUTLIERS = False
    eight_highest = df[col].nlargest(8)
    if len(eight_highest.unique()) <= 3:
        return []
    largest = eight_highest.iloc[0]
    for value in eight_highest:
        if value != 0:
            if largest/value > THRESHOLD_MULTIPLIER:
                POSSIBLE_OUTLIERS = True
                return POSSIBLE_OUTLIERS
    return POSSIBLE_OUTLIERS

def check_lowest(df, col):
    NEGATIVE = False
    POSSIBLE_OUTLIERS = False
    eight_lowest = df[df[col] != 0][col].nsmallest(8)
    if len(eight_lowest.unique()) <= 3:
        return []
    
    lowest = eight_lowest.iloc[0]

    if lowest < 0:
        NEGATIVE = True
    if NEGATIVE:
        eight_lowest = abs(eight_lowest)
        lowest = abs(eight_lowest.iloc[0])
    else:
        eight_lowest = abs(eight_lowest.iloc[::-1])
    temp_counter = 0
    for value in eight_lowest:
        if value != 0 and temp_counter + 1 < len(eight_lowest):
            if lowest/value > THRESHOLD_MULTIPLIER:
                POSSIBLE_OUTLIERS = True
                return POSSIBLE_OUTLIERS
    return POSSIBLE_OUTLIERS

def find_outliers(df):
    outliers = pd.DataFrame(columns=['Column', 'Value'])
    df_no_NA = df.dropna()
    numerical_columns = df_no_NA.select_dtypes(include=[np.number]).columns
    
    # Iterate over each numerical column
    for col in numerical_columns:
        
        # Check for high and low outliers
        possibly_high_outliers = check_high(df, col)
        possibly_low_outliers = check_lowest(df, col)
        
        # Add to the DataFrame if high outliers exist
        if possibly_high_outliers:
            outliers = pd.concat([outliers, pd.DataFrame({'Column': [col], 'Value': ['Possibly High Outlier(s)']})])
        
        # Add to the DataFrame if low outliers exist
        if possibly_low_outliers:
            outliers = pd.concat([outliers, pd.DataFrame({'Column': [col], 'Value': ['Possibly Low Outlier(s)']})])
    
    # Return the final list of outliers
    return outliers
