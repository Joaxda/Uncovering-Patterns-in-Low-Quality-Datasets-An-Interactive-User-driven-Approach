import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

class SimpleFeatureConverter:
    def __init__(self, df, ordinal_dict, perform_ordinal):
        self.df = df
        self.ordinal_dict = ordinal_dict
        self.PERFORM_ORDINAL = perform_ordinal

    def encode_and_normalize(self):
        """
        Performs encoding and normalization on different types of columns:
        - Categorical columns (<=5 unique values): One-hot encoding only
        - Large categorical columns (>5 unique values): Feature hashing + normalization
        - Binary columns: Ordinal encoding only
        - Numerical columns: Normalization only
        """
        categorical_columns = []
        binary_columns = []
        numerical_columns = []
        ordinal_columns = []
        ordinal_order = []
        for col in self.df.columns:
            ORDINAL = False
            col_dtype = self.df[col].dtype
            if col_dtype == 'int64' or col_dtype == 'float64':  # Fixed typo in 'int64'
                numerical_columns.append(col)
            elif col_dtype == 'object':
                col_unique_values = len(pd.unique(self.df[col]))  # Fixed df to self.df
                if col_unique_values >= 2 and self.ordinal_dict is not None:
                    if len(self.ordinal_dict) > 0:
                        if col in self.ordinal_dict.keys():
                            ORDINAL = True
                            item = self.ordinal_dict.get(col)
                            #sorted_categories = list(self.ordinal_dict.keys()) 
                            sorted_categories = list(sorted(item, key=item.get))
                            ordinal_columns.append(col)
                            ordinal_order.append(sorted_categories)
                if not ORDINAL:
                    if col_unique_values <= 2:
                        binary_columns.append(col)
                    else:
                        categorical_columns.append(col)
        # Initialize encoders and scaler
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # Set sparse_output=False
        ordinal_encoder = OrdinalEncoder()
        scaler = MinMaxScaler()
        
        # Create a copy of the dataframe to avoid modifying the original
        processed_df = self.df.copy()
        
        # Process categorical columns (One-hot encoding only)
        if categorical_columns:
            # Perform one-hot encoding
            onehot_encoded = onehot_encoder.fit_transform(processed_df[categorical_columns])
            
            # Get feature names from encoder
            onehot_column_names = onehot_encoder.get_feature_names_out(categorical_columns)
            
            # Create a DataFrame with encoded values and concatenate
            onehot_df = pd.DataFrame(onehot_encoded, columns=onehot_column_names, index=processed_df.index)
            processed_df = pd.concat([processed_df.drop(categorical_columns, axis=1), onehot_df], axis=1)

        # Process binary columns (Ordinal encoding only)
        if binary_columns:
            # Perform ordinal encoding
            binary_encoded = ordinal_encoder.fit_transform(processed_df[binary_columns])
            
            # Replace original columns with encoded values
            for i, col in enumerate(binary_columns):
                processed_df[col] = binary_encoded[:, i]
        
        # Process numerical columns (Normalization only)
        if numerical_columns:
            numerical_normalized = scaler.fit_transform(processed_df[numerical_columns])
            
            # Replace original columns with normalized ones
            for i, col in enumerate(numerical_columns):
                processed_df[col] = numerical_normalized[:, i]
        
        if ordinal_columns:
            try:
                ordinal_encoded = ordinal_encoder.fit_transform(processed_df[ordinal_columns], ordinal_order)
                ordinal_encoded = scaler.fit_transform(ordinal_encoded)
            except Exception as e:
                print("asd", e)
            for i, col in enumerate(ordinal_columns):
                processed_df[col] = ordinal_encoded[:, i]

        self.df = processed_df
        return self.df

# # Example usage
# if __name__ == "__main__":
#     df = pd.read_csv('Src/temp/johan_imputed.csv')
#     print(len(df.columns))
#     SFC = SimpleFeatureConverter(df)
#     results= SFC.encode_and_normalize().head(5)
#     print(results.head(5))
#     print(len(results.columns))
#     print(results.columns)
