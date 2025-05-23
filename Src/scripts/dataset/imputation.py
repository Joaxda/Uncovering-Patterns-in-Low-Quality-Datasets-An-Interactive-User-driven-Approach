import os
import shutil
from autogluon.tabular import TabularDataset, TabularPredictor

# from autogluon.tabular.models import RFModel # <<<---- for running one only


# Inputs:
# df : data in pd.Dataframe format
# filename : for saving files correctly so that models can be extracted if users use multiple files.
# quality : what quality the user choose on the imputation technique
# models : which models that should be used 
# fast_run : bool that makes the algorithm run 2 models only
# DELETE_OLD_MODELS : if user wants to delete previous models of this file (based of csv data)
#
# Outputs:
# new_df : The new df with imputed values.
# path_to_pkl_models : the paths to the best predictor.pkl models that can be used later if needed.
class Imputation:
    def __init__(self, df, train_df, columns_to_impute, filename, cpu_amount, fast_run, delete_old_models):  #quality, models={}
        """
        Initialize Imputation class.

        Args:
            df: Full dataset to impute
            train_df: Training dataset used to train the imputation models
            columns_to_impute: List of columns to perform imputation on
            filename: Base filename for saving models
            quality: Quality preset for AutoGluon
            models: Dictionary of models to use
            delete_old_models: Whether to delete old model files
            cpu_amount: Number of CPUs to use
        """
        self.df = df
        self.train_df = train_df
        self.columns_to_impute = columns_to_impute
        self.filename = filename
        self.delete_old_models = delete_old_models
        self.cpu_amount = cpu_amount
        self.fast_run = fast_run

    def delete_models(self):
        autogluon_path = './AutogluonModels/'#os.path.join("..", "AutogluonModels")
        if os.path.exists(autogluon_path):
            shutil.rmtree(autogluon_path)
            os.mkdir(autogluon_path)
            print("AutogluonModels has been wiped clean.")
        else:
            print("Folder does not exist.")

    def perform_imputation(self):
        """
        Perform imputation using the training data to train models and impute the full dataset.
        """
        try:
            paths_to_pkl_models = []

            # Use provided columns or detect missing values
            if self.columns_to_impute is None:
                self.columns_to_impute = self.df.columns[
                    self.df.isnull().any()
                ].tolist()

            print(f"Columns being imputed: {self.columns_to_impute}")

            # Convert both dataframes to TabularDataset
            self.df = TabularDataset(self.df)
            self.train_df = TabularDataset(self.train_df)

            for col in self.columns_to_impute:
                if col not in self.df.columns:
                    print(f"Warning: Column {col} not found in dataset. Skipping.")
                    continue

                paths_to_pkl_models.append(f"AutogluonModels/{self.filename}_{col}")
                missing_rows = self.df[col].isna()

                # Only proceed with imputation if there are actually missing values
                if not missing_rows.any():
                    print(
                        f"Warning: No missing values found in column {col}. Skipping imputation."
                    )
                    continue

                # Filter out rows where the target column is missing in training data
                train_data = self.train_df[~self.train_df[col].isna()]

                if len(train_data) == 0:
                    print(
                        f"Warning: No complete training data available for column {col}. Skipping imputation."
                    )
                    continue
                    
                predictor = TabularPredictor(
                    path=f"AutogluonModels/{self.filename}_{col}",
                    label=col,
                    problem_type=None,
                )
                
                if self.fast_run == True:
                    predictor.fit(
                        train_data=train_data,
                        presets=['medium_quality'],
                        hyperparameters={'RF' : {}, 'KNN' : {}},
                        num_cpus=self.cpu_amount,
                    )
                else:
                    params = {
                            'GBM' : {'ag_args_fit': {'num_gpus': 0}},
                            'XGB' : {'ag_args_fit': {'num_gpus': 0}},
                            'RF' : {'ag_args_fit': {'num_gpus': 0}},
                            'XT' : {'ag_args_fit': {'num_gpus': 0}},
                            'KNN' : {'ag_args_fit': {'num_gpus': 0}},
                            'LR' : {'ag_args_fit': {'num_gpus': 0}},
                    }
                    predictor.fit(
                        train_data=train_data,
                        presets=['medium_quality'],
                        hyperparameters = params,
                        num_cpus=self.cpu_amount,
                    )

                # Predict missing values only for rows that were originally missing
                self.df.loc[missing_rows, col] = predictor.predict(self.df.loc[missing_rows])
                if self.delete_old_models:
                    self.delete_models()
            return self.df#, paths_to_pkl_models
        except Exception as e:
            if self.delete_old_models:
                self.delete_models()
            print(f"ERROR IN IMPUTATION: {str(e)}")
            return None
