import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List
import numpy as np
from .cluster_similarity import ClusterSimilarity
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
import pickle

class Preprocessor:

    def load_data(self, path:str)->pd.DataFrame:
        """
               Load a DataFrame from a CSV file and reset its index.

               Parameters:
               - path (str): The path to the CSV file.

               Returns:
               - pd.DataFrame: The loaded DataFrame.
        """

        df = pd.read_csv(path).reset_index(drop=True)
        return df

    @staticmethod
    def drop_unnecessary_columns(df: pd.DataFrame, cols: List[str]) -> None:
        """
                Drop specified columns from the DataFrame in-place.

                Parameters:
                - df (pd.DataFrame): The DataFrame to modify.
                - cols (List[str]): A list of column names to drop.
        """

        df.drop(columns=cols, inplace=True)

    @staticmethod
    def obtain_non_null_rows(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        """
                Return a DataFrame containing rows where a specific column is not null.

                Parameters:
                - df (pd.DataFrame): The DataFrame to filter.
                - col_name (str): The column name to check for null values.

                Returns:
                - pd.DataFrame: The filtered DataFrame.
        """

        return df[~df[col_name].isnull()]

    @staticmethod
    def stratify_sample(df:pd.DataFrame, col: str, col_cutted: str)-> (pd.DataFrame, pd.DataFrame):

        """
               Stratify and split the DataFrame into training and testing sets based on a specified column.

               Parameters:
               - df (pd.DataFrame): The DataFrame to split.
               - col (str): The column to stratify the split.
               - col_cutted (str): The column to be cut for stratification.

               Returns:
               - Tuple[pd.DataFrame, pd.DataFrame]: The stratified training and testing DataFrames.
        """

        df[col] = pd.cut(df[col_cutted],
                                bins=[0, 500000, 1000000, 5000000, 10000000, np.inf],
                                labels = [1, 2, 3, 4, 5])
        strat_train_set, strat_test_set = train_test_split(df,
                                                           stratify=df[col],
                                                           random_state=42)
        return strat_train_set, strat_test_set

    @staticmethod
    def logarithm_target(df: pd.DataFrame, col: str)->pd.DataFrame:
        """
               Apply the natural logarithm to a specified column in the DataFrame.

               Parameters:
               - df (pd.DataFrame): The DataFrame containing the target column.
               - col (str): The target column to apply the logarithm.

               Returns:
               - pd.DataFrame: The DataFrame with the transformed column.
        """

        df[col] = np.log(df[col])
        return df[col]


    def build_preprocessing_pipeline(self, use_svd = True)-> Pipeline:
        '''
        -surface_total_in_m2 null values will get replaced by mean or median imputer
        -lat and lon null values will get replaced by mode (frequent value)
        -Since there is no ordinal connection: house,apartment,store and PH we will use OneHotEncoder
        -Same approach with borough
        -We will scale features in the end (those that are numerical, not categorical)
        -We will reduce dimensionality of both train and test set, by using TruncatedSVD algorithm
        -We will add also the cluster_similarity by ClusterSimilarity to group latitude and longitude
        '''

        surface_pipeline = Pipeline([('imputer',SimpleImputer(strategy="median")),
                                     ('scaler',StandardScaler())
                                     ])

        lat_lon_pipeline = Pipeline([('imputer',SimpleImputer(strategy="most_frequent")),
                                     ('cluster_similarity',ClusterSimilarity(n_clusters=10, gamma=1., random_state=42))
                                     ])

        property_borough = Pipeline([
            ('one_hot_encoder', OneHotEncoder(handle_unknown="ignore"))
        ])

        # Initializing TruncatedSVD algorithm (We apply TruncatedSVD, since it's suited for sparse data)
        truncated_svd = TruncatedSVD(random_state=42)

        transformer = ColumnTransformer(transformers=[
            ("surface", surface_pipeline, ["surface_total_in_m2"]),
            ("lat_lon", lat_lon_pipeline, ["lat","lon"]),
            ("property_borough", property_borough, ["property_type","borough"])
        ])

        preprocessor = Pipeline([('transformer',transformer),
                                 ('svd',truncated_svd)]) if use_svd else Pipeline([('transformer',transformer)])

        return preprocessor

    def save_preprocessor(self, pipeline: Pipeline,  file_path:str)->None:
        with open(file_path, 'wb') as file:
            pickle.dump(pipeline, file)


    def load_preprocessor(self, file_path:str):
        with open(file_path, 'rb') as file:
            loaded_preprocessor = pickle.load(file)

        return loaded_preprocessor



















