import os
import pandas as pd
from typing import List


class Wrangling:

    def __init__(self, file_path: str):

        """
        Initializes the Wrangling object with the specified file path.

        Parameters:
            - file_path (str): The path to the directory containing data files.
        """

        self.file_path = file_path

    def load_data(self, data:str)->pd.DataFrame:

        """
        Loads and combines multiple CSV files from the 'data' subdirectory and returns a pandas DataFrame.

        Parameters:
            - data (str): Placeholder parameter, not used in the method.

        Returns:
            - pd.DataFrame: A combined DataFrame containing data from all CSV files.
        """

        data_dir = os.path.join(self.file_path, 'data')
        csv_files = [file for file in os.listdir(data_dir)]
        df_list_path = []

        for csv_file in csv_files:
            csv_dir = os.path.join(data_dir, csv_file)
            df_list_path.append(csv_dir)

        combined_df = pd.DataFrame()

        for df in df_list_path:
            dataframes = pd.read_csv(df)
            combined_df = pd.concat([dataframes, combined_df], axis=0, ignore_index=True)

        return combined_df

    '''
    @staticmethod
    def drop_unnecessary_columns(df:pd.DataFrame, cols: List[str])->None:

        df.drop(columns = cols, inplace=True)
    '''


    @staticmethod
    def clean_borough(df:pd.DataFrame, existing_col: str, new_col: str)->None:

        '''

        Clean place_with_parent_names by | symbol, taking only the relevant value: the borough

        :param df:
        :param existing_col:
        :param new_col:
        :return:
        '''
        df[new_col] = df[existing_col].str.split('|', expand=True)[1]


    @staticmethod
    def clean_latitude_longitude(df:pd.DataFrame, existing_col:str, new_cols:List[str])->None:

        '''

        Splitting lat-lon into two separatable columns: lat, lon

        :param df:
        :param existing_col:
        :param new_cols:
        :return:
        '''

        df[new_cols] = df[existing_col].str.split(',', expand=True).astype(float)

    '''
    @staticmethod
    def clean_scientific_code(df:pd.DataFrame, cols: List[str])->None:
        if "price" in cols:
            df.apply(lambda col: col.apply(lambda x: format(x, '.15f')))
    '''





