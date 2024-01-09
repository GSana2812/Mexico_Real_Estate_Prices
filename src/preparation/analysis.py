import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

class Analysis:

    def load_data(self, path: str)->pd.DataFrame:
        df = pd.read_csv(path).reset_index(drop=True)
        return df

    @staticmethod
    def insights(df:pd.DataFrame)->None:
        print(df.info())
        print(df.describe())

    @staticmethod
    def head(df:pd.DataFrame)->None:
        print(df.head())

    @staticmethod
    def build_histogram(df:pd.DataFrame, col:str, bins = 50)->None:
        df[col].hist(bins= bins, figsize=(5,5))
        plt.show()

    @staticmethod
    def build_coordinates_scatter(df: pd.DataFrame, x:str, y:str)->None:
        df.plot(kind = "scatter", x = x, y = y, grid=True, alpha=0.2)
        plt.show()

    @staticmethod
    def build_coordinates_scatter_range(df: pd.DataFrame, lat:str, lon:str, color:str) ->None:

        fig = px.scatter_mapbox(
            df,
            lat=lat,
            lon=lon,
            width=600,
            height=600,
            color=color,
            hover_data=[color],
            range_color=[50000, 1000000]
        )
        fig.update_layout(mapbox_style="open-street-map")
        fig.show()

    @staticmethod
    def correlation(df: pd.DataFrame)->None:

        sns.heatmap(df.corr(), annot=True, cmap='viridis', fmt=".1f", linewidths=.5)
        plt.show()

    @staticmethod
    def check_null(df: pd.DataFrame)->None:
        print(df.isna().sum())






