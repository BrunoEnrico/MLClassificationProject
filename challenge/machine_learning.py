import pandas as pd
import plotly.express as px
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder



class MachineLearning():
    def __init__(self):
        pass

    def plot_histogram(self, data: pd.DataFrame, column: str, target_column: str, **kwargs) -> None:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data.")
        
        px.histogram(data, x = column, color = target_column, **kwargs).show()

    def plot_box(self, data: pd.DataFrame, column: str, target_column: str, **kwargs) -> None:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data.")
        
        px.box(data, x = column, color = target_column, **kwargs).show()

    def get_target_column(self, data: pd.DataFrame, target: str) -> pd.Series:
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data.")

        return data[target]

    def drop_column(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data.")

        return data.drop(column, axis=1)

    def get_one_hot(self, columns: list) -> ColumnTransformer:
        one_hot = make_column_transformer((
            OneHotEncoder(drop='if_binary'),
            columns),
            remainder='passthrough',
            sparse_threshold=0)
        return one_hot

    def one_hot_transform_data(self, one_hot: ColumnTransformer, data: pd.DataFrame):
        return one_hot.fit_transform(data)

    def dummy_columns(self, data: pd.DataFrame, columns: list, columns_to_dummy: list) -> pd.DataFrame:
        one_hot = self.get_one_hot(columns=columns_to_dummy)
        data = self.one_hot_transform_data(one_hot=one_hot, data=data)
        return pd.DataFrame(data, columns=one_hot.get_feature_names_out(columns))
    
    def label_encoder_transform_data(self, label_encoder: LabelEncoder, data: pd.Series):
        return label_encoder.fit_transform(data)
    
    def dummy_column(self, data: pd.Series):
        label_encoder = LabelEncoder()
        return self.label_encoder_transform_data(label_encoder, data)
        

