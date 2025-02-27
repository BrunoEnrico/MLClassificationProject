import pandas as pd
import plotly.express as px
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class Analyzer():
    def __init__(self):
        pass

    def histogram(self, data: pd.DataFrame, column: str, target_column: str) -> None:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data.")
        
        px.histogram(data, x = column, text_auto=True, color = target_column, barmode = 'group').show()


    def get_target_column(self, data: pd.DataFrame, target: str) -> pd.Series:
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data.")

        return data[target]
    
    def remove_target_column(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data.")

        return data.drop(target, axis=1)
    
    def make_column_transformer(self, columns: list) -> ColumnTransformer:
        one_hot = make_column_transformer((
            OneHotEncoder(drop = 'if_binary',),
            columns),
            remainder= 'passthrough',
            sparse_threshold=0
            )
        return one_hot

    def fit_transform_data(self, one_hot: ColumnTransformer, data: pd.DataFrame):
        return one_hot.fit_transform(data)
    
    def dummy_target_column(self, target_column: pd.Series):
        label_encoder = LabelEncoder()
        return label_encoder.fit_transform(target_column)

    def dummy_dataframe_columns(self, data: pd.DataFrame, columns: pd.Index, columns_to_dummy: list) -> pd.DataFrame:
        one_hot = self.make_column_transformer(columns=columns_to_dummy)
        data = self.fit_transform_data(one_hot, data)
        return pd.DataFrame(data,
                            columns=one_hot.get_feature_names_out(columns))
