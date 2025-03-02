import pandas as pd
import plotly.express as px
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

class MachineLearning:
    def __init__(self):
        pass

    @staticmethod
    def plot_histogram(data: pd.DataFrame, column: str, target_column: str, **kwargs) -> None:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data.")
        
        px.histogram(data, x = column, color = target_column, **kwargs).show()

    @staticmethod
    def plot_box(data: pd.DataFrame, column: str, target_column: str, **kwargs) -> None:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data.")
        
        px.box(data, x = column, color = target_column, **kwargs).show()

    @staticmethod
    def get_target_column(data: pd.DataFrame, target: str) -> pd.Series:
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data.")

        return data[target]

    @staticmethod
    def drop_column(data: pd.DataFrame, column: str) -> pd.DataFrame:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data.")

        return data.drop(column, axis=1)

    @staticmethod
    def get_one_hot(columns: list) -> ColumnTransformer:
        one_hot = make_column_transformer((
            OneHotEncoder(drop='if_binary'),
            columns),
            remainder='passthrough',
            sparse_threshold=0)
        return one_hot

    @staticmethod
    def one_hot_transform_data(one_hot: ColumnTransformer, data: pd.DataFrame):
        return one_hot.fit_transform(data)

    @staticmethod
    def label_encoder_transform_data(label_encoder: LabelEncoder, data: pd.Series):
        return label_encoder.fit_transform(data)

    @staticmethod
    def split_test_data(data: pd.DataFrame, target: pd.DataFrame, **kwargs):
        return train_test_split(data, target, **kwargs)

    @staticmethod
    def get_dummy_fit(data: pd.DataFrame, target: pd.DataFrame):
        dummy = DummyClassifier()
        return dummy.fit(data, target)

    @staticmethod
    def get_dummy_score(dummy: DummyClassifier, data: pd.DataFrame, target: pd.DataFrame):
        return dummy.score(data, target)

    @staticmethod
    def get_tree_fit(data: pd.DataFrame, target: pd.DataFrame, **kwargs):
        tree = DecisionTreeClassifier(random_state=5, **kwargs)
        return tree.fit(data, target)

    @staticmethod
    def get_tree_predict(tree: DecisionTreeClassifier, data: pd.DataFrame):
        return tree.predict(data)

    @staticmethod
    def get_tree_score(tree: DecisionTreeClassifier, data: pd.DataFrame, target: pd.DataFrame):
        return tree.score(data, target)

    @staticmethod
    def plot_results(tree: DecisionTreeClassifier, class_names: list, **kwargs):
        plt.figure(figsize=(15, 7))
        plot_tree(tree, class_names=class_names, **kwargs)
        plt.show()

    def dummy_columns(self, data: pd.DataFrame, columns: list, columns_to_dummy: list) -> pd.DataFrame:
        one_hot = self.get_one_hot(columns=columns_to_dummy)
        data = self.one_hot_transform_data(one_hot=one_hot, data=data)
        return pd.DataFrame(data, columns=one_hot.get_feature_names_out(columns))

    def dummy_column(self, data: pd.Series):
        label_encoder = LabelEncoder()
        return self.label_encoder_transform_data(label_encoder, data)
        

