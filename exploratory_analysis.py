import pandas as pd
import plotly.express as px
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pickle


class Analyzer:
    def __init__(self):
        pass

    @staticmethod
    def histogram(data: pd.DataFrame, column: str, target_column: str) -> None:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data.")

        px.histogram(data, x = column, text_auto=True, color = target_column, barmode = 'group').show()

    @staticmethod
    def get_target_column(data: pd.DataFrame, target: str) -> pd.Series:
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data.")

        return data[target]

    @staticmethod
    def remove_target_column(data: pd.DataFrame, target: str) -> pd.DataFrame:
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data.")

        return data.drop(target, axis=1)

    @staticmethod
    def make_column_transformer(columns: list) -> ColumnTransformer:
        one_hot = make_column_transformer((
            OneHotEncoder(drop = 'if_binary',),
            columns),
            remainder= 'passthrough',
            sparse_threshold=0
            )
        return one_hot

    @staticmethod
    def fit_transform_data(one_hot: ColumnTransformer, data: pd.DataFrame):
        return one_hot.fit_transform(data)

    @staticmethod
    def dummy_target_column(target_column: pd.Series):
        label_encoder = LabelEncoder()
        return label_encoder.fit_transform(target_column)

    @staticmethod
    def split_test_data(data: pd.DataFrame, target: pd.DataFrame, **kwargs):
        return train_test_split(data, target, **kwargs)

    @staticmethod
    def get_dummy_fit(data: pd.DataFrame, target: pd.DataFrame):
        dummy = DummyClassifier()
        return dummy.fit(data, target)

    @staticmethod
    def get_model_score(dummy: DummyClassifier, data: pd.DataFrame, target: pd.DataFrame):
        return dummy.score(data, target)

    @staticmethod
    def get_tree_fit(data: pd.DataFrame, target: pd.DataFrame, max_depth: int = None):
        tree = DecisionTreeClassifier(random_state=5, max_depth=max_depth)
        return tree.fit(data, target)

    @staticmethod
    def get_tree_predict(tree: DecisionTreeClassifier, data: pd.DataFrame):
        return tree.predict(data)

    @staticmethod
    def get_tree_score(tree: DecisionTreeClassifier, data: pd.DataFrame, target: pd.DataFrame):
        return tree.score(data, target)

    @staticmethod
    def plot_figure(tree: DecisionTreeClassifier, class_names: list, feature_names: list, **kwargs):
        plt.figure(figsize = (15, 6))
        plot_tree(tree, class_names = class_names, feature_names = feature_names, **kwargs)
        plt.show()

    @staticmethod
    def get_min_max() -> MinMaxScaler:
        return MinMaxScaler()

    @staticmethod
    def minmax_transform(minmax: MinMaxScaler, data: pd.DataFrame):
        return minmax.transform(data)

    @staticmethod
    def minmax_fit_transform(minmax: MinMaxScaler, data: pd.DataFrame) -> pd.DataFrame:
        normalized_data = minmax.fit_transform(data)
        return pd.DataFrame(normalized_data)

    @staticmethod
    def get_knn_fit(data: pd.DataFrame, target: pd.DataFrame) -> KNeighborsClassifier:
        knn = KNeighborsClassifier()
        return knn.fit(data, target)

    @staticmethod
    def get_knn_score(knn: KNeighborsClassifier, data: pd.DataFrame, target: pd.DataFrame):
        return knn.score(data, target)

    @staticmethod
    def pickle_dump(name: str, dump):
        with open(f"model_{name}.pkl", 'wb') as file:
            # noinspection PyTypeChecker
            pickle.dump(dump, file)


    def dummy_dataframe_columns(self, data: pd.DataFrame, columns_to_dummy: list):
        one_hot = self.make_column_transformer(columns=columns_to_dummy)
        data = self.fit_transform_data(one_hot, data)
        return one_hot, data

    @staticmethod
    def convert_onehot_to_dataframe(data, one_hot, columns: pd.Index):
        return pd.DataFrame(data, columns=one_hot.get_feature_names_out(columns))
