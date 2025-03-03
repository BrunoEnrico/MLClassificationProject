import pandas as pd
import plotly.express as px
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pickle


class DataProcessor:
    """
    Classe responsável pelo pré-processamento dos dados para análise e modelagem.
    """

    @staticmethod
    def histogram(data: pd.DataFrame, column: str, target_column: str) -> None:
        """
        Plota um histograma para visualizar a distribuição de uma variável.

        :param data: DataFrame contendo os dados
        :param column: Coluna a ser plotada no eixo X
        :param target_column: Coluna de segmentação por cor
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data.")
        px.histogram(data, x=column, text_auto=True, color=target_column, barmode='group').show()

    @staticmethod
    def get_target_column(data: pd.DataFrame, target: str) -> pd.Series:
        """
        Obtém a coluna alvo (target) dos dados.

        :param data: DataFrame contendo os dados
        :param target: Nome da coluna alvo
        :return: Série Pandas contendo os valores da coluna alvo
        """
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data.")
        return data[target]

    @staticmethod
    def remove_target_column(data: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Remove a coluna alvo dos dados.

        :param data: DataFrame original
        :param target: Nome da coluna alvo
        :return: DataFrame sem a coluna alvo
        """
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in data.")
        return data.drop(target, axis=1)

    @staticmethod
    def get_one_hot_transformer(columns: list) -> ColumnTransformer:
        """
        Cria um transformador para converter colunas categóricas em variáveis dummy (One-Hot Encoding).

        :param columns: Lista de colunas categóricas a serem transformadas
        :return: Objeto ColumnTransformer configurado
        """
        return make_column_transformer((OneHotEncoder(drop='if_binary'), columns), remainder='passthrough',
                                       sparse_threshold=0)

    @staticmethod
    def fit_transform_data(one_hot: ColumnTransformer, data: pd.DataFrame):
        """
        Aplica a transformação One-Hot nos dados.

        :param one_hot: Objeto ColumnTransformer configurado
        :param data: DataFrame original
        :return: Dados transformados
        """
        return one_hot.fit_transform(data)

    @staticmethod
    def dummy_target_column(target_column: pd.Series):
        """
        Converte a coluna alvo em valores numéricos utilizando Label Encoding.

        :param target_column: Série Pandas com os valores da coluna alvo
        :return: Série transformada
        """
        label_encoder = LabelEncoder()
        return label_encoder.fit_transform(target_column)

    @staticmethod
    def get_min_max() -> MinMaxScaler:
        """
        Retorna um objeto MinMaxScaler para normalização dos dados.
        """
        return MinMaxScaler()

    @staticmethod
    def minmax_transform(minmax: MinMaxScaler, data: pd.DataFrame):
        """
        Aplica a transformação Min-Max nos dados.

        :param minmax: Objeto MinMaxScaler
        :param data: DataFrame com os dados
        :return: Dados normalizados
        """
        return minmax.transform(data)

    @staticmethod
    def convert_onehot_to_dataframe(data: pd.DataFrame, one_hot, columns: pd.Index) -> pd.DataFrame:
        """
        Converte os dados transformados por One-Hot Encoding de volta para um DataFrame Pandas.

        :param data: Dados transformados
        :param one_hot: Transformador One-Hot Encoding
        :param columns: Índices das colunas originais
        :return: DataFrame transformado
        """
        return pd.DataFrame(data, columns=one_hot.get_feature_names_out(columns))

    @staticmethod
    def plot_figure(tree: DecisionTreeClassifier, class_names: list, feature_names: list, **kwargs):
        """
        Plota a árvore de decisão treinada.

        :param tree: Modelo treinado de DecisionTreeClassifier
        :param class_names: Nomes das classes
        :param feature_names: Nomes das features do modelo
        """
        plt.figure(figsize=(15, 6))
        plot_tree(tree, class_names=class_names, feature_names=feature_names, **kwargs)
        plt.show()

    @staticmethod
    def pickle_dump(name: str, dump):
        """
        Salva um objeto treinado em um arquivo pickle.

        :param name: Nome do arquivo
        :param dump: Objeto a ser salvo
        """
        with open(f"model_{name}.pkl", 'wb') as file:
            pickle.dump(dump, file)


class ModelTrainer:
    """
    Classe responsável pelo treinamento dos modelos de Machine Learning.
    """

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
