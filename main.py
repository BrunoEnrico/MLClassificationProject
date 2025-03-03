import pandas as pd
from exploratory_analysis import DataProcessor, ModelTrainer


class MLProject:
    def __init__(self):

        # Carrega os dados do arquivo CSV
        self.data = pd.read_csv('marketing_investimento.csv')

    def process(self):
        # Obtém a coluna alvo
        target = DataProcessor.get_target_column(self.data, "aderencia_investimento")

        # Remove a coluna alvo dos dados para separação entre features e target
        data = DataProcessor.remove_target_column(self.data, target=str(target.name))

        # Obtém as colunas do dataset
        columns = data.columns

        # Aplica One-Hot Encoding às colunas categóricas
        one_hot = DataProcessor.get_one_hot_transformer(
            columns=["estado_civil", "escolaridade", "inadimplencia", "fez_emprestimo"])
        data = DataProcessor.fit_transform_data(one_hot, data)
        data = DataProcessor.convert_onehot_to_dataframe(data, one_hot, columns)

        # Converte a variável alvo para valores numéricos
        target = DataProcessor.dummy_target_column(target)

        # Divide os dados em conjunto de treino e teste
        data_train, data_test, target_train, target_test = ModelTrainer.split_test_data(data, target, stratify=target,
                                                                                        random_state=5)

        # Treina um classificador Dummy e avalia a performance
        dummy = ModelTrainer.get_dummy_fit(data_train, target_train)
        result = ModelTrainer.get_model_score(dummy, data_test, target_test)
        print(f"The result of the classification dummy model was {result}")

        # Treina um modelo de árvore de decisão e avalia a performance
        tree = ModelTrainer.get_tree_fit(data_train, target_train)
        ModelTrainer.get_tree_predict(tree, data_test)
        result = ModelTrainer.get_tree_score(tree, data_test, target_test)
        print(f"The result of the classification tree model was {result}")

        # Avalia a performance da árvore no conjunto de treino
        score = ModelTrainer.get_tree_score(tree, data_train, target_train)
        print(f"The score of the decision tree was {score}")

        # Treina uma árvore com profundidade limitada a 3 níveis
        tree = ModelTrainer.get_tree_fit(data_train, target_train, max_depth=3)
        ModelTrainer.get_tree_predict(tree, data_test)
        result = ModelTrainer.get_tree_score(tree, data_test, target_test)
        print(f"The result of the classification tree model was {result}")

        # Define os nomes das colunas para visualização da árvore de decisão
        column_names = ['casado (a)', 'divorciado (a)', 'solteiro (a)', 'fundamental', 'medio', 'superior',
                        'inadimplencia', 'fez_emprestimo', 'idade', 'saldo', 'tempo_ult_contato', 'numero_contatos']

        # Plota a árvore de decisão treinada
        DataProcessor.plot_figure(tree, class_names=['não', 'sim'], feature_names=column_names, fontsize=1, filled=True)

        # Avalia novamente a árvore de decisão no conjunto de treino
        score = ModelTrainer.get_tree_score(tree, data_train, target_train)
        print(f"The score of the decision tree was {score}")

        # Normaliza os dados de treino
        minmax = DataProcessor.get_min_max()
        normalized_data_train = ModelTrainer.minmax_fit_transform(minmax, data_train)

        # Treina um modelo KNN com os dados normalizados
        knn = ModelTrainer.get_knn_fit(normalized_data_train, target_train)

        # Normaliza os dados de teste e avalia o modelo KNN
        normalized_data_test = DataProcessor.minmax_transform(minmax, data_test)
        knn_score = ModelTrainer.get_knn_score(knn, normalized_data_test, target_test)
        print(f"The score of the KNN Model was {knn_score}")

        # Salva os modelos treinados usando pickle
        DataProcessor.pickle_dump("onehot", one_hot)
        DataProcessor.pickle_dump("tree", tree)

        # Carrega os modelos salvos
        one_hot_pickle = pd.read_pickle('model_onehot.pkl')
        tree_pickle = pd.read_pickle('model_tree.pkl')

        # Testa o modelo com novos dados
        novo_dado = pd.DataFrame({
            'idade': [45],
            'estado_civil': ['solteiro (a)'],
            'escolaridade': ['superior'],
            'inadimplencia': ['nao'],
            'saldo': [23040],
            'fez_emprestimo': ['nao'],
            'tempo_ult_contato': [800],
            'numero_contatos': [4]
        })

        # Transforma os novos dados com o OneHotEncoder treinado
        novo_dado = one_hot_pickle.transform(novo_dado)

        # Faz a previsão usando a árvore de decisão treinada
        result = tree_pickle.predict(novo_dado)
        print(f"The model prediction for the new data was {result[0]}")


if __name__ == "__main__":
    ml = MLProject()
    ml.process()
