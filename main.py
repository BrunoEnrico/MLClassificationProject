import pandas as pd
from exploratory_analysis import Analyzer

class MLProject:
    def __init__(self):
        self.data = pd.read_csv('marketing_investimento.csv')
    
    def process(self):
        analyzer = Analyzer()
        target = analyzer.get_target_column(self.data, "aderencia_investimento")
        data = analyzer.remove_target_column(self.data, target= str(target.name))
        columns = data.columns
        one_hot, data = analyzer.dummy_dataframe_columns(data,
                                                columns_to_dummy=["estado_civil",
                                                                  "escolaridade",
                                                                  "inadimplencia",
                                                                  "fez_emprestimo"])
        data = analyzer.convert_onehot_to_dataframe(data, one_hot, columns)
        target = analyzer.dummy_target_column(target)
        data_train, data_test, target_train, target_test = analyzer.split_test_data(data, target, stratify = target, random_state = 5)
        dummy = analyzer.get_dummy_fit(data_train, target_train)
        result = analyzer.get_model_score(dummy, data_test, target_test)
        print(f"The result of the classification dummy model was {result}")

        tree = analyzer.get_tree_fit(data_train, target_train)
        analyzer.get_tree_predict(tree, data_test)
        result = analyzer.get_tree_score(tree, data_test, target_test)
        print(f"The result of the classification tree model was {result}")

        # column_names = ['casado (a)',
        #                 'divorciado (a)',
        #                 'solteiro (a)',
        #                 'fundamental',
        #                 'medio',
        #                 'superior',
        #                 'inadimplencia',
        #                 'fez_emprestimo',
        #                 'idade',
        #                 'saldo',
        #                 'tempo_ult_contato',
        #                 'numero_contatos']

        #analyzer.plot_figure(tree, class_names=['não', 'sim'], feature_names=column_names, fontsize = 1, filled = True)
        score = analyzer.get_tree_score(tree, data_train, target_train)
        print(f"The score of the decision tree was {score}")

        tree = analyzer.get_tree_fit(data_train, target_train, max_depth=3)
        analyzer.get_tree_predict(tree, data_test)
        result = analyzer.get_tree_score(tree, data_test, target_test)
        print(f"The result of the classification tree model was {result}")

        score = analyzer.get_tree_score(tree, data_train, target_train)
        print(f"The score of the decision tree was {score}")

        minmax = analyzer.get_min_max()
        normalized_data_train = analyzer.minmax_fit_transform(minmax, data_train)
        knn = analyzer.get_knn_fit(normalized_data_train, target_train)
        normalized_data_test = analyzer.minmax_transform(minmax, data_test)
        knn_score = analyzer.get_knn_score(knn, normalized_data_test, target_test)
        print(f"The score of the KNN Model was {knn_score}")

        #analyzer.pickle_dump("onehot", one_hot)
        #analyzer.pickle_dump("tree", tree)
        one_hot_pickle = pd.read_pickle('model_onehot.pkl')
        tree_pickle = pd.read_pickle('model_tree.pkl')

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
        novo_dado = pd.DataFrame(novo_dado)
        novo_dado = one_hot_pickle.transform(novo_dado)
        result = tree_pickle.predict(novo_dado)
        print(f"The model prediction for the new data was {result[0]}")





if __name__ == "__main__":
    ml = MLProject()
    ml.process()


