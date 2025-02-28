import pandas as pd
from exploratory_analysis import Analyzer

class MLProject:
    def __init__(self):
        self.data = pd.read_csv('marketing_investimento.csv')
    
    def process(self):
        analyzer = Analyzer()
        target = analyzer.get_target_column(self.data, "aderencia_investimento")
        data = analyzer.remove_target_column(self.data, target= str(target.name))
        data = analyzer.dummy_dataframe_columns(data,
                                                columns=data.columns,
                                                columns_to_dummy=["estado_civil",
                                                                  "escolaridade",
                                                                  "inadimplencia",
                                                                  "fez_emprestimo"])
        target = analyzer.dummy_target_column(target)
        data_train, data_test, target_train, target_test = analyzer.split_test_data(data, target, stratify = target, random_state = 5)
        dummy = analyzer.get_dummy_fit(data_train, target_train)
        result = analyzer.get_model_score(dummy, data_test, target_test)
        print(f"The result of the classification dummy model was {result}")

        tree = analyzer.get_tree_fit(data_train, target_train)
        analyzer.get_tree_predict(tree, data_test)
        result = analyzer.get_tree_score(tree, data_test, target_test)
        print(f"The result of the classification tree model was {result}")

        column_names = ['casado (a)',
                        'divorciado (a)',
                        'solteiro (a)',
                        'fundamental',
                        'medio',
                        'superior',
                        'inadimplencia',
                        'fez_emprestimo',
                        'idade',
                        'saldo',
                        'tempo_ult_contato',
                        'numero_contatos']

        #analyzer.plot_figure(tree, class_names=['não', 'sim'], feature_names=column_names, fontsize = 1, filled = True)
        score = analyzer.get_tree_score(tree, data_train, target_train)
        print(f"The score of the decision tree was {score}")

        tree = analyzer.get_tree_fit(data_train, target_train, max_depth=3)
        analyzer.get_tree_predict(tree, data_test)
        result = analyzer.get_tree_score(tree, data_test, target_test)
        print(f"The result of the classification tree model was {result}")

        score = analyzer.get_tree_score(tree, data_train, target_train)
        print(f"The score of the decision tree was {score}")




if __name__ == "__main__":
    ml = MLProject()
    ml.process()


