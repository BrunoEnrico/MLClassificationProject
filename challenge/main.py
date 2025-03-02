import pandas as pd
from machine_learning import MachineLearning


class Main:
    def __init__(self):
        pd.set_option('display.max_columns', 200)
        self.data = pd.read_csv('churn.csv')
    
    def process(self):
        ml = MachineLearning()
        target = ml.get_target_column(self.data, "churn")
        data = ml.drop_column(self.data, "id_cliente")
        data = ml.drop_column(data, "churn")
        data = ml.dummy_columns(data, data.columns,
                         ["pais", "sexo_biologico",
                                        'tem_cartao_credito', 'membro_ativo'])
        target = ml.dummy_column(target)

        data_train, data_test, target_train, target_test = ml.split_test_data(data, target, stratify = target, random_state = 5)
        dummy = ml.get_dummy_fit(data_train, target_train)
        score = ml.get_dummy_score(dummy, data_test, target_test)
        print(f"The score of the dummy model is {score}")

        tree = ml.get_tree_fit(data_train, target_train, max_depth=4)
        ml.get_tree_predict(tree, data_test)
        score = ml.get_tree_score(tree, data_test, target_test)
        print(f"The score of the tree model is {score}")

        ml.plot_results(tree, class_names=['não', 'sim'], fontsize = 5, filled = True)


if __name__ == '__main__':
    main = Main()
    main.process()
        