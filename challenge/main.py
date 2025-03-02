import pandas as pd
from machine_learning import MachineLearning as ML


class Main:
    def __init__(self):
        pd.set_option('display.max_columns', 200)
        self.data = pd.read_csv('churn.csv')
    
    def process(self):
        target = ML.get_target_column(self.data, "churn")
        data = ML.drop_column(self.data, "id_cliente")
        data = ML.drop_column(data, "churn")
        one_hot = ML.get_one_hot(columns=["pais", "sexo_biologico"])
        data = ML.one_hot_transform_data(one_hot=one_hot, data=data)


        target = ML.dummy_column(target)

        data_train, data_test, target_train, target_test = ML.split_test_data(data, target, stratify = target, random_state = 5)
        dummy = ML.get_dummy_fit(data_train, target_train)
        score = ML.get_dummy_score(dummy, data_test, target_test)
        print(f"The score of the dummy model is {score}")

        tree = ML.get_tree_fit(data_train, target_train, max_depth=4)
        ML.get_tree_predict(tree, data_test)
        score = ML.get_tree_score(tree, data_test, target_test)
        print(f"The score of the tree model is {score}")

        #ML.plot_results(tree, class_names=['não', 'sim'], fontsize = 5, filled = True)

        minmax = ML.get_min_max()
        normalized_data_train = ML.minmax_fit_transform(minmax, data_train)
        knn = ML.get_knn_fit(normalized_data_train, target_train)
        normalized_data_test = ML.minmax_fit_transform(minmax, data_test)
        score = ML.get_knn_score(knn, normalized_data_test, target_test)
        print(f"The KNN model score is {score}")

        #ML.pickle_dump(one_hot, "one_hot")
        #ML.pickle_dump(tree, "tree")

        one_hot_pickle = pd.read_pickle("model_one_hot.pkl")
        tree_model_pickle = pd.read_pickle("model_tree.pkl")

        new_data = pd.DataFrame({
            'score_credito': [850],
            'pais': ['França'],
            'sexo_biologico': ['Homem'],
            'idade': [27],
            'anos_de_cliente': [3],
            'saldo': [56000],
            'servicos_adquiridos': [1],
            'tem_cartao_credito': [1],
            'membro_ativo': [1],
            'salario_estimado': [85270.00]
        })

        normalized_new_data = ML.one_hot_transform(one_hot_pickle, new_data)
        result = tree_model_pickle.predict(normalized_new_data)

        print(f"The prediction for the pickle tree model on the new data was {result}")


if __name__ == '__main__':
    main = Main()
    main.process()
        