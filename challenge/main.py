import pandas as pd
from machine_learning import MachineLearning


class Main():
    def __init__(self):
        pd.set_option('display.max_columns', 200)
        self.data = pd.read_csv('challenge/churn.csv')
    
    def process(self):
        ml = MachineLearning()
        target = ml.get_target_column(self.data, "churn")
        data = ml.drop_column(self.data, "id_cliente")
        data = ml.dummy_columns(data, data.columns,
                         ["pais", "sexo_biologico"])
        target = ml.dummy_column(target)
        print(target)


if __name__ == '__main__':
    main = Main()
    main.process()
        