import pandas as pd
from exploratory_analysis import Analyzer

class MLProject():
    def __init__(self):
        self.data = pd.read_csv('marketing_investimento.csv')
    
    def process(self):
        analyzer = Analyzer()
        target = analyzer.get_target_column(self.data, "aderencia_investimento")
        data = analyzer.remove_target_column(self.data, target=target.name)
        data = analyzer.dummy_dataframe_columns(data,
                                                columns=data.columns,
                                                columns_to_dummy=["estado_civil",
                                                                  "escolaridade",
                                                                  "inadimplencia",
                                                                  "fez_emprestimo"])
        target = analyzer.dummy_target_column(target)
        print(target)




if __name__ == "__main__":
    ml = MLProject()
    ml.process()


