import pandas as pd
import plotly.express as px

dados = pd.read_csv('marketing_investimento.csv')

dados.info()

px.histogram(dados, x = 'estado_civil', text_auto=True, color = 'aderencia_investimento', barmode = 'group').show()

px.histogram(dados, x = 'escolaridade', text_auto=True, color = 'aderencia_investimento', barmode = 'group').show()

px.histogram(dados, x = 'inadimplencia', text_auto=True, color = 'aderencia_investimento', barmode = 'group').show()

px.histogram(dados, x = 'fez_emprestimo', text_auto=True, color = 'aderencia_investimento', barmode = 'group').show()


px.box(dados, x = 'idade', color = 'aderencia_investimento').show()

px.box(dados, x = 'saldo', color = 'aderencia_investimento').show()

px.box(dados, x = 'tempo_ult_contato', color = 'aderencia_investimento').show()

px.box(dados, x = 'numero_contatos', color = 'aderencia_investimento').show()
