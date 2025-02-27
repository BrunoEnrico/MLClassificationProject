import pandas as pd
import plotly.express as px

pd.set_option('display.max_columns', 200)

dados = pd.read_csv('churn.csv')

dados.info()

dados.drop(columns='id_cliente')

px.histogram(dados, x = 'pais', color = 'churn').show()

px.histogram(dados, x = 'sexo_biologico', color = 'churn').show()

px.histogram(dados, x = 'tem_cartao_credito', color = 'churn').show()

px.histogram(dados, x = 'membro_ativo', color = 'churn', barmode='group').show()

px.box(dados, x = 'salario_estimado', color = 'churn').show()

px.box(dados, x = 'saldo', color = 'churn').show()

px.box(dados, x = 'servicos_adquiridos', color = 'churn').show()