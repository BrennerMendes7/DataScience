# pip install pandas, numpy, openpyxl, matplotlib, seaborn, scikit-learn
import pandas as pd
import seaborn as sns # biblioteca de gráfico
import matplotlib.pyplot as plt # biblioteca de gráfico
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression # IA
from sklearn.ensemble import RandomForestRegressor # IA
from sklearn import metrics # porcentagem (R ao quadrado) de desempenho da IA

# Passo 1: Extração/Obtenção de dados
tabela = pd.read_csv("advertising.csv") # ler arquivo csv

# Passo 2: Ajuste de dados (tratamento)
print(tabela.info()) # mostrar as informações da tabela

# Passo 3: Análise exploratória
sns.heatmap(tabela.corr(), cmap="Wistia", annot=True) # criar gráfico de calor com a correlação dos dados da tabela
plt.show() # exibir o gráfico

# Passo 4: Modelagem + Algoritmos (Aqui entra a inteligência artificial)
# separar os nossos dados (em x e y) em dados de treino e dados de teste
y = tabela["Vendas"] # quem eu quero prever
x = tabela[["TV", "Radio", "Jornal"]] # quem eu vou usar pra fazer a previsão, duplo colchetes para passar mais de uma coluna
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3) # dividir os dados em dados de treino e teste

modelo_regressaolinear = LinearRegression() # criando a IA
modelo_arvoredecisao = RandomForestRegressor() # criando a IA

modelo_regressaolinear.fit(x_treino, y_treino) # fazer a IA treinar com os dados de treino
modelo_arvoredecisao.fit(x_treino, y_treino) # fazer a IA treinar com os dados de treino

# Passo 5: Interpretação de Resultados
# qual é o melhor modelo/IA
previsao_regressaolinear = modelo_regressaolinear.predict(x_teste) # fazer IA (já treinada) montar a previsão com os dados de teste
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste) # fazer IA (já treinada) montar a previsão com os dados de teste

print(metrics.r2_score(y_teste, previsao_regressaolinear)) # calcular e mostrar eficácia da IA
print(metrics.r2_score(y_teste, previsao_arvoredecisao)) # calcular e mostrar eficácia da IA - melhor modelo

# importar nova tabela com as informações de investimento em propaganda em TV, Rádio e Jornal
novos = pd.read_csv("novos.csv") # ler arquivo CSV
print(novos)
# passa a nova tabela para o predict do seu modelo
previsao = modelo_arvoredecisao.predict(novos) # fazer a previsão de vendas com os dados da nova tabela
print(previsao)

