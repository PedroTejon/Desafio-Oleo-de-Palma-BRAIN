from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from pandas import read_csv, DataFrame
from seaborn import jointplot, set
from sklearn.ensemble import RandomForestRegressor
from pandas import read_csv, to_datetime, concat, DataFrame
from matplotlib.pyplot import subplots
from seaborn import lineplot, set

set(style="whitegrid")


def to_date(dados):
    return to_datetime(dados.year * 10000 + dados.month * 100 + 1, format='%Y%m%d')


def analise_propria(dados_treino, colunas):
    # Divisão dos dados
    x = dados_treino[colunas]
    y = dados_treino['production']

    # Normalização dos dados
    x = StandardScaler().fit_transform(x)

    # Divisão de dados para treino e teste
    x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=1, shuffle=False)
    modelo = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=1)
    modelo.fit(x_treino, y_treino)
    previsao = modelo.predict(x_teste)

    # Calculo de pontuação
    pontuacao_media = mean_absolute_error(y_teste, previsao)
    print('Score:', pontuacao_media)

    # Plot de tendência comparada com o resultado da predição 
    jointplot(x=y_teste, y=previsao, kind="reg", color="m", height=7).figure.savefig('graphs/resultado_previsao.png')


def previsao_kaggle(dados_treino, dados_teste, colunas):
    # Treinamento do modelo
    modelo = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=1)
    modelo.fit(dados_treino[colunas],  dados_treino['production'])
    dados_teste['production'] = modelo.predict(dados_teste[colunas])

    # Criação do Gráfico para análise
    dados_teste['date'] = to_date(dados_teste)
    dados_treino['date'] = to_date(dados_treino)
    dados_todos = concat([dados_treino, dados_teste])
    _, eixos = subplots(figsize=(12, 10))
    lineplot(x='date', y='production', data=dados_todos, ax=eixos).figure.savefig('graphs/production_by_year.png')

    return dados_teste


def main():
    dados_treino = read_csv('datasets/train new.csv')
    dados_teste = read_csv('datasets/test new.csv')

    props = ['field','age','type','year','month','temperature','dewpoint','windspeed','Soilwater_L1', 'Soilwater_L2', 'Soilwater_L3', 'Soilwater_L4','Precipitation','temperature_b1','temperature_b2','temperature_b3','temperature_b4','temperature_b5','temperature_b6','temperature_b7','temperature_b8','temperature_b9','temperature_b10','temperature_b11','temperature_b12','dewpoint_b1','dewpoint_b2','dewpoint_b3','dewpoint_b4','dewpoint_b5','dewpoint_b6','dewpoint_b7','dewpoint_b8','dewpoint_b9','dewpoint_b10','dewpoint_b11','dewpoint_b12','windspeed_b1','windspeed_b2','windspeed_b3','windspeed_b4','windspeed_b5','windspeed_b6','windspeed_b7','windspeed_b8','windspeed_b9','windspeed_b10','windspeed_b11','windspeed_b12','Soilwater_L1_b1','Soilwater_L1_b2','Soilwater_L1_b3','Soilwater_L1_b4','Soilwater_L1_b5','Soilwater_L1_b6','Soilwater_L1_b7','Soilwater_L1_b8','Soilwater_L1_b9','Soilwater_L1_b10','Soilwater_L1_b11','Soilwater_L1_b12','Precipitation_b1','Precipitation_b2','Precipitation_b3','Precipitation_b4','Precipitation_b5','Precipitation_b6','Precipitation_b7','Precipitation_b8','Precipitation_b9','Precipitation_b10','Precipitation_b11','Precipitation_b12']
    previsoes = previsao_kaggle(dados_treino, dados_teste, props)
    analise_propria(dados_treino, props)

    res_submissao = DataFrame({'Id': previsoes.Id, 'production': previsoes.production})
    res_submissao.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()