from sklearn.ensemble import RandomForestRegressor
from pandas import read_csv, to_datetime, concat, DataFrame
from matplotlib.pyplot import subplots
from seaborn import despine, displot, lineplot, set
set(style="whitegrid")


def to_date(dados):
    return to_datetime(dados.year * 10000 + dados.month * 100 + 1, format='%Y%m%d')


def main():
    path = 'datasets/'

    dados_treino = read_csv(path + 'train new.csv')
    dados_teste = read_csv(path + 'test new.csv')

    dados_teste['date'] = to_date(dados_teste)
    dados_treino['date'] = to_date(dados_treino)

    dados_todos = concat([dados_teste, dados_treino])


    # Proporção de dados de produção
    # displot(dados_todos.production, color='g', kind='kde').figure.savefig('graphs/distribution_production.png')


    # Média de produção
    # print(dados_todos.production.mean())


    # Normalização conforme a média
    # dados_todos = dados_todos[(dados_todos.production < (dados_todos.production.mean() +dados_todos.production.std()*4)) | (dados_todos.production.isna())]


    # Gráfico de produção conforme os anos
    figura, eixos = subplots(figsize=(12, 10))
    lineplot(x='date', y='production', data=dados_todos, ax=eixos).figure.savefig('graphs/production_by_year.png')


    # Gráfico de produção entre os meses
    # figura, eixos = subplots(figsize=(12, 10))
    # lineplot(x='month', y='production', data=dados_todos, ax=eixos).figure.savefig('graphs/production_by_month.png')


    # Grafico de produção de acordo com a idade da árvore
    # figura, eixos = subplots(figsize=(12, 10))
    # lineplot(x='Precipitation', y='production', data=dados_todos, ax=eixos).figure.savefig('graphs/production_by_age.png')


    # Verificar importância das colunas
    modelo = RandomForestRegressor()

    x = dados_treino.drop(columns=['Id', 'date', 'production'])
    y = dados_treino.production

    modelo.fit(x, y)

    print(DataFrame(modelo.feature_importances_, index = x.columns, columns=['importance']).sort_values('importance', ascending=False).reset_index())


if __name__ == '__main__':
    main()