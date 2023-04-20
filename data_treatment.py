from pandas import read_csv, DataFrame, merge, concat
from os import listdir


def get_previous_data(dataframe, prop):
    for x in range(1,13):
        dataframe[f'{prop}_b{x}'] = dataframe[prop].shift(x)
    return dataframe


def main():
    train_data = read_csv('datasets/train.csv')
    test_data = read_csv('datasets/test.csv')

    # Remoção de dados dos anos de 2004-2005 devido à divergências no padrão 
    # train_data = train_data[train_data.year >= 2006]

    # Normalização conforme a média
    # train_data = train_data[(train_data.production < (train_data.production.mean() +train_data.production.std()*4)) | (train_data.production.isna())]

    # Dados do solo não serão utilizados devido a pouca relevância
    # soil_data = read_csv('datasets/soil_data.csv')
    
    # train_data = merge(train_data, soil_data, on='field', how='left')
    # test_data = merge(test_data, soil_data, on='field', how='left')

    # Concatenação dos campos em um só DataFrame
    props = ['temperature','dewpoint','windspeed','Soilwater_L1','Precipitation']
    field_data = read_csv(f'datasets/field-0.csv')
    field_data['field'] = 0
    for prop in props:
        field_data = get_previous_data(field_data, prop)

    for x, field in enumerate(filter(lambda x: 'field-' in x and 'field-0' not in x, listdir('datasets/'))):
        new_field_data = read_csv(f'datasets/{field}')
        for prop in props:
            new_field_data = get_previous_data(new_field_data, prop)

        new_field_data['field'] = x + 1

        field_data = concat([field_data, new_field_data])

    # Colunas com dados muito semelhantes a Soilwater_L1, então não tem pq utilizá-las
    # field_data = field_data.drop(columns=['Soilwater_L2', 'Soilwater_L3', 'Soilwater_L4'])

    train_data = merge(train_data, field_data, on=['field', 'year', 'month'], sort=True, how='left')
    train_data.sort_values('Id').to_csv('datasets/train new.csv', index=False)

    test_data = merge(test_data, field_data, on=['field', 'year', 'month'], sort=True, how='left')
    test_data.sort_values('Id').to_csv('datasets/test new.csv', index=False)


if __name__ == '__main__':
    main()