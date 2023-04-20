from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from pandas import read_csv, get_dummies, DataFrame
from numpy import linspace


def main():
    dados_treino = read_csv('train new.csv')

    resultados = dados_treino['production']
    dados_treino = dados_treino[['month','age','BDRICM_BDRICM_M','BDRLOG_BDRLOG_M','BDTICM_BDTICM_M','BLDFIE_sl1','BLDFIE_sl2','BLDFIE_sl3','BLDFIE_sl4','CECSOL_sl1','CECSOL_sl2','CECSOL_sl3','CECSOL_sl4','CLYPPT_sl1','CLYPPT_sl2','CLYPPT_sl3','CLYPPT_sl4','CRFVOL_sl1','CRFVOL_sl2','CRFVOL_sl3','CRFVOL_sl4','OCSTHA_sd1','OCSTHA_sd2','OCSTHA_sd3','OCSTHA_sd4','ORCDRC_sl1','ORCDRC_sl2','ORCDRC_sl3','ORCDRC_sl4','PHIHOX_sl1','PHIHOX_sl2','PHIHOX_sl3','PHIHOX_sl4','PHIKCL_sl1','PHIKCL_sl2','PHIKCL_sl3''SLTPPT_sl1','SLTPPT_sl2','SLTPPT_sl3','SLTPPT_sl4','SNDPPT_sl1','SNDPPT_sl2','SNDPPT_sl3','SNDPPT_sl4','temperature','dewpoint','windspeed','Precipitation']]
    dados_treino_categorizados = get_dummies(dados_treino)

    n_estimators = [x for x in range(100, 2100, 100)]
    max_features = ['sqrt', 'log2', None]
    max_depth = [x for x in range(10, 120, 10)] + [None]
    min_samples_split = [2, 5, 7, 10, 12]
    min_samples_leaf = [1, 2, 4, 8]
    criterion = ['poisson', 'squared_error', 'absolute_error', 'friedman_mse']
    bootstrap = [True, False]

    parametros = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap,
                'criterion': criterion}

    modelo = RandomForestRegressor()
    rf_random = RandomizedSearchCV(estimator=modelo, param_distributions=parametros, n_iter=100, cv=5, verbose=2, random_state=1, n_jobs=-1)
    rf_random.fit(dados_treino_categorizados, resultados)

    print(rf_random.best_params_)


if __name__ == '__main__':
    main()