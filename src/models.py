import numpy as np

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


def crear_linea_regresion_lineal(x, y):
    x = x.values
    x = x.reshape(-1, 1)

    modelo = LinearRegression()
    modelo.fit(x, y)

    x_regresion = np.linspace(x.min(), x.max(), 100)
    y_regresion = modelo.predict(x_regresion.reshape(-1, 1))

    return x_regresion, y_regresion


def crear_linea_k_vecinos(x, y, n_vecinos):
    x = x.values
    x = x.reshape(-1, 1)

    modelo = KNeighborsRegressor(n_vecinos, weights='uniform')
    modelo.fit(x, y)

    x_polinomio = np.linspace(x.min(), x.max(), 300)
    y_polinomio = modelo.predict(x_polinomio.reshape(-1, 1))

    return x_polinomio, y_polinomio


def crear_linea_regresion_logistica(x_train, y_train, x_test, una_variable):

    # Modelo dos variables
    modelo_dos_varaiables = LogisticRegression()
    modelo_dos_varaiables.fit(x_train, y_train)
    prediccion_dos_variables = modelo_dos_varaiables.predict_proba(x_test)

    # Modelo una variables
    x_train = x_train[una_variable].values
    x_train = x_train.reshape(-1, 1)
    x_test_una_variable = x_test[una_variable].values
    x_test_una_variable = x_test_una_variable.reshape(-1, 1)

    modelo_una_variable = LogisticRegression()
    modelo_una_variable.fit(x_train, y_train)
    prediccion_una_variable = modelo_una_variable.predict_proba(x_test_una_variable)

    x_modelo = np.linspace(x_test_una_variable.min(), x_test_una_variable.max(), 300)
    y_modelo = modelo_una_variable.predict_proba(x_modelo.reshape(-1, 1))

    return {'x_prueba': x_test,
            'prediccion_una_variable': prediccion_una_variable,
            'prediccion_dos_variables': prediccion_dos_variables,
            'x_modelo': x_modelo,
            'y_modelo': y_modelo}


def entrena_regresion_lineal(x, y):
    x = x.values
    x = x.reshape(-1, 1)

    modelo = LinearRegression()
    modelo.fit(x, y)

    return modelo


def entrena_k_vecinos(x, y, n_vecinos):
    x = x.values
    x = x.reshape(-1, 1)

    modelo = KNeighborsRegressor(n_vecinos, weights='uniform')
    modelo.fit(x, y)

    return modelo


def calcula_error(tipo_error, modelo, datos_altura_peso):
    x = datos_altura_peso['x_' + tipo_error]
    x = x.values
    x = x.reshape(-1, 1)

    y_true = datos_altura_peso['y_' + tipo_error]
    y_pred = modelo.predict(x)

    return mean_squared_error(y_true, y_pred)


