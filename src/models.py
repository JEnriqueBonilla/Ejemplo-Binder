import numpy as np

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor


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
