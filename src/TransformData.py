import pandas as pd
from sklearn.model_selection import train_test_split

from .models import entrena_regresion_lineal, entrena_k_vecinos, calcula_error


def dividir_datos(x, y, test_size=.2, validation_size=.2):
    """

    :param x:
    :param y:
    :param test_size:
    :param validation_size:
    :return:
    """
    x_entrenamiento, x_prueba, y_entrenamiento, y_prueba = train_test_split(x,
                                                                            y,
                                                                            test_size=test_size,
                                                                            random_state=0)
    split_size = validation_size/(1-validation_size)
    x_entrenamiento, x_validacion, y_entrenamiento, y_validacion = train_test_split(x_entrenamiento,
                                                                                    y_entrenamiento,
                                                                                    test_size=split_size,
                                                                                    random_state=0)
    return {'x_entrenamiento': x_entrenamiento,
            'y_entrenamiento': y_entrenamiento,
            'x_prueba': x_prueba,
            'y_prueba': y_prueba,
            'x_validacion': x_validacion,
            'y_validacion': y_validacion}


def crear_df_error(datos_altura_peso):
    modelos = {'modelo_lineal': entrena_regresion_lineal(datos_altura_peso['x_entrenamiento'],
                                                         datos_altura_peso['y_entrenamiento'])}
    for n_vecinos in [240, 120, 60, 10, 2, 1]:
        modelos[f'modelo_{n_vecinos}_vecinos'] = entrena_k_vecinos(datos_altura_peso['x_entrenamiento'],
                                                                   datos_altura_peso['y_entrenamiento'],
                                                                   n_vecinos)

    nombre_modelos = modelos.keys()
    errores = {'nombre_modelos': nombre_modelos}
    for tipo_error in ['entrenamiento', 'prueba', 'validacion']:
        valor_error = [calcula_error(tipo_error, modelos[nombre_modelo], datos_altura_peso)
                       for nombre_modelo in nombre_modelos]
        errores[tipo_error] = valor_error

    df_errores = pd.DataFrame(errores)

    # reordena
    df_errores = pd.concat([df_errores.iloc[1:4],
                            df_errores.iloc[0:1],
                            df_errores.iloc[4:]], axis=0).reset_index(drop=True)

    return df_errores
