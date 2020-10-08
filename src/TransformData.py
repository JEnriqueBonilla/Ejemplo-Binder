from sklearn.model_selection import train_test_split


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
