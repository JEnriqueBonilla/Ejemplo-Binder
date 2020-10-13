import plotly.express as px
import plotly.graph_objects as go


LABELS_MAP = {'Age':'edad',
              'Fare':'Tarifa',
              'Sex': 'Sexo',
              'Pclass': 'Clase del Pasajero',
              'Cabin': 'Cabina',
              'Embarked': 'Embarcó?'}


def crear_figura_regresion(x, y, x_modelo, y_modelo, nombre_modelo, titulo):
    constante = 10

    figura = px.scatter(x=x, y=y, opacity=0.65)
    figura.add_traces(go.Scatter(x=x_modelo, y=y_modelo, name=nombre_modelo))
    figura.update_layout(title=titulo, title_x=0.5, xaxis_title=x.name, yaxis_title=y.name)
    figura.update_xaxes(range=[x.min() - constante, x.max() + constante])
    figura.update_yaxes(range=[y.min() - constante, y.max() + constante])

    return figura


def crear_figuras_entrenamiento_prueba_validacion(datos, x_modelo, y_modelo, nombre_modelo):
    figura_entrenamiento = crear_figura_regresion(datos['x_entrenamiento'],
                                                  datos['y_entrenamiento'],
                                                  x_modelo,
                                                  y_modelo,
                                                  nombre_modelo,
                                                  'Entrenamiento')

    figura_prueba = crear_figura_regresion(datos['x_prueba'],
                                           datos['y_prueba'],
                                           x_modelo,
                                           y_modelo,
                                           nombre_modelo,
                                           'Prueba')

    figura_validacion = crear_figura_regresion(datos['x_validacion'],
                                               datos['y_validacion'],
                                               x_modelo,
                                               y_modelo,
                                               nombre_modelo,
                                               'Validación')

    return {'figura_entrenamiento': figura_entrenamiento,
            'figura_prueba': figura_prueba,
            'figura_validacion': figura_validacion}


def crear_figura_regresion_logistica_una_variable(x, y, x_modelo, y_modelo, limite, variable_options):
    categoria_predicha = [0 if prob_1 <= limite else 1 for _, prob_1 in y]
    y_modelo = [prob_1 for _, prob_1 in y_modelo]

    figura = px.scatter(x=x, y=categoria_predicha, opacity=0.65)
    figura.add_traces(go.Scatter(x=x_modelo, y=y_modelo, name="Regresión logística"))
    figura.update_layout(title='Regresión Logística con una variable', title_x=0.5,
                         xaxis_title=LABELS_MAP[variable_options[0]], yaxis_title='Probabilidad de supervivencia')

    return figura


def crear_figura_regresion_logistica_dos_varaibles(x, y, limite, variable_options):
    categoria_predicha = ['No Sobrevivió' if prob_1 <= limite else 'Sobrevivió' for _, prob_1 in y]

    if variable_options[1] == 'Sex':
        x[variable_options[1]] = ['Masculino' if sexo == 1 else 'Femenino' for sexo in x[variable_options[1]].values]

    figura = px.scatter(x=x[variable_options[0]], y=x[variable_options[1]], opacity=0.65, color=categoria_predicha)

    figura.update_layout(title='Regresión Logística con dos variables',
                         title_x=0.5,
                         xaxis_title=LABELS_MAP[variable_options[0]],
                         yaxis_title=LABELS_MAP[variable_options[1]])
    return figura


def crear_figuras_regresion_logistica(datos_logisticos, variable_options, limite=.5):
    figura_una_variable = crear_figura_regresion_logistica_una_variable(datos_logisticos['x_prueba'][variable_options[0]],
                                                                        datos_logisticos['prediccion_una_variable'],
                                                                        datos_logisticos['x_modelo'],
                                                                        datos_logisticos['y_modelo'],
                                                                        limite,
                                                                        variable_options)

    figura_dos_variables = crear_figura_regresion_logistica_dos_varaibles(datos_logisticos['x_prueba'],
                                                                          datos_logisticos['prediccion_dos_variables'],
                                                                          limite,
                                                                          variable_options)

    return figura_una_variable, figura_dos_variables


def graficar_error(figura_error, tipo_de_error, df_errores):
    df_un_error = df_errores[['nombre_modelos', tipo_de_error]]

    figura_error.add_traces(go.Scatter(x=df_un_error['nombre_modelos'], y=df_un_error[tipo_de_error],
                                       name='Error de ' + tipo_de_error + ' k vecinos'))

    return figura_error
