import plotly.express as px
import plotly.graph_objects as go


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

