from scipy import interpolate

def prediccion_poblacion_simple_2019(valores):
    funcion_prediccion = interpolate.interp1d([1,2,3],valores, fill_value="extrapolate")
    return int(funcion_prediccion(4))

def prediccion_urbana_simple_2019(valores):
    funcion_prediccion = interpolate.interp1d([1,2,3],valores, fill_value="extrapolate")
    return int(funcion_prediccion(4))
