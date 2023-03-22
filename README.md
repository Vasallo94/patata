# PATATA
Funciones útiles con Python para DataScience

<img src="potato-cat.gif" width="250" height="250"/>

#### SE MANTENDRÁ ACTUALIZADO CON DIFERENTES FUNCIONES Y CLASES CREADAS POR LA COMUNIDAD

## Objetivos:
- Aprender funciones útiles en Python para DataScience
- Mantenerse actualizado con nuevas funciones y clases creadas por la comunidad

### Funciones útiles que contiene este repo :

# encode_categorical_columns

"Si tienes una tabla de datos con diferentes tipos de información, esta función puede ayudarte a codificar las columnas que contienen datos categóricos (como nombres o categorías) utilizando una técnica llamada 'Label Encoding'. Esto te permitirá trabajar más fácilmente con estos datos en tu análisis. La función toma como entrada una tabla de datos en formato pandas, aplica el Label Encoding a las columnas categóricas y devuelve una copia de la tabla de datos con las nuevas codificaciones aplicadas."

# best_k

Si tienes una tabla de datos en formato pandas y estás interesado en predecir el valor de una columna específica, esta función puede ayudarte a encontrar el mejor valor para 'k' (número de vecinos) en el modelo de regresión K-NN. La función toma como entrada la tabla de datos, el nombre de la columna objetivo, un rango de valores 'k' y un número mínimo de muestras por pliegue, y utiliza la técnica de validación cruzada para evaluar los diferentes valores de 'k' y encontrar el que minimiza el error cuadrático medio. En otras palabras, esta función te ayuda a encontrar el mejor modelo de regresión K-NN para predecir el valor de la columna objetivo
