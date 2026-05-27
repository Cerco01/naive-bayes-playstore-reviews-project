# Análisis de Sentimientos con Naive Bayes: GridSearchCV, Comparativa de Modelos y Métricas de Clasificación

## Contexto

Este proyecto trabaja con un dataset de reseñas de aplicaciones de la Google Play Store.

El objetivo es construir y comparar modelos de clasificación de texto para diferenciar entre reseñas `negativas` y `positivas`.

El proyecto parte del ejercicio original de [4Geeks Academy](https://github.com/4GeeksAcademy), disponible en el repositorio [naive-bayes-project-tutorial](https://github.com/4GeeksAcademy/naive-bayes-project-tutorial/blob/main/README.es.md).

## Dataset

El dataset original contiene `891` reseñas en formato `.csv`.

La fuente del dataset es este enlace de [4Geeks](https://raw.githubusercontent.com/4GeeksAcademy/naive-bayes-project-tutorial/main/playstore_reviews.csv).

El archivo está incluido en el repositorio en:

```text
data/raw/playstore_reviews.csv
```

Columnas:

- `package_name`: aplicación de origen de la reseña.
- `review`: texto de la reseña.
- `polarity`: clase objetivo (`0` negativa, `1` positiva).

## Qué incluye el proyecto

- Carga y revisión inicial del dataset.
- Limpieza del texto (`strip`, `lower`) y eliminación de columnas sin valor predictivo.
- Vectorización del texto con `CountVectorizer(stop_words="english")`.
- Entrenamiento de tres implementaciones de `Naive Bayes`: `GaussianNB`, `MultinomialNB` y `BernoulliNB`.
- Optimización del mejor Naive Bayes con `GridSearchCV` sobre el parámetro `alpha`.
- Comparación con un modelo alternativo: `LogisticRegression`.
- Evaluación con métricas de clasificación (`accuracy`, `precision`, `recall`, `f1-score`) y matrices de confusión visualizadas.
- Guardado del modelo optimizado y del vectorizador en la carpeta `models/`.

## Resultados

Los modelos se comparan usando `test accuracy`.

| Modelo | Test accuracy |
| --- | ---: |
| `LogisticRegression` | `0.8324` |
| `MultinomialNB` optimizado (`alpha=0.5`) | `0.8268` |
| `MultinomialNB` default | `0.8156` |
| `GaussianNB` | `0.8045` |
| `BernoulliNB` | `0.7709` |

El mejor modelo es `LogisticRegression`, que supera a todos los Naive Bayes en accuracy y tiene mejor recall en la clase positiva (`0.81` vs `0.66`).

El dataset está desbalanceado (`65%` negativas, `35%` positivas), por lo que accuracy sola no basta: es clave mirar el recall de la clase minoritaria.

## Modelo final

El notebook guarda el modelo Naive Bayes optimizado y el vectorizador en:

```text
models/multinomial_nb_optimized.pkl
models/multinomial_nb_vectorizer.pkl
```

Si los archivos no aparecen en `models/`, hay que ejecutar la celda de guardado del Paso 5 en `src/explore.ipynb`.

## Cómo usar este proyecto

1. Clonar el repositorio.
2. Crear o activar un entorno de Python compatible.
3. Instalar las dependencias.

```bash
pip install -r requirements.txt
```

4. Ejecutar el notebook principal.

```text
src/explore.ipynb
```

El notebook está preparado para trabajar con el kernel `data-science` (Python 3.11).

## Archivos principales

- `src/explore.ipynb`: notebook principal del proyecto.
- `src/apple.mplstyle`: estilo visual usado en los gráficos.
- `requirements.txt`: dependencias necesarias para ejecutar el proyecto.
- `models/`: carpeta donde se guardan los modelos entrenados.
- `data/raw/`: carpeta local para el dataset. El archivo `.csv` está incluido.

## Mejora futura

Como mejora futura, se podría:

- Probar técnicas de balanceo de clases (`SMOTE`, pesos en el modelo) para mejorar el recall de las reseñas positivas.
- Explorar `TF-IDF` en lugar de `CountVectorizer`.
- Realizar un análisis de las palabras más predictivas de cada clase.

## Créditos

Este proyecto fue realizado como parte del [Bootcamp de Data Science y Machine Learning de 4Geeks](https://4geeksacademy.com/es/bootcamps/curso-datascience-machine-learning).

El enunciado original pertenece a [4Geeks Academy](https://github.com/4GeeksAcademy).
