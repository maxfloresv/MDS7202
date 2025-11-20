## 🔗 Acerca del DAG de Airflow

Este DAG es reproducible ejecutando el archivo `docker-compose.yml` que se encuentra en la raíz del proyecto con `docker compose build` y `docker compose up`. Se entregan los archivos `Dockerfile` y `docker-compose.yml`.

Las tareas del DAG (`dag_deployment.py`) se describen a continuación, en el orden de ejecución que corresponde de arriba abajo, de manera lineal (es decir, no hay bifurcaciones).

- `initial_task`: es el marcador de posición del inicio del grafo.
- `create_folders`: crea todas las carpetas necesarias para guardar los archivos del proceso de entrenamiento (p. ej., modelos, imágenes, etc.).
- `dl_transactions`: descarga el _dataset_ de las nuevas transacciones. Se usa `gdown` para descargarlo desde Google Drive. Este último fue el supuesto que se tomó para la "aparición" de nuevos datos, aunque no es realista, dado que para una URL de Google Drive, los datos son estáticos.
- `preprocess`: ejecuta el preprocesamiento mínimo que debe aplicarse al _dataset_ para ser trabajable. Por ejemplo, elimina entradas duplicadas, valores inconsistentes, etc.
- `generate_base_dataframe`: genera el _dataset_ base para entrenar, realizando un _merge_ entre los tres _datasets_ que se poseen (`clientes` y `productos`, que son fijos, y las `transacciones`).
- `clean_base_dataframe_types`: limpia los tipos de datos del _dataset_, dado que el original no estaba optimizado. Por ejemplo, el tipo `object` ocupa mucha memoria RAM con respecto a `category`. Esto reduce considerablemente el peso del DataFrame.
- `split_data`: separa el _dataset mergeado_ en tres conjuntos: _train_ ($75\,\%$), _val_ ($15\,\%$) y _test_ ($15\,\%$). Para las nuevas predicciones, se usó un conjunto que toma **una semana específica** del conjunto de _test_.
- `create_data_transformations`: aplica _Feature Engineering_ sobre el _dataset_, modificando sus variables con escaladores y codificadores. También, separa cada conjunto del paso anterior en sus regresores y su variable objetivo ($X$, $y$).
- `construct_model`: instancia el modelo LightGBM (mejor modelo bajo el contexto del problema) con las aplicaciones de escaladores y codificadores.
- `save_optimization_study`: ejecuta una optimización con Optuna sobre todos los hiperparámetros, tanto para los de los escaladores y codificadores como para los del modelo LightGBM.
- `generate_optuna_plots`: genera los gráficos de Optuna: importancia de hiperparámetros, coordenadas paralelas, y evolución del estudio. Se guarda en `/images` en el contendor.
- `setup_optimized_model`: usa los mejores hiperparámetros encontrados en el ítem anterior para entrenar el modelo LightGBM definitivo. Este lo guarda en la carpeta `app/backend/models` para que lo pueda consumir el _backend_ posteriormente.
- `apply_shap_values`: aplica la técnica de interpretabilidad de SHAP (_SHAP values_) para generar posteriormente gráficos que permitan conocer mejor las decisiones que tomó el modelo.
- `generate_shap_summary`: genera el gráfico de resumen de interpretabilidad del modelo mediante `summary_plot()`, guardando la imagen asociada.

## 📊 Diagrama de flujo

En esta sección, se presenta el diagrama de flujo asociado al DAG, de manera comprensible y resaltando lo más importante.

![](https://raw.githubusercontent.com/maxfloresv/MDS7202/refs/heads/entrega2/entrega_2/diagrama_flujo_airflow.png)

## 🧩 DAG de Airflow

El DAG de Airflow completamente ejecutado (con todas las tareas en estado _success_) se muestra a continuación. El tiempo de ejecución es relativamente corto, porque se usó una muestra aleatoria de las transacciones originales de tamaño $100$ para probar las tareas.

![](https://raw.githubusercontent.com/maxfloresv/MDS7202/refs/heads/entrega2/entrega_2/dag_airflow.png)

## 🧮 Lógica de integración y reentrenamiento

Para integrar nuevos datos y reentrenar el modelo cuando exista _drift_ (tanto en $X$, $y$ o $y \mid X$), se optó por ejecutar el DAG con periodicidad semanal. Esta no es la opción más robusta, porque cada semana consulta el archivo con las transacciones y reentrena, independientemente de si hay _drift_ o no.

Como trabajo futuro, se guardará un _dataset_ de referencia ya procesado, y con un cierto periodo se reejecutará el DAG, viendo si las distribuciones del _dataset_ nuevo cambiaron con respecto al de referencia, reentrenando cuando se excede un cierto umbral en medidas de disimilitud (p. ej., mirando la distancia de Jensen-Shannon o la divergencia de Kullback-Leibler).