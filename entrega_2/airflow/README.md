## 🔗 Acerca del DAG de Airflow

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
- `setup_optimized_model`: usa los mejores hiperparámetros encontrados en el ítem anterior para entrenar el modelo LightGBM definitivo. Este lo guarda en la carpeta `app/backend/models` para que lo pueda consumir el _backend_ posteriormente.
- `apply_shap_values`: aplica la técnica de interpretabilidad de SHAP (_SHAP values_) para generar posteriormente gráficos que permitan conocer mejor las decisiones que tomó el modelo.
- `generate_shap_summary`: genera el gráfico de resumen de interpretabilidad del modelo mediante `summary_plot()`, guardando la imagen asociada.