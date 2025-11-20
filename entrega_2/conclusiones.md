## Conclusiones Entrega 2 MDS7202

- Estudiantes: Naomi Cautivo B. y Máximo Flores Valenzuela.
---
- ¿Cómo mejoró el desarrollo del proyecto al utilizar herramientas de *tracking* y despliegue?
> **Respuesta:**  El desarrollo de esta entrega permitió mejorar diferentes aristas importantes en un proyecto de ciencia de datos, tales como:
> 1. Trazabilidad: Los errores al utilizar _Airflow_ son más descriptivos y permiten la trazabilidad de errores dentro del proyecto.
> 2. Atomicidad y control de versiones: Al trabajar con _Docker_ en la aplicación web, fue mucho mejor para replicar la aplicación en los computadores del equipo y trabajar o probar bajo las mismas condiciones, lo cual es clave para asegurar que el proyecto sea reproducible y utilizable para otras personas.
> 3. Limpieza y refactorización de código: La utilización de _Airflow_ forzó a que se realizara una refactorización desde la entrega 1 a los componentes del _DAG_, lo cual generó una instancia de limpieza y optimización de código, esto hace que el proyecto logre ser más mantenible en el tiempo y tenga mejores prácticas de programación.
- ¿Qué aspectos del despliegue con `Gradio/FastAPI` fueron más desafiantes o interesantes?
> **Respuesta:** El despliegue en general fue desafiante, dado que eran las primeras interacciones con las herramientas fuera de los laboratorios. La parte más interesante fue personalizar la plataforma para que fuera intuitiva y clara, ya que _Gradio_ posee personalización por medio de sus _themes_ y la introducción de objetos en _CSS_, lo cual se realizó parcialmente para modificar la tipografía del tema _'Soft'_  para adecuarse mejor a la estética moderna y futurista de _SodAI Drinks_.
>
> Por otro lado, al trabajar con _FastAPI_, la implementación del _backend_ se simplificó bastante en comparación a otros _frameworks_ con los que el equipo había trabajado previamente, dejando una impresión bastante grata para proyectos futuros donde se requiera disponer de una plataforma para permitir la interacción con un modelo en _Python_.
- ¿Cómo aporta `Airflow` a la robustez y escalabilidad del pipeline?

- ¿Qué se podría mejorar en una versión futura del flujo? ¿Qué partes automatizarían más, qué monitorearían o qué métricas agregarían?
> **Respuesta:** Una forma interesante de mejorar en una versión futura es la aplicación de _MLFlow_ para el monitoreo de métricas e interpretabilidad, lo cual en el presente trabajo no se realizó, esto permitiría mejorar la trazabilidad de métricas y mejora continua.
> 
> Un aspecto que dentro de producción podría sumar bastante es un _benchmarking_ a medida que los datos presenten _drift_, es decir, que si los datos divergen de su distribución, en el re-entrenamiento sea posible comparar diversos modelos. Debido que si las ventas de  _SodAI Drinks_ se vuelven suficientemente diferentes, puede generar que el modelo ya no sea del todo efectivo, ya que no tendría aprendidos esos patrones de venta. Lo cual, con un monitoreo y prueba constante, se puede generar una  etapa de mejora continua, sin embargo esto sería bastante costoso computacionalmente, así que sería útil en instancias donde la complejidad computacional no sea una limitante.
> 
> Adicionalmente, agregaríamos métricas ligadas al desbalance de clases, ya que pudimos notar que el problema en general posee bastante desbalance en los datos, lo cual hace que las métricas clásicas puedan verse perjudicadas. 
>


### Reflexión final de Entrega 2

Dentro del trabajo relacionado a esta entrega, el equipo pudo notar el avance en su forma de programar con herramientas ligadas a la ciencia de datos, lo cual aporta valor a su formación como _Data Scientist_, ya que luego de cursar gran parte del curso, adquirieron herramientas y prácticas claves para la disciplina como lo puede ser, _Airflow_, _Pipelines_, _Docker_, interpretabilidad, prevención de _Data Leakage_, entre otros.

Por tanto, este proyecto muestra la evolución de sus integrantes en mejores prácticas como cientistas de datos, lo cual es sumamente importante de cara a la inmersión profesional y académica dentro del área. 