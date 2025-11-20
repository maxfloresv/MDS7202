FROM python:3.10-slim

ENV AIRFLOW_HOME=/opt/airflow
WORKDIR $AIRFLOW_HOME

ENV DEBUG_MODE=True

RUN apt-get update && apt-get install -y curl gdown \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "apache-airflow[pandas]==2.8.4" \
    scikit-learn \
    feature-engine \
    pyarrow

RUN airflow db init

EXPOSE 8080

RUN airflow users create --role Admin --username admin --email admin \
 --firstname admin --lastname admin --password admin

COPY ./dags $AIRFLOW_HOME/dags
COPY ./plugins $AIRFLOW_HOME/plugins

CMD ["sh", "-c", "airflow webserver -p 8080 & airflow scheduler"]