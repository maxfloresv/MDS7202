FROM python:3.10-slim

WORKDIR /root/airflow

ENV AIRFLOW_HOME=/root/airflow

RUN apt-get update && apt-get install -y curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install "apache-airflow==2.8.4"
RUN pip install scikit-learn
RUN pip install gradio
RUN pip install pandas
RUN pip install joblib
RUN pip install -U email-validator

RUN airflow db init

EXPOSE 8080

RUN airflow users create --role Admin --username admin --email admin \
 --firstname admin --lastname admin --password admin

COPY ./dags $AIRFLOW_HOME/dags
COPY ./logs $AIRFLOW_HOME/logs
COPY ./plugins $AIRFLOW_HOME/plugins

CMD ["sh", "-c", "airflow webserver -p 8080 & airflow scheduler"]