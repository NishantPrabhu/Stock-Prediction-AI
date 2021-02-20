# Stock Prediction AI

Analytics Club project 2020

## Starting App

```shell
# building docker image
docker build -f Dockerfile -t cfi-stock-ai .

docker run -it -v <absolute-path-to-dags-folder>:/usr/local/airflow/dags -p 85:8501 cfi-stock-ai streamlit run dags/app.py
# app is available at localhost:85
```
