from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator

from utils import (
    fetch_news_for_summarization
)
from datetime import datetime, timedelta


default_args = {
    "owner": "cfi-stock-prediction-team",
    "depends_on_past": False,
    "start_time": datetime(2021, 2, 15),
    "email": ["7vasudevgupta@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": True,
    "retries": 2,
    "retry_delay": timedelta(hours=2),
}

dag = DAG("stock-pipeline", default_args=default_args, schedule_interval="@daily")

fetch_news_for_summarization = PythonOperator(
                                task_id='metadata/fetch_news_for_summarization',
                                python_callable=fetch_news_for_summarization,
                                dag=dag
                            )

fetch_news_for_summarization
