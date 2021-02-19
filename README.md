# Stock Prediction AI

Analytics Club project 2020

## Starting App

```shell
# init airflow webserver
docker-compose -f docker-compose-LocalExecutor.yml up -d

# get <container_id>
docker ps

# init streamlit app
docker exec -it <container_id> python dags/app.py 
```
