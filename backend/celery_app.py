import os
from celery import Celery

# Load env (chỉ cần nếu bạn dùng python-dotenv, còn không thì uvicorn tự load .env trong Next.js, 
# với Python backend thì bạn có thể export env trong terminal hoặc dùng docker-compose)
from dotenv import load_dotenv
load_dotenv()

user = os.getenv("RABBITMQ_USER", "guest")
password = os.getenv("RABBITMQ_PASS", "guest")
host = os.getenv("RABBITMQ_HOST", "rabbitmq")  # Sử dụng tên service trong Docker
port = os.getenv("RABBITMQ_PORT", "5672")
vhost = os.getenv("RABBITMQ_VHOST", "/")

broker_url = f"amqp://{user}:{password}@{host}:{port}{vhost}"

celery_app = Celery(
    "worker",
    broker=broker_url,
    backend="rpc://"
)

celery_app.conf.update(
    task_routes={
        "tasks.retrain_model": {"queue": "eeg"}
    },
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Fix Windows permission issues
    worker_pool="solo",  # Use solo pool instead of prefork on Windows
    worker_concurrency=1,  # Single worker process
    # Auto-discover tasks
    include=['app.tasks']
)
