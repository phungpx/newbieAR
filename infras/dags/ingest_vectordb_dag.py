from datetime import datetime, timedelta
from airflow.sdk import dag, task
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.task.trigger_rule import TriggerRule
from constants.dag_id import DagId
from constants.vectordb_ingestion_tasks import VectordbIngestionTasks
from tasks.ingest_vectordb import (
    store_uploaded_file_to_minio_task,
    chunk_document_task,
    embed_and_store_task,
    convert_to_markdown_task,
)

# Default arguments for all tasks
default_args = {
    "owner": "phungpx",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


@dag(
    dag_id=DagId.VECTORDB_INGESTION.value,
    default_args=default_args,
    description="RAG Ingestion: Upload -> Raw Chunking -> Vector Store",
    schedule=None,  # Triggered manually or via API
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["rag", "ingestion", "qdrant", "minio"],
)
def ingest_vectordb_dag():
    start = EmptyOperator(task_id="start")
    end = EmptyOperator(task_id="end", trigger_rule=TriggerRule.NONE_FAILED)

    @task(task_id=VectordbIngestionTasks.UPLOAD_RAW_FILE.value)
    def upload_file(**context):
        conf = context.get("dag_run").conf or {}
        file_path = conf.get("file_path")
        if not file_path:
            raise ValueError("Missing 'file_path' in DAG run configuration")

        return store_uploaded_file_to_minio_task(file_path)

    @task(task_id=VectordbIngestionTasks.CHUNK_RAW_FILE.value)
    def chunk_file(upload_result: dict):
        if upload_result.get("status") != "success":
            raise ValueError(f"Upload failed: {upload_result.get('error')}")
        # Pass the raw object path (e.g., "uploads/doc.pdf")
        return chunk_document_task(upload_result["object_path"])

    @task(task_id=VectordbIngestionTasks.EMBED_AND_INDEX.value)
    def embed_and_index(chunk_result: dict):
        if chunk_result.get("status") != "success":
            raise ValueError(f"Chunking failed: {chunk_result.get('error')}")
        return embed_and_store_task(chunks_object_path=chunk_result["chunks_path"])

    @task(task_id=VectordbIngestionTasks.GENERATE_MARKDOWN_PREVIEW.value)
    def generate_markdown(upload_result: dict):
        if upload_result.get("status") != "success":
            return None
        return convert_to_markdown_task(upload_result["object_path"])

    upload_data = upload_file()
    chunk_data = chunk_file(upload_data)
    index_data = embed_and_index(chunk_data)
    markdown_data = generate_markdown(upload_data)

    # Set flow
    start >> upload_data

    # Connect dependencies
    upload_data >> chunk_data >> index_data >> end
    upload_data >> markdown_data >> end


ingest_vectordb_dag()
