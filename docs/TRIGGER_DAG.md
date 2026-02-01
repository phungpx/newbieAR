# How to Trigger `vectordb_ingestion` DAG

The `vectordb_ingestion` DAG requires a `file_path` parameter in the DAG run configuration. This file will be uploaded to MinIO and processed through the ingestion pipeline.

## Prerequisites

- Airflow is running and accessible
- The file path exists on the Airflow worker/scheduler machine
- You have appropriate permissions to trigger DAGs

## Method 1: Airflow CLI

### Basic Trigger
```bash
airflow dags trigger vectordb_ingestion
```

### Trigger with Configuration
```bash
# Using JSON configuration
airflow dags trigger vectordb_ingestion \
  --conf '{"file_path": "/path/to/your/document.pdf"}'

# Using a JSON file
airflow dags trigger vectordb_ingestion \
  --conf '{"file_path": "/Users/phung.pham/Documents/PHUNGPX/deepeval_exploration/data/papers/files/docling.pdf"}'
```

### Example with Local File
```bash
airflow dags trigger vectordb_ingestion \
  --conf '{"file_path": "/Users/phung.pham/Documents/PHUNGPX/deepeval_exploration/data/papers/files/docling.pdf"}'
```

## Method 2: Airflow UI (Web Interface)

1. Navigate to the Airflow UI (typically `http://localhost:8080`)
2. Find the `vectordb_ingestion` DAG in the DAGs list
3. Click the **Play** button (▶️) next to the DAG name
4. In the "Trigger DAG w/ config" dialog:
   - Select "JSON" format
   - Enter the configuration:
     ```json
     {
       "file_path": "/path/to/your/document.pdf"
     }
     ```
5. Click **Trigger**

## Method 3: Airflow REST API

### Using cURL
```bash
curl -X POST \
  http://localhost:8080/api/v1/dags/vectordb_ingestion/dagRuns \
  -H "Content-Type: application/json" \
  -H "Authorization: Basic <base64_encoded_credentials>" \
  -d '{
    "dag_run_id": "manual__'$(date +%s)'",
    "conf": {
      "file_path": "/path/to/your/document.pdf"
    }
  }'
```

### Using Python `requests`
```python
import requests
from datetime import datetime

url = "http://localhost:8080/api/v1/dags/vectordb_ingestion/dagRuns"
auth = ("airflow", "airflow")  # Update with your credentials

payload = {
    "dag_run_id": f"manual__{int(datetime.now().timestamp())}",
    "conf": {
        "file_path": "/path/to/your/document.pdf"
    }
}

response = requests.post(url, json=payload, auth=auth)
print(response.json())
```

## Method 4: Python Airflow Client

### Using Airflow 2.x Client API
```python
from airflow.api.client.local_client import Client
from datetime import datetime

client = Client(None, None)

# Trigger the DAG
dag_run = client.trigger_dag(
    dag_id="vectordb_ingestion",
    run_id=f"manual__{datetime.now().isoformat()}",
    conf={"file_path": "/path/to/your/document.pdf"}
)

print(f"DAG run created: {dag_run}")
```

### Using Airflow 2.x REST API Client
```python
from airflow.api.client.local_client import Client
from datetime import datetime

client = Client(None, None)

# Trigger with configuration
dag_run = client.trigger_dag(
    dag_id="vectordb_ingestion",
    run_id=f"manual__{datetime.now().isoformat()}",
    conf={"file_path": "/path/to/your/document.pdf"}
)
```

## Method 5: Programmatic Trigger (Within Python Code)

If you want to trigger this from another Python script or DAG:

```python
from airflow.models import DagBag
from airflow.api.common.trigger_dag import trigger_dag
from datetime import datetime

# Trigger the DAG programmatically
dag_run = trigger_dag(
    dag_id="vectordb_ingestion",
    run_id=f"manual__{datetime.now().isoformat()}",
    conf={"file_path": "/path/to/your/document.pdf"},
    replace_microseconds=False
)
```

## Method 6: Using Airflow Variables (Alternative Approach)

If you prefer to use Airflow Variables instead of conf, you could modify the DAG to read from Variables:

```python
# In your DAG file
from airflow.models import Variable

file_path = Variable.get("vectordb_ingestion_file_path", default_var=None)
```

Then set the variable:
```bash
airflow variables set vectordb_ingestion_file_path "/path/to/your/document.pdf"
airflow dags trigger vectordb_ingestion
```

## Important Notes

1. **File Path**: The `file_path` must be accessible from the Airflow worker/scheduler. If using Docker, ensure the file is mounted or accessible within the container.

2. **File Format**: The DAG supports various document formats (PDF, Markdown, etc.) based on your `DocLoader` implementation.

3. **Error Handling**: If `file_path` is missing, the DAG will fail with: `ValueError: Missing 'file_path' in DAG run configuration`

4. **Run ID**: If not specified, Airflow will auto-generate a run ID. You can specify a custom one using `--run-id` in CLI or `run_id` in API calls.

## Example: Trigger from Script

Create a helper script `scripts/trigger_ingestion.sh`:

```bash
#!/bin/bash

FILE_PATH="${1:-/Users/phung.pham/Documents/PHUNGPX/deepeval_exploration/data/papers/files/docling.pdf}"

if [ ! -f "$FILE_PATH" ]; then
    echo "Error: File not found: $FILE_PATH"
    exit 1
fi

airflow dags trigger vectordb_ingestion \
  --conf "{\"file_path\": \"$FILE_PATH\"}"

echo "DAG triggered with file: $FILE_PATH"
```

Usage:
```bash
chmod +x scripts/trigger_ingestion.sh
./scripts/trigger_ingestion.sh /path/to/document.pdf
```
