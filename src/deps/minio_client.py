from pathlib import Path
from typing import Optional
from io import BytesIO

from minio import Minio
from minio.error import S3Error
from loguru import logger


class MinIOClient:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool = False,
    ):
        """Initialize MinIO client.
        Args:
            endpoint: MinIO server endpoint (e.g., 'localhost:9000')
            access_key: MinIO access key
            secret_key: MinIO secret key
            secure: Whether to use HTTPS (default: False for local development)
        """
        self.endpoint = endpoint
        self.access_key = access_key
        self.secret_key = secret_key
        self.secure = secure
        self.client = Minio(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        logger.info(f"Initialized MinIO client for endpoint: {endpoint}")

    def ensure_bucket(self, bucket_name: str) -> None:
        """Create bucket if it doesn't exist.

        Args:
            bucket_name: Name of the bucket to ensure exists

        Raises:
            S3Error: If bucket creation fails
        """
        try:
            if not self.client.bucket_exists(bucket_name):
                self.client.make_bucket(bucket_name)
                logger.info(f"Created bucket: {bucket_name}")
            else:
                logger.debug(f"Bucket already exists: {bucket_name}")
        except S3Error as e:
            logger.error(f"Failed to ensure bucket '{bucket_name}': {e}")
            raise

    def upload_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str | Path,
        content_type: Optional[str] = None,
    ) -> str:
        """Upload a file to MinIO.

        Args:
            bucket_name: Name of the bucket
            object_name: Object name (path) in the bucket
            file_path: Path to the local file to upload
            content_type: Optional content type (MIME type)

        Returns:
            Object path in format: bucket_name/object_name

        Raises:
            S3Error: If upload fails
            FileNotFoundError: If local file doesn't exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        self.ensure_bucket(bucket_name)

        try:
            self.client.fput_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=str(file_path),
                content_type=content_type,
            )
            object_path = f"{bucket_name}/{object_name}"
            logger.info(f"Uploaded file to: {object_path}")
            return object_path
        except S3Error as e:
            logger.error(
                f"Failed to upload file '{file_path}' to '{bucket_name}/{object_name}': {e}"
            )
            raise

    def upload_string(
        self,
        bucket_name: str,
        object_name: str,
        content: str,
        content_type: Optional[str] = None,
    ) -> str:
        """Upload string content to MinIO.

        Args:
            bucket_name: Name of the bucket
            object_name: Object name (path) in the bucket
            content: String content to upload
            content_type: Optional content type (default: 'text/plain')

        Returns:
            Object path in format: bucket_name/object_name

        Raises:
            S3Error: If upload fails
        """
        self.ensure_bucket(bucket_name)

        if content_type is None:
            content_type = "text/plain"

        try:
            content_bytes = content.encode("utf-8")
            content_stream = BytesIO(content_bytes)

            self.client.put_object(
                bucket_name=bucket_name,
                object_name=object_name,
                data=content_stream,
                length=len(content_bytes),
                content_type=content_type,
            )
            object_path = f"{bucket_name}/{object_name}"
            logger.info(
                f"Uploaded string content to: {object_path} ({len(content_bytes)} bytes)"
            )
            return object_path
        except S3Error as e:
            logger.error(
                f"Failed to upload string to '{bucket_name}/{object_name}': {e}"
            )
            raise

    def download_file(
        self,
        bucket_name: str,
        object_name: str,
        file_path: str | Path,
    ) -> Path:
        """Download a file from MinIO.

        Args:
            bucket_name: Name of the bucket
            object_name: Object name (path) in the bucket
            file_path: Local path where to save the file

        Returns:
            Path to the downloaded file

        Raises:
            S3Error: If download fails
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self.client.fget_object(
                bucket_name=bucket_name,
                object_name=object_name,
                file_path=str(file_path),
            )
            logger.info(
                f"Downloaded file from '{bucket_name}/{object_name}' to '{file_path}'"
            )
            return file_path
        except S3Error as e:
            logger.error(f"Failed to download '{bucket_name}/{object_name}': {e}")
            raise

    def download_string(
        self,
        bucket_name: str,
        object_name: str,
    ) -> str:
        """Download string content from MinIO.

        Args:
            bucket_name: Name of the bucket
            object_name: Object name (path) in the bucket

        Returns:
            String content of the object

        Raises:
            S3Error: If download fails
        """
        try:
            response = self.client.get_object(
                bucket_name=bucket_name,
                object_name=object_name,
            )
            content = response.read().decode("utf-8")
            response.close()
            response.release_conn()
            logger.debug(
                f"Downloaded string from '{bucket_name}/{object_name}' ({len(content)} bytes)"
            )
            return content
        except S3Error as e:
            logger.error(
                f"Failed to download string from '{bucket_name}/{object_name}': {e}"
            )
            raise

    def list_objects(
        self,
        bucket_name: str,
        prefix: Optional[str] = None,
        recursive: bool = True,
    ) -> list[str]:
        """List objects in a bucket.

        Args:
            bucket_name: Name of the bucket
            prefix: Optional prefix to filter objects
            recursive: Whether to list recursively (default: True)

        Returns:
            List of object names

        Raises:
            S3Error: If listing fails
        """
        try:
            objects = self.client.list_objects(
                bucket_name=bucket_name,
                prefix=prefix,
                recursive=recursive,
            )
            object_names = [obj.object_name for obj in objects]
            logger.debug(
                f"Listed {len(object_names)} objects in '{bucket_name}' with prefix '{prefix}'"
            )
            return object_names
        except S3Error as e:
            logger.error(f"Failed to list objects in '{bucket_name}': {e}")
            raise

    def delete_object(
        self,
        bucket_name: str,
        object_name: str,
    ) -> None:
        """Delete an object from MinIO.

        Args:
            bucket_name: Name of the bucket
            object_name: Object name (path) to delete

        Raises:
            S3Error: If deletion fails
        """
        try:
            self.client.remove_object(
                bucket_name=bucket_name,
                object_name=object_name,
            )
            logger.info(f"Deleted object: {bucket_name}/{object_name}")
        except S3Error as e:
            logger.error(f"Failed to delete '{bucket_name}/{object_name}': {e}")
            raise

    def object_exists(
        self,
        bucket_name: str,
        object_name: str,
    ) -> bool:
        """Check if an object exists in MinIO.

        Args:
            bucket_name: Name of the bucket
            object_name: Object name (path) to check

        Returns:
            True if object exists, False otherwise
        """
        try:
            self.client.stat_object(
                bucket_name=bucket_name,
                object_name=object_name,
            )
            return True
        except S3Error:
            return False
