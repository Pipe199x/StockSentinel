import os
from typing import Optional

try:
    from azure.storage.blob import BlobServiceClient
except Exception:  # pragma: no cover
    BlobServiceClient = None  # type: ignore


def upload_dir_to_azure(
    source_dir: str,
    container: str,
    dest_prefix: str,
    connection_string: Optional[str] = None,
):
    if BlobServiceClient is None:
        raise RuntimeError(
            "Falta azure-storage-blob. Instala con: pip install azure-storage-blob"
        )
    conn = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING no definido")
    svc = BlobServiceClient.from_connection_string(conn)
    container_client = svc.get_container_client(container)
    try:
        container_client.create_container()
    except Exception:
        pass  # ya existe

    for root, _, files in os.walk(source_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, source_dir).replace("\\", "/")
            blob_name = f"{dest_prefix}/{rel}" if dest_prefix else rel
            with open(fpath, "rb") as f:
                container_client.upload_blob(name=blob_name, data=f, overwrite=True)
