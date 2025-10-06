# etl_common/storage.py — REPLACE COMPLETE
import io
import os
from typing import Optional

try:
    from azure.storage.blob import BlobServiceClient, ContentSettings
except Exception:  # pragma: no cover
    BlobServiceClient = None  # type: ignore
    ContentSettings = None  # type: ignore


def _get_container_client(connection_string: Optional[str], container: str):
    if BlobServiceClient is None:
        raise RuntimeError("azure-storage-blob no instalado")
    conn = connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn:
        raise RuntimeError("AZURE_STORAGE_CONNECTION_STRING no definido")
    svc = BlobServiceClient.from_connection_string(conn)
    container_client = svc.get_container_client(container)
    try:
        container_client.create_container()
    except Exception:
        pass
    return container_client


def upload_file(*, local_path: str, container: str, dest_path: str,
                connection_string: Optional[str] = None, content_type: Optional[str] = None):
    """Sube un archivo único (compatibilidad general)."""
    cc = _get_container_client(connection_string, container)
    kwargs = {}
    if content_type and ContentSettings is not None:
        kwargs["content_settings"] = ContentSettings(content_type=content_type)
    with open(local_path, "rb") as f:
        cc.upload_blob(name=dest_path, data=f, overwrite=True, **kwargs)


def upload_bytes_dual(*, data: bytes, container: str, path_dated: str, path_latest: str,
                      connection_string: Optional[str] = None, content_type: Optional[str] = None):
    """Sube el mismo buffer a dos rutas: histórico (fechado) y alias estable latest/."""
    cc = _get_container_client(connection_string, container)
    kwargs = {}
    if content_type and ContentSettings is not None:
        kwargs["content_settings"] = ContentSettings(content_type=content_type)
    cc.upload_blob(name=path_dated, data=io.BytesIO(data), overwrite=True, **kwargs)
    cc.upload_blob(name=path_latest, data=io.BytesIO(data), overwrite=True, **kwargs)


def upload_file_dual(*, local_path: str, container: str, path_dated: str, path_latest: str,
                     connection_string: Optional[str] = None, content_type: Optional[str] = None):
    """Conveniencia para subir dos destinos a partir de un archivo local."""
    with open(local_path, "rb") as f:
        data = f.read()
    upload_bytes_dual(
        data=data,
        container=container,
        path_dated=path_dated,
        path_latest=path_latest,
        connection_string=connection_string,
        content_type=content_type,
    )


def upload_dir_to_azure(source_dir: str, container: str, dest_prefix: str,
                        connection_string: Optional[str] = None):
    """Compatibilidad con funciones existentes; no se usa para latest/."""
    cc = _get_container_client(connection_string, container)
    for root, _, files in os.walk(source_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            rel = os.path.relpath(fpath, source_dir).replace("\\", "/")
            blob_name = f"{dest_prefix}/{rel}" if dest_prefix else rel
            with open(fpath, "rb") as f:
                cc.upload_blob(name=blob_name, data=f, overwrite=True)
