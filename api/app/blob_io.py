import os, re
from typing import Iterable, Optional
from azure.storage.blob import BlobServiceClient

CONN_STR = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER = os.getenv("DATASETS_CONTAINER", "datasets")

_bsc = BlobServiceClient.from_connection_string(CONN_STR)
_container = _bsc.get_container_client(CONTAINER)

DATE_RE = re.compile(r"(\d{8})")

def list_blobs(prefix: str) -> list[str]:
    return [b.name for b in _container.list_blobs(name_starts_with=prefix)]

def latest_blob(prefix: str, ext: str) -> Optional[str]:
    names = list_blobs(prefix)
    dated = []
    for n in names:
        if n.endswith(ext):
            m = DATE_RE.search(n)
            if m:
                dated.append((m.group(1), n))
    if not dated: return None
    dated.sort(key=lambda x: x[0])
    return dated[-1][1]

def blob_text(path: str) -> str:
    blob = _container.get_blob_client(path)
    return blob.download_blob().readall().decode("utf-8")
