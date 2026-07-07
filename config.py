import os
import tempfile

CHROMA_DB_PATH = os.path.join(
    tempfile.gettempdir(),
    "chroma_db"
)