import sqlite3, pathlib
DB_PATH = pathlib.Path('data/algae.db')

def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    with get_conn() as c:
        c.executescript(pathlib.Path('schema.sql').read_text())
