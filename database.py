import sqlite3

def create_connection():
    return sqlite3.connect("agrisense.db")

def setup_tables():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS farmer_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            crop TEXT,
            yield REAL,
            sustainability_score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

