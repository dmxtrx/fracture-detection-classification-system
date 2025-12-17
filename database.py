import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Tuple

DB_PATH = "fracture_detection.db"

def init_db():
    """Инициализация базы данных"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute("""CREATE TABLE IF NOT EXISTS xrayimages_new (
        id INTEGER PRIMARY KEY,
        filename TEXT UNIQUE NOT NULL,
        filepath TEXT NOT NULL,
        uploaddate TEXT NOT NULL,
        filesize INTEGER
    )""")
    
    c.execute("""CREATE TABLE IF NOT EXISTS fracture_detections (
        id INTEGER PRIMARY KEY,
        image_id INTEGER,
        filename TEXT NOT NULL,
        classname TEXT NOT NULL,
        confidence REAL NOT NULL,
        bbox_x1 REAL,
        bbox_y1 REAL,
        bbox_x2 REAL,
        bbox_y2 REAL,
        detection_date TEXT NOT NULL
    )""")
    
    c.execute("CREATE TABLE IF NOT EXISTS xrayimages_train (id INTEGER PRIMARY KEY, filename TEXT, filepath TEXT, uploaddate TEXT, filesize INTEGER)")
    c.execute("CREATE TABLE IF NOT EXISTS xrayimages_valid (id INTEGER PRIMARY KEY, filename TEXT, filepath TEXT, uploaddate TEXT, filesize INTEGER)")
    c.execute("CREATE TABLE IF NOT EXISTS xrayimages_test (id INTEGER PRIMARY KEY, filename TEXT, filepath TEXT, uploaddate TEXT, filesize INTEGER)")
    
    conn.commit()
    conn.close()

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def save_new_image(filename: str, filepath: str, filesize: int) -> int:
    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "INSERT INTO xrayimages_new (filename, filepath, uploaddate, filesize) VALUES (?, ?, ?, ?)",
        (filename, filepath, datetime.utcnow().isoformat(), filesize)
    )
    conn.commit()
    image_id = c.lastrowid
    conn.close()
    return image_id

def get_image_path(image_id: int) -> str:
    """Получить путь к файлу по его ID"""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT filepath FROM xrayimages_new WHERE id = ?", (image_id,))
    row = c.fetchone()
    conn.close()
    return row['filepath'] if row else None

def save_detection(image_id: int, filename: str, classname: str, confidence: float, bbox=None):
    conn = get_connection()
    c = conn.cursor()
    x1, y1, x2, y2 = bbox if bbox else (None, None, None, None)
    
    c.execute(
        """INSERT INTO fracture_detections 
           (image_id, filename, classname, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2, detection_date)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (image_id, filename, classname, confidence, x1, y1, x2, y2, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()

def get_detection_stats() -> Dict:
    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute("SELECT COUNT(*) FROM fracture_detections")
        total = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM fracture_detections WHERE classname = 'Fracture'")
        fracture_count = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM fracture_detections WHERE classname = 'Normal'")
        normal_count = c.fetchone()[0]
        c.execute("SELECT AVG(confidence) FROM fracture_detections")
        row = c.fetchone()
        avg_conf = row[0] if row and row[0] is not None else 0.0
    except Exception:
        total, fracture_count, normal_count, avg_conf = 0, 0, 0, 0.0
        
    conn.close()
    return {
        "total_analyzed": total,
        "fracture_count": fracture_count,
        "normal_count": normal_count,
        "avg_confidence": round(avg_conf, 4)}

if __name__ == "__main__":
    init_db()
