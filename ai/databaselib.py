from dotenv import load_dotenv
from neo4j import GraphDatabase
import os
import sqlite3
from pathlib import Path
from fastapi import HTTPException


# โหลด .env จากโฟลเดอร์ปัจจุบัน
# load_dotenv()

# uri = os.getenv("NEO4J_URI")
# user = os.getenv("NEO4J_USERNAME")
# password = os.getenv("NEO4J_PASSWORD")

# # print(uri,user,password)

# driver = GraphDatabase.driver(
#     uri,
#     auth=(user, password)
# )

# with driver.session() as session:
#     result = session.run("RETURN 'Neo4j OK' AS msg")
#     print(result.single()["msg"])

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

def get_db():
    return sqlite3.connect(os.getenv("DATABASE_URL"))

def post_message(role, content):
    db = get_db()
    cursor = db.cursor()

    try:
        # 1️⃣ save user message
        cursor.execute(
            """
            INSERT INTO messages ( role, content)
            VALUES ( ?, ?)
            """,
            (role, content),
        )

        # 2️⃣ call AI
        # answer = call_ai(question)

        # # 3️⃣ save AI message
        # cursor.execute(
        #     """
        #     INSERT INTO messages (session_id, role, content, model, source)
        #     VALUES (?, 'ai', ?, ?, ?)
        #     """,
        #     (session_id, answer, "typhoon", "rag"),
        # )

        db.commit()

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        db.close()

def get_messages(limit: int = 50):
    db = get_db()
    cursor = db.cursor()

    rows = cursor.execute(
        """
        SELECT role, content, created_at
        FROM messages
        ORDER BY created_at ASC
        LIMIT ?
        """,
        (limit,)
    ).fetchall()

    db.close()

    return [
        {"role": r, "text": c, "created_at": t}
        for r, c, t in rows
    ]


