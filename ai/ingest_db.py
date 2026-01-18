import sqlite3
from neo4j import GraphDatabase
from dotenv import load_dotenv
from pathlib import Path
import os
from databaselib import get_db



# --- Neo4j ---
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(
        os.getenv("NEO4J_USERNAME"),
        os.getenv("NEO4J_PASSWORD"),
    )
)

# --- DB ---
# def get_db():
#     return sqlite3.connect(os.getenv("DATABASE_URL"))


def ingest():
    db = get_db()
    cursor = db.cursor()

    rows = cursor.execute(
        "SELECT id, name, content, core, rank FROM file_data"
    ).fetchall()

    with driver.session(database=os.getenv("NEO4J_DATABASE")) as session:
        for id_, name, content, core, rank in rows:
            session.run(
                """
                MERGE (d:Document {id: $id})
                SET d.name = $name,
                    d.core = $core,
                    d.rank = $rank
                """,
                {
                    "id": id_,
                    "name": name,
                    "core": core,
                    "rank": rank,
                }
            )

    db.close()
    return {"inserted": len(rows)}
