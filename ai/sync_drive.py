from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import pdfplumber
import os
import re

from openai import OpenAI
from dotenv import load_dotenv
from databaselib import get_db, post_message
from fastapi import HTTPException

from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from google import genai

import io
import os
from databaselib import get_db


client = genai.Client(api_key=os.getenv("GOOGLE_AI_STUDIO_KEY"))


# ---- PDF reader ----

def read_pdf(buffer: bytes) -> str:
    full_text = []

    with pdfplumber.open(io.BytesIO(buffer)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                # ใส่ marker กัน context หาย
                full_text.append(f"\n--- Page {page_number} ---\n")
                full_text.append(text)

    return "\n".join(full_text)


# ---- Google Drive ----
def get_drive():
    creds = Credentials.from_service_account_info(
        {
            "client_email": os.getenv("GOOGLE_CLIENT_EMAIL"),
            "private_key": os.getenv("GOOGLE_PRIVATE_KEY").replace("\\n", "\n"),
            "token_uri": "https://oauth2.googleapis.com/token",
        },
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )

    return build("drive", "v3", credentials=creds)

def get_parent_folder_name(drive, file):
    parents = file.get("parents")
    if not parents:
        return None

    folder = drive.files().get(
        fileId=parents[0],
        fields="name",
        supportsAllDrives=True,
    ).execute()

    return folder.get("name")

# ---- API ----

def sync_drive():
    db = get_db()
    cursor = db.cursor()
    drive = get_drive()

    files = drive.files().list(
        q="mimeType='application/pdf' and trashed=false",
        fields="files(id, name, parents)",
        pageSize=1000,
        supportsAllDrives=True,
    ).execute().get("files", [])

    inserted = 0
    skipped = 0

    for file in files:
        if not file.get("name"):
            skipped += 1
            continue
        file_name = file["name"].replace(".pdf", "").replace(".PDF", "")

        cursor.execute(
            "SELECT id FROM file_data WHERE name = ?",
            (file_name,),
        )
        if cursor.fetchone():
            skipped += 1
            continue

        folder = get_parent_folder_name(drive, file)

        core = None
        rank = None

        if folder and folder.isdigit() and 0 <= int(folder) <= 6:
            core = "commoncore"
            rank = int(folder)
        elif folder == "outercore":
            core = "outercore"
            rank = None
        else:
            skipped += 1
            continue

        # download file
        request = drive.files().get_media(
            fileId=file["id"],
            supportsAllDrives=True,
        )

        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        content = read_pdf(fh.getvalue())

        summmry = None
        if core == "commoncore":
            actual_response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"summarize the following document {content}"
            )
            summmry = actual_response.text

        cursor.execute(
            """
            INSERT INTO file_data (name, content,summary, core, rank)
            VALUES (?, ?, ?, ?,?)
            """,
            (file_name, content,summmry, core, rank),
        )
        print(file_name,core,rank)
        inserted += 1

    db.commit()
    db.close()

    return {
        "success": True,
        "inserted": inserted,
        "skipped": skipped,
    }
