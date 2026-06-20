# database.py - MongoDB integration for V2 RAG Engine
import logging
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import unified_db
    db = unified_db.db
except ImportError:
    unified_db = None
    db = None

logger = logging.getLogger("db.database")

def create_tables():
    if db is not None:
        try:
            db.documents.create_index("file_name", unique=True)
            db.documents.create_index("document_id", unique=True)
        except Exception as e:
            logger.error(f"Error creating document indexes: {e}")

def insert_document(document_id, file_name, doc_type, file_data, status="processing"):
    if db is None: return "dummy_id"
    
    doc = {
        "document_id": document_id,
        "file_name": file_name,
        "doc_type": doc_type,
        "status": status,
        "upload_time": datetime.utcnow()
    }
    # Optional: file_data could be saved if needed, but it's usually large so better omitted unless required
    db.documents.insert_one(doc)
    return document_id

def get_document_by_filename(file_name):
    if db is None: return None
    doc = db.documents.find_one({"file_name": file_name})
    return doc if doc else None

def update_document_status(file_name, status):
    if db is None: return
    db.documents.update_one({"file_name": file_name}, {"$set": {"status": status}})

def get_processing_documents():
    if db is None: return []
    return list(db.documents.find({"status": "processing"}))

def cleanup_stuck_documents():
    if db is None: return
    db.documents.update_many({"status": "processing"}, {"$set": {"status": "failed"}})

def delete_document_record(file_name):
    if db is None: return False
    res = db.documents.delete_one({"file_name": file_name})
    return res.deleted_count > 0

def list_documents():
    if db is None: return []
    docs = list(db.documents.find().sort("upload_time", -1))
    return [
        {
            "file_name": doc.get("file_name", ""),
            "doc_type": doc.get("doc_type", ""),
            "status": doc.get("status", ""),
            "upload_time": doc.get("upload_time").isoformat() if doc.get("upload_time") else None,
        }
        for doc in docs
    ]

def clear_all_documents():
    if db is None: return False
    db.documents.delete_many({})
    return True

def health_check_db():
    if db is None:
        raise Exception("MongoDB is not connected.")
    # simple ping
    unified_db.client.admin.command('ping')

def insert_conversation(session_id, question, answer, source=None, page=None, page_label=None):
    if not unified_db: return
    metrics = {"source": source, "page": page, "page_label": page_label}
    # User message is already inserted by unified_app.py
    unified_db.append_message(session_id, "assistant", answer, rag_version="v2", metrics=metrics)

class DummyTurn:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer

def fetch_conversation_history(session_id, limit=6):
    if not unified_db: return []
    try:
        history = unified_db.get_chat_history(session_id, limit * 2) # *2 because user/assistant pairs
        turns = []
        i = 0
        while i < len(history):
            if history[i]["role"] == "user":
                q = history[i]["content"]
                a = ""
                if i + 1 < len(history) and history[i+1]["role"] == "assistant":
                    a = history[i+1]["content"]
                    i += 2
                else:
                    i += 1
                turns.append(DummyTurn(q, a))
            else:
                i += 1
        return turns[-limit:] if limit > 0 else turns
    except Exception as e:
        logger.error(f"Error fetching MongoDB history: {e}")
        return []

def fetch_all_sessions():
    if not unified_db: return []
    class DummySession:
        def __init__(self, s_id, t):
            self.session_id = s_id
            self.title = t
    
    # We grab sessions for default_user if not specified
    raw_sessions = unified_db.get_user_sessions("default_user")
    return [DummySession(s["session_id"], s["title"]) for s in raw_sessions]

def upsert_session_title(session_id, title):
    if not unified_db: return
    # Create if not exists (upsert)
    if not unified_db.db.sessions.find_one({"session_id": session_id}):
        unified_db.create_session("default_user", title)
        # However create_session generates a new ID. We want to force this session_id.
        unified_db.db.sessions.insert_one({
            "session_id": session_id,
            "user_id": "default_user",
            "title": title,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })
    else:
        unified_db.update_session_title(session_id, title)

def delete_session(session_id):
    if not unified_db: return
    unified_db.delete_session(session_id)
