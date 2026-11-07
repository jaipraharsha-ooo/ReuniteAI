"""SQLite storage for ReuniteAI report records.

Includes additive migrations for older local demo databases. Existing rows are
preserved. Reports are owned by a user ID and retain created_by email as a
backward-compatible fallback for records written by older app versions.
"""
from pathlib import Path
import sqlite3
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "reuniteai.db"
UPLOAD_DIR = PROJECT_ROOT / "uploads"

REPORT_COLUMNS = [
    "report_id", "item_name", "category", "description",
    "location", "date_found", "status", "source_type",
]


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    """Create the report table and safely add columns missing in older DBs."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS item_reports (
                report_id TEXT PRIMARY KEY,
                report_type TEXT NOT NULL CHECK(report_type IN ('lost', 'found')),
                item_name TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                location TEXT DEFAULT '',
                report_date TEXT NOT NULL,
                status TEXT NOT NULL,
                image_path TEXT DEFAULT '',
                contact_name TEXT DEFAULT '',
                contact_info TEXT DEFAULT '',
                created_by TEXT DEFAULT '',
                owner_user_id INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(item_reports)")}
        migrations = {
            "contact_name": "TEXT DEFAULT ''",
            "contact_info": "TEXT DEFAULT ''",
            "created_by": "TEXT DEFAULT ''",
            "owner_user_id": "INTEGER",
            "image_path": "TEXT DEFAULT ''",
            "created_at": "TEXT DEFAULT CURRENT_TIMESTAMP",
        }
        for column, declaration in migrations.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE item_reports ADD COLUMN {column} {declaration}")


def add_report(
    report_type, item_name, category, description, location,
    report_date, image_path="", contact_name="", contact_info="",
    created_by="", owner_user_id=None,
):
    """Save a report with the signed-in owner's ID and email."""
    if report_type not in ("lost", "found"):
        raise ValueError("Report type must be lost or found.")
    item_name = str(item_name or "").strip()
    description = str(description or "").strip()
    if not item_name or not description:
        raise ValueError("Item name and description are required.")
    if len(item_name) > 200:
        raise ValueError("Item name must be 200 characters or fewer.")
    if len(description) > 2000:
        raise ValueError("Description must be 2000 characters or fewer.")

    owner_id = int(owner_user_id) if owner_user_id is not None else None
    owner_email = str(created_by or "").strip().lower()
    if owner_id is None and not owner_email:
        raise ValueError("A signed-in account is required to own a report.")

    with get_connection() as conn:
        prefix = "L" if report_type == "lost" else "F"
        rows = conn.execute(
            "SELECT report_id FROM item_reports WHERE report_type = ?", (report_type,)
        ).fetchall()
        numbers = []
        for row in rows:
            try:
                numbers.append(int(str(row["report_id"]).split("-", 1)[1]))
            except (ValueError, IndexError):
                continue
        report_id = f"{prefix}-{max(numbers, default=0) + 1:04d}"
        status = "Missing" if report_type == "lost" else "Available"
        conn.execute("""
            INSERT INTO item_reports (
                report_id, report_type, item_name, category, description,
                location, report_date, status, image_path, contact_name,
                contact_info, created_by, owner_user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            report_id, report_type, item_name, str(category or "").strip(), description,
            str(location or "").strip(), str(report_date), status, str(image_path or ""),
            str(contact_name or "").strip(), str(contact_info or "").strip(),
            owner_email, owner_id,
        ))
    return report_id


def get_reports(report_type=None, created_by=None, owner_user_id=None):
    """Read reports; with owner arguments, return only that user's reports.

    If a user ID is supplied, rows with that owner_user_id are included, plus
    older rows with no owner ID whose created_by email matches the account.
    Unassigned legacy rows are deliberately excluded from personal views.
    """
    clauses = []
    params = []
    if report_type in ("lost", "found"):
        clauses.append("report_type = ?")
        params.append(report_type)

    if owner_user_id is not None:
        clauses.append(
            "(owner_user_id = ? OR (owner_user_id IS NULL AND LOWER(COALESCE(created_by, '')) = LOWER(?)))"
        )
        params.extend([int(owner_user_id), str(created_by or "").strip()])
    elif created_by:
        clauses.append("LOWER(COALESCE(created_by, '')) = LOWER(?)")
        params.append(str(created_by).strip())

    query = "SELECT * FROM item_reports"
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY created_at DESC, report_id DESC"

    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return pd.DataFrame([dict(row) for row in rows])


def get_found_reports_for_matcher(seed_csv="data/found_items.csv"):
    """Combine sample and user-submitted found reports for community matching."""
    seed_path = PROJECT_ROOT / seed_csv
    frames = []
    if seed_path.exists():
        seed = pd.read_csv(seed_path).fillna("")
        if "date_found" not in seed.columns:
            seed["date_found"] = ""
        frames.append(seed)

    submitted = get_reports("found")  # Candidate reports are shared for matching.
    if not submitted.empty:
        submitted = submitted.rename(columns={"report_date": "date_found"})
        frames.append(submitted)

    if not frames:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    combined = pd.concat(frames, ignore_index=True).fillna("")
    for col in REPORT_COLUMNS:
        if col not in combined.columns:
            combined[col] = ""
    combined = combined[REPORT_COLUMNS]
    combined = combined[combined["description"].astype(str).str.strip().ne("")]
    combined = combined.drop_duplicates(subset=["report_id"], keep="last")
    return combined.reset_index(drop=True)
