"""Simple local-demo authentication for ReuniteAI.

Passwords are salted and hashed with PBKDF2-HMAC-SHA256. This module is
intended for a local capstone demo, not as a replacement for a production
identity provider.
"""
from __future__ import annotations

import hashlib
import re
import secrets
import sqlite3
from typing import Any

from .database import get_connection

PBKDF2_ITERATIONS = 310_000
EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def initialize_auth_db() -> None:
    """Create the local user table if it does not already exist."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_users (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def _hash_password(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    ).hex()


def register_user(full_name: str, email: str, password: str) -> tuple[bool, str, dict[str, Any] | None]:
    """Register a user. Returns (success, message, public_user_record)."""
    name = str(full_name or "").strip()
    normalized_email = str(email or "").strip().lower()

    if len(name) < 2 or len(name) > 100:
        return False, "Enter your name (2–100 characters).", None
    if not EMAIL_PATTERN.match(normalized_email) or len(normalized_email) > 254:
        return False, "Enter a valid email address.", None
    if len(password) < 8:
        return False, "Use a password with at least 8 characters.", None
    if len(password) > 128:
        return False, "Password must be 128 characters or fewer.", None

    salt = secrets.token_bytes(16)
    password_hash = _hash_password(password, salt)
    try:
        with get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO app_users (full_name, email, password_salt, password_hash)
                   VALUES (?, ?, ?, ?)""",
                (name, normalized_email, salt.hex(), password_hash),
            )
            user_id = int(cursor.lastrowid)
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists. Please sign in.", None

    return True, "Account created successfully.", {
        "user_id": user_id,
        "full_name": name,
        "email": normalized_email,
    }


def verify_user(email: str, password: str) -> dict[str, Any] | None:
    """Check login credentials and return a safe user record, or None."""
    normalized_email = str(email or "").strip().lower()
    if not normalized_email or not password:
        return None

    with get_connection() as conn:
        row = conn.execute(
            """SELECT user_id, full_name, email, password_salt, password_hash
               FROM app_users WHERE email = ? COLLATE NOCASE""",
            (normalized_email,),
        ).fetchone()

    if row is None:
        # Perform a dummy hash to make the missing-account path less distinct.
        _hash_password(password, b"reuniteai-dummy-salt")
        return None

    try:
        salt = bytes.fromhex(row["password_salt"])
    except (TypeError, ValueError):
        return None

    candidate_hash = _hash_password(password, salt)
    if not secrets.compare_digest(candidate_hash, row["password_hash"]):
        return None

    return {
        "user_id": int(row["user_id"]),
        "full_name": str(row["full_name"]),
        "email": str(row["email"]),
    }
