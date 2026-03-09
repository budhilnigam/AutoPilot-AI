"""Authentication and account connection persistence using SQLite."""

import base64
import hashlib
import hmac
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet, InvalidToken


class AuthStore:
    """SQLite-backed user auth and account connection store."""

    def __init__(self, db_path: str = "./local_kb/configs/autopilot_auth.db", key_path: str = "./local_kb/configs/auth.key"):
        self.db_path = Path(db_path)
        self.key_path = Path(key_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._fernet = Fernet(self._load_or_create_key())
        self._apply_migrations()

    def _load_or_create_key(self) -> bytes:
        if self.key_path.exists():
            return self.key_path.read_bytes().strip()

        key = Fernet.generate_key()
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        self.key_path.write_bytes(key)
        return key

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _apply_migrations(self) -> None:
        migrations = [
            ("001_initial_auth_schema", self._migration_001_initial_auth_schema),
            ("002_github_oauth_states", self._migration_002_github_oauth_states),
        ]

        with self._conn() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT NOT NULL UNIQUE,
                    applied_at TEXT NOT NULL
                )
                """
            )

            applied = {
                row["version"]
                for row in conn.execute("SELECT version FROM schema_migrations").fetchall()
            }

            for version, migration in migrations:
                if version in applied:
                    continue
                migration(conn)
                conn.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (version, datetime.utcnow().isoformat()),
                )

    def _migration_001_initial_auth_schema(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                token_hash TEXT NOT NULL UNIQUE,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                revoked_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS aws_connections (
                user_id INTEGER PRIMARY KEY,
                account_id TEXT,
                arn TEXT,
                region TEXT NOT NULL,
                access_key_id_enc TEXT NOT NULL,
                secret_access_key_enc TEXT NOT NULL,
                session_token_enc TEXT,
                validated_permissions TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );

            CREATE TABLE IF NOT EXISTS github_connections (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                token_enc TEXT NOT NULL,
                scopes TEXT,
                repo_owner TEXT,
                repo_name TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            """
        )

    def _migration_002_github_oauth_states(self, conn: sqlite3.Connection) -> None:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS github_oauth_states (
                state TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                repo_owner TEXT,
                repo_name TEXT,
                expires_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                consumed_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            """
        )

    def _hash_password(self, password: str, salt: Optional[bytes] = None) -> tuple[str, str]:
        salt_bytes = salt or os.urandom(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 180000)
        return base64.b64encode(digest).decode("utf-8"), base64.b64encode(salt_bytes).decode("utf-8")

    def _encrypt(self, value: str) -> str:
        return self._fernet.encrypt(value.encode("utf-8")).decode("utf-8")

    def _decrypt(self, value: str) -> str:
        try:
            return self._fernet.decrypt(value.encode("utf-8")).decode("utf-8")
        except (InvalidToken, ValueError):
            return ""

    def create_user(self, email: str, password: str) -> Dict[str, Any]:
        now = datetime.utcnow().isoformat()
        password_hash, password_salt = self._hash_password(password)

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO users (email, password_hash, password_salt, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (email.lower().strip(), password_hash, password_salt, now, now),
            )
            user_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        return {"id": user_id, "email": email.lower().strip(), "created_at": now}

    def authenticate(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower().strip(),)).fetchone()
            if not row:
                return None

            salt = base64.b64decode(row["password_salt"])
            computed_hash, _ = self._hash_password(password, salt=salt)
            if not hmac.compare_digest(computed_hash, row["password_hash"]):
                return None

            return {"id": row["id"], "email": row["email"]}

    def create_session(self, user_id: int, ttl_hours: int = 24) -> str:
        token = secrets.token_urlsafe(48)
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        now = datetime.utcnow()
        expires_at = now + timedelta(hours=ttl_hours)

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO sessions (user_id, token_hash, expires_at, created_at, revoked_at)
                VALUES (?, ?, ?, ?, NULL)
                """,
                (user_id, token_hash, expires_at.isoformat(), now.isoformat()),
            )

        return token

    def revoke_session(self, token: str) -> None:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET revoked_at = ? WHERE token_hash = ?",
                (datetime.utcnow().isoformat(), token_hash),
            )

    def get_user_by_session(self, token: str) -> Optional[Dict[str, Any]]:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._conn() as conn:
            row = conn.execute(
                """
                SELECT u.id, u.email, s.expires_at, s.revoked_at
                FROM sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.token_hash = ?
                """,
                (token_hash,),
            ).fetchone()

            if not row:
                return None
            if row["revoked_at"]:
                return None
            if datetime.fromisoformat(row["expires_at"]) < datetime.utcnow():
                return None

            return {"id": row["id"], "email": row["email"]}

    def upsert_aws_connection(
        self,
        user_id: int,
        account_id: str,
        arn: str,
        region: str,
        access_key_id: str,
        secret_access_key: str,
        session_token: Optional[str],
        validated_permissions: str,
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO aws_connections (
                    user_id, account_id, arn, region, access_key_id_enc, secret_access_key_enc,
                    session_token_enc, validated_permissions, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    account_id = excluded.account_id,
                    arn = excluded.arn,
                    region = excluded.region,
                    access_key_id_enc = excluded.access_key_id_enc,
                    secret_access_key_enc = excluded.secret_access_key_enc,
                    session_token_enc = excluded.session_token_enc,
                    validated_permissions = excluded.validated_permissions,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    account_id,
                    arn,
                    region,
                    self._encrypt(access_key_id),
                    self._encrypt(secret_access_key),
                    self._encrypt(session_token) if session_token else None,
                    validated_permissions,
                    now,
                    now,
                ),
            )

    def upsert_github_connection(
        self,
        user_id: int,
        username: str,
        token: str,
        scopes: str,
        repo_owner: Optional[str],
        repo_name: Optional[str],
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO github_connections (user_id, username, token_enc, scopes, repo_owner, repo_name, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    username = excluded.username,
                    token_enc = excluded.token_enc,
                    scopes = excluded.scopes,
                    repo_owner = excluded.repo_owner,
                    repo_name = excluded.repo_name,
                    updated_at = excluded.updated_at
                """,
                (
                    user_id,
                    username,
                    self._encrypt(token),
                    scopes,
                    repo_owner,
                    repo_name,
                    now,
                    now,
                ),
            )

    def update_github_repository(self, user_id: int, repo_owner: Optional[str], repo_name: Optional[str]) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE github_connections
                SET repo_owner = ?, repo_name = ?, updated_at = ?
                WHERE user_id = ?
                """,
                (
                    repo_owner,
                    repo_name,
                    datetime.utcnow().isoformat(),
                    user_id,
                ),
            )

    def delete_aws_connection(self, user_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM aws_connections WHERE user_id = ?", (user_id,))

    def delete_github_connection(self, user_id: int) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM github_connections WHERE user_id = ?", (user_id,))

    def create_github_oauth_state(
        self,
        user_id: int,
        state: str,
        repo_owner: Optional[str] = None,
        repo_name: Optional[str] = None,
        ttl_minutes: int = 10,
    ) -> None:
        now = datetime.utcnow()
        expires_at = now + timedelta(minutes=ttl_minutes)
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO github_oauth_states (state, user_id, repo_owner, repo_name, expires_at, created_at, consumed_at)
                VALUES (?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    state,
                    user_id,
                    repo_owner,
                    repo_name,
                    expires_at.isoformat(),
                    now.isoformat(),
                ),
            )

    def consume_github_oauth_state(self, state: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM github_oauth_states WHERE state = ?",
                (state,),
            ).fetchone()
            if not row:
                return None
            if row["consumed_at"]:
                return None
            if datetime.fromisoformat(row["expires_at"]) < datetime.utcnow():
                return None

            conn.execute(
                "UPDATE github_oauth_states SET consumed_at = ? WHERE state = ?",
                (datetime.utcnow().isoformat(), state),
            )

            return {
                "state": row["state"],
                "user_id": row["user_id"],
                "repo_owner": row["repo_owner"],
                "repo_name": row["repo_name"],
            }

    def get_aws_connection(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM aws_connections WHERE user_id = ?", (user_id,)).fetchone()
            if not row:
                return None
            return {
                "user_id": row["user_id"],
                "account_id": row["account_id"],
                "arn": row["arn"],
                "region": row["region"],
                "access_key_id": self._decrypt(row["access_key_id_enc"]),
                "secret_access_key": self._decrypt(row["secret_access_key_enc"]),
                "session_token": self._decrypt(row["session_token_enc"]) if row["session_token_enc"] else None,
                "validated_permissions": row["validated_permissions"] or "",
            }

    def get_github_connection(self, user_id: int) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM github_connections WHERE user_id = ?", (user_id,)).fetchone()
            if not row:
                return None
            return {
                "user_id": row["user_id"],
                "username": row["username"],
                "token": self._decrypt(row["token_enc"]),
                "scopes": row["scopes"] or "",
                "repo_owner": row["repo_owner"],
                "repo_name": row["repo_name"],
            }

    def get_connection_status(self, user_id: int) -> Dict[str, Any]:
        aws = self.get_aws_connection(user_id)
        gh = self.get_github_connection(user_id)
        return {
            "aws_connected": aws is not None,
            "github_connected": gh is not None,
            "aws": {
                "account_id": aws.get("account_id") if aws else None,
                "arn": aws.get("arn") if aws else None,
                "region": aws.get("region") if aws else None,
                "validated_permissions": aws.get("validated_permissions") if aws else None,
            },
            "github": {
                "username": gh.get("username") if gh else None,
                "scopes": gh.get("scopes") if gh else None,
                "repo_owner": gh.get("repo_owner") if gh else None,
                "repo_name": gh.get("repo_name") if gh else None,
            },
        }
