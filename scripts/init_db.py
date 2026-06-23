"""Initialize database tables (bootstrap migration)."""

from backend.database.session import init_db

if __name__ == "__main__":
    init_db()
    print("Database tables created successfully.")
