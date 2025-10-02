"""Check if database dependencies are available."""

DATABASE_AVAILABLE = False
DATABASE_IMPORT_ERROR = None

try:
    import asyncpg
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from dotenv import load_dotenv

    DATABASE_AVAILABLE = True
except ImportError as e:
    DATABASE_IMPORT_ERROR = str(e)
    DATABASE_AVAILABLE = False
