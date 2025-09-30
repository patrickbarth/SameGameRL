"""Database initialization script."""

import asyncio
from pathlib import Path

from samegamerl.database.connection import db_manager


async def create_schema():
    """Create database schema from SQL file."""
    schema_file = Path(__file__).parent / "schema.sql"

    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_file}")

    schema_sql = schema_file.read_text()

    conn = await db_manager.get_async_connection()
    try:
        # Execute schema creation
        await conn.execute(schema_sql)
        print("✅ Database schema created successfully")

        # Verify tables were created
        tables = await conn.fetch(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
        """
        )

        print(f"📋 Created {len(tables)} tables:")
        for table in tables:
            print(f"   - {table['table_name']}")

    finally:
        await conn.close()


async def test_connection():
    """Test database connection."""
    print("🔌 Testing database connection...")

    if await db_manager.test_connection():
        print("✅ Database connection successful")
        return True
    else:
        print("❌ Database connection failed")
        return False


async def main():
    """Initialize database."""
    print("🚀 Initializing SameGameRL database...")

    # Test connection first
    if not await test_connection():
        print("\n💡 Make sure PostgreSQL is running:")
        print("   brew services start postgresql")
        print("   createdb samegamerl_dev")
        return

    # Create schema
    await create_schema()

    print("\n✨ Database initialization complete!")


if __name__ == "__main__":
    asyncio.run(main())
