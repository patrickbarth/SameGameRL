#!/usr/bin/env python3
"""Test script to demonstrate optional database dependencies."""

from samegamerl.database.availability import DATABASE_AVAILABLE, DATABASE_IMPORT_ERROR
from samegamerl.evaluation.benchmark_repository_factory import BenchmarkRepositoryFactory

print("=" * 60)
print("Database Dependency Status")
print("=" * 60)
print(f"Database available: {DATABASE_AVAILABLE}")

if DATABASE_AVAILABLE:
    print("✓ All database dependencies are installed")
    print("  You can use storage_type='database'")
else:
    print("✗ Database dependencies not available")
    print(f"  Import error: {DATABASE_IMPORT_ERROR}")
    print("  Install with: poetry install -E database")

print()
print("Supported storage types:", BenchmarkRepositoryFactory.get_supported_types())
print("=" * 60)