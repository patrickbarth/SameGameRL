"""
Tests for optional database dependency handling.

Validates that the system gracefully handles missing database dependencies
and falls back to pickle storage when appropriate.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile

from samegamerl.game.game_config import GameFactory


class TestDatabaseAvailability:
    """Test database availability detection"""

    def test_database_available_flag_exists(self):
        """Verify DATABASE_AVAILABLE flag is accessible"""
        from samegamerl.database.availability import DATABASE_AVAILABLE

        assert isinstance(DATABASE_AVAILABLE, bool)

    def test_database_import_error_when_unavailable(self):
        """Verify import error is captured when database unavailable"""
        from samegamerl.database.availability import (
            DATABASE_AVAILABLE,
            DATABASE_IMPORT_ERROR,
        )

        if not DATABASE_AVAILABLE:
            assert DATABASE_IMPORT_ERROR is not None
            assert isinstance(DATABASE_IMPORT_ERROR, str)
        else:
            # When available, error should be None
            assert DATABASE_IMPORT_ERROR is None


class TestBenchmarkRepositoryFactory:
    """Test factory behavior with optional dependencies"""

    def test_factory_imports_successfully(self):
        """Factory should import regardless of database availability"""
        from samegamerl.evaluation.benchmark_repository_factory import (
            BenchmarkRepositoryFactory,
        )

        assert BenchmarkRepositoryFactory is not None

    def test_is_database_available_method(self):
        """Test the is_database_available helper method"""
        from samegamerl.evaluation.benchmark_repository_factory import (
            BenchmarkRepositoryFactory,
        )

        result = BenchmarkRepositoryFactory.is_database_available()
        assert isinstance(result, bool)

    def test_get_supported_types_includes_pickle(self):
        """Pickle should always be supported"""
        from samegamerl.evaluation.benchmark_repository_factory import (
            BenchmarkRepositoryFactory,
        )

        types = BenchmarkRepositoryFactory.get_supported_types()
        assert "pickle" in types
        assert isinstance(types, list)

    def test_get_supported_types_database_when_available(self):
        """Database should be in supported types when available"""
        from samegamerl.evaluation.benchmark_repository_factory import (
            BenchmarkRepositoryFactory,
        )
        from samegamerl.database.availability import DATABASE_AVAILABLE

        types = BenchmarkRepositoryFactory.get_supported_types()

        if DATABASE_AVAILABLE:
            assert "database" in types
        else:
            assert "database" not in types

    def test_create_pickle_repository(self):
        """Test creating pickle repository"""
        from samegamerl.evaluation.benchmark_repository_factory import (
            BenchmarkRepositoryFactory,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "test.pkl"
            repo = BenchmarkRepositoryFactory.create(
                storage_type="pickle", benchmark_path=path
            )

            assert repo is not None
            assert "Pickle" in repo.__class__.__name__

    def test_create_pickle_repository_requires_path(self):
        """Test that pickle repository requires benchmark_path"""
        from samegamerl.evaluation.benchmark_repository_factory import (
            BenchmarkRepositoryFactory,
        )

        with pytest.raises(ValueError, match="benchmark_path is required"):
            BenchmarkRepositoryFactory.create(storage_type="pickle")

    def test_create_database_repository_when_available(self):
        """Test creating database repository when dependencies available"""
        from samegamerl.evaluation.benchmark_repository_factory import (
            BenchmarkRepositoryFactory,
        )
        from samegamerl.database.availability import DATABASE_AVAILABLE

        if not DATABASE_AVAILABLE:
            pytest.skip("Database dependencies not installed")

        config = GameFactory.small()
        repo = BenchmarkRepositoryFactory.create(
            storage_type="database", config=config, base_seed=42
        )

        assert repo is not None
        assert "Database" in repo.__class__.__name__

    def test_create_database_repository_fallback_when_unavailable(self):
        """Test fallback to pickle when database requested but unavailable"""
        from samegamerl.evaluation.benchmark_repository_factory import (
            BenchmarkRepositoryFactory,
        )
        from samegamerl.database.availability import DATABASE_AVAILABLE

        if DATABASE_AVAILABLE:
            pytest.skip("Database dependencies are installed - cannot test fallback")

        config = GameFactory.small()

        # Should fall back to pickle without raising error
        repo = BenchmarkRepositoryFactory.create(
            storage_type="database", config=config, base_seed=42
        )

        assert repo is not None
        assert "Pickle" in repo.__class__.__name__

    def test_invalid_storage_type_raises_error(self):
        """Test that invalid storage type raises ValueError"""
        from samegamerl.evaluation.benchmark_repository_factory import (
            BenchmarkRepositoryFactory,
        )

        with pytest.raises(ValueError, match="Invalid storage_type"):
            BenchmarkRepositoryFactory.create(storage_type="invalid")


class TestDatabaseLazyImports:
    """Test that database package imports are lazy"""

    def test_database_package_imports_without_error(self):
        """Database package should import even without dependencies"""
        # This should not raise ImportError
        import samegamerl.database

        assert samegamerl.database is not None

    def test_database_exports_availability_flag(self):
        """Database package should always export DATABASE_AVAILABLE"""
        from samegamerl.database import DATABASE_AVAILABLE

        assert isinstance(DATABASE_AVAILABLE, bool)

    def test_database_exports_classes_when_available(self):
        """When available, database classes should be exported"""
        from samegamerl.database.availability import DATABASE_AVAILABLE

        if DATABASE_AVAILABLE:
            from samegamerl.database import (
                DatabaseManager,
                DatabaseRepository,
                GamePool,
            )

            assert DatabaseManager is not None
            assert DatabaseRepository is not None
            assert GamePool is not None
        else:
            # Should not be able to import these when unavailable
            with pytest.raises(ImportError):
                from samegamerl.database import DatabaseManager


class TestBenchmarkIntegration:
    """Test Benchmark class integration with optional dependencies"""

    def test_benchmark_accepts_database_storage_type(self):
        """Benchmark should accept storage_type parameter"""
        from samegamerl.evaluation.benchmark import Benchmark
        from samegamerl.database.availability import DATABASE_AVAILABLE

        config = GameFactory.small()

        if DATABASE_AVAILABLE:
            benchmark = Benchmark(
                config=config, num_games=10, storage_type="database"
            )
            assert "Database" in benchmark.repository.__class__.__name__

    def test_benchmark_accepts_pickle_storage_type(self):
        """Benchmark should accept pickle storage type"""
        from samegamerl.evaluation.benchmark import Benchmark

        config = GameFactory.small()

        benchmark = Benchmark(config=config, num_games=10, storage_type="pickle")
        assert "Pickle" in benchmark.repository.__class__.__name__

    def test_benchmark_falls_back_gracefully(self):
        """Benchmark should fall back to pickle when database unavailable"""
        from samegamerl.evaluation.benchmark import Benchmark
        from samegamerl.database.availability import DATABASE_AVAILABLE

        if DATABASE_AVAILABLE:
            pytest.skip("Database available - cannot test fallback")

        config = GameFactory.small()

        # Should work without error despite requesting database
        benchmark = Benchmark(config=config, num_games=10, storage_type="database")
        assert benchmark is not None
        assert "Pickle" in benchmark.repository.__class__.__name__


class TestMockedDatabaseUnavailable:
    """Test behavior when database is mocked as unavailable"""

    @patch("samegamerl.evaluation.benchmark_repository_factory.DATABASE_AVAILABLE", False)
    def test_factory_returns_pickle_when_database_mocked_unavailable(self):
        """Test fallback when DATABASE_AVAILABLE is mocked to False"""
        from samegamerl.evaluation.benchmark_repository_factory import (
            BenchmarkRepositoryFactory,
        )

        config = GameFactory.small()

        repo = BenchmarkRepositoryFactory.create(
            storage_type="database", config=config, base_seed=42
        )

        assert "Pickle" in repo.__class__.__name__

    @patch("samegamerl.evaluation.benchmark_repository_factory.DATABASE_AVAILABLE", False)
    def test_supported_types_excludes_database_when_mocked(self):
        """Test that database is not in supported types when mocked unavailable"""
        from samegamerl.evaluation.benchmark_repository_factory import (
            BenchmarkRepositoryFactory,
        )

        types = BenchmarkRepositoryFactory.get_supported_types()

        assert "pickle" in types
        assert "database" not in types