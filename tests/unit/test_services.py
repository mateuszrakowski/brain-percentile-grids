"""Unit tests for ReferenceDataService and ModelPersistenceService.

Tests the service layer directly against an in-memory SQLite database,
bypassing the HTTP layer. The test_dataset_db fixture provides an empty
dataset, while test_reference_dataset provides one pre-populated with
two patient records and a "hippo" structure column.
"""

from unittest.mock import Mock

import pandas as pd
import pytest

from app.fastapi.services.model_persistence import ModelPersistenceService
from app.fastapi.services.reference_data import ReferenceDataService


class TestReferenceDataService:
    """Tests for ReferenceDataService database operations."""

    def test_save_reference_data(self, test_session, test_dataset_db):
        """Verify records are inserted and ProcessingResult reflects the operation."""
        reference_service = ReferenceDataService(session=test_session)
        result = reference_service.save_reference_data(
            dataset_id=test_dataset_db.id,
            dataframes=[
                pd.DataFrame(
                    {
                        "PatientID": ["p1", "p2"],
                        "AgeYears": [25, 35],
                        "StudyDate": ["2024-01-01", "2024-01-02"],
                        "StudyDescription": ["scan1", "scan2"],
                        "hippo": [0.5, 0.6],
                    }
                )
            ],
        )

        assert result.records_added == 2
        assert result.files_processed == 1
        assert result.duplicates_found == 0
        assert result.total_records == 2
        assert result.structures == ["hippo"]

    def test_save_reference_data_duplicates(self, test_session, test_reference_dataset):
        """Verify saving the same records again detects them as duplicates."""
        service = ReferenceDataService(session=test_session)
        result = service.save_reference_data(
            dataset_id=test_reference_dataset.id,
            dataframes=[
                pd.DataFrame(
                    {
                        "PatientID": ["p1", "p2"],
                        "AgeYears": [25, 35],
                        "StudyDate": ["2024-01-01", "2024-01-02"],
                        "StudyDescription": ["scan1", "scan2"],
                        "hippo": [0.5, 0.6],
                    }
                )
            ],
        )

        assert result.duplicates_found == 2
        assert result.records_added == 0
        assert result.total_records == 2

    def test_get_reference_summary(self, test_session, test_reference_dataset):
        """Verify summary returns correct counts, structures, and sample records."""
        service = ReferenceDataService(session=test_session)
        summary = service.get_reference_summary(test_reference_dataset.id)

        assert summary.total_records == 2
        assert summary.structures == ["hippo"]
        assert len(summary.sample) == 2
        assert summary.sample[0].patient_id == "p1"
        assert summary.sample[0].study_date == "2024-01-01"
        assert summary.sample[1].patient_id == "p2"
        assert summary.sample[0].created_at is not None

    def test_get_reference_dataframe(self, test_session, test_reference_dataset):
        """Verify dataframe reconstructs patient records with structure columns."""
        service = ReferenceDataService(session=test_session)
        df = service.get_reference_dataframe(test_reference_dataset.id)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "PatientID" in df.columns
        assert "AgeYears" in df.columns
        assert "hippo" in df.columns
        assert list(df["PatientID"]) == ["p1", "p2"]
        assert list(df["hippo"]) == [0.5, 0.6]

    def test_get_reference_summary_empty(self, test_session, test_dataset_db):
        """Verify empty dataset returns a ReferenceSummary with zero defaults."""
        service = ReferenceDataService(session=test_session)
        summary = service.get_reference_summary(test_dataset_db.id)

        assert summary.total_records == 0
        assert summary.structures == []
        assert summary.sample == []

    def test_get_reference_dataframe_empty(self, test_session, test_dataset_db):
        """Verify empty dataset returns an empty DataFrame."""
        service = ReferenceDataService(session=test_session)
        df = service.get_reference_dataframe(test_dataset_db.id)

        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_clear_reference_data(self, test_session, test_reference_dataset):
        """Verify clearing a populated dataset removes all records and returns the count."""
        service = ReferenceDataService(session=test_session)
        record_count = service.clear_reference_data(test_reference_dataset.id)

        assert record_count == 2

        # Confirm data is actually gone
        summary = service.get_reference_summary(test_reference_dataset.id)
        assert summary.total_records == 0

    def test_clear_reference_data_empty(self, test_session, test_dataset_db):
        """Verify clearing an empty dataset returns zero and does not error."""
        service = ReferenceDataService(session=test_session)
        record_count = service.clear_reference_data(test_dataset_db.id)

        assert record_count == 0

    def test_get_available_structures(self, test_session, test_reference_dataset):
        """Verify available structures match those in the populated dataset."""
        service = ReferenceDataService(session=test_session)
        structures = service.get_available_structures(test_reference_dataset.id)

        assert structures == ["hippo"]

    def test_get_available_structures_empty(self, test_session, test_dataset_db):
        """Verify empty dataset returns no structures."""
        service = ReferenceDataService(session=test_session)
        structures = service.get_available_structures(test_dataset_db.id)

        assert structures == []


class TestModelPersistenceService:
    """Tests for ModelPersistenceService database and file operations."""

    @pytest.fixture
    def persistence_service(self, test_session, tmp_path, monkeypatch):
        """Create a ModelPersistenceService with settings pointing to tmp_path."""
        monkeypatch.setattr(
            "app.fastapi.services.model_persistence.settings",
            Mock(models_dir=str(tmp_path)),
        )
        return ModelPersistenceService(session=test_session)

    def test_save_model(self, persistence_service, test_dataset_db):
        """Verify save_model creates a DB record and calls fitted_model.save()."""
        mock_fitted_model = Mock()
        mock_fitted_model.aic = 100.0
        mock_fitted_model.bic = 110.0
        mock_fitted_model.model.rx2.return_value = ["NO"]

        result = persistence_service.save_model(
            fitted_model=mock_fitted_model,
            user_id=test_dataset_db.user_id,
            dataset_id=test_dataset_db.id,
            structure="hippo",
        )

        assert result.dataset_id == test_dataset_db.id
        assert result.structure == "hippo"
        assert result.family == "NO"
        assert result.aic == 100.0
        assert result.bic == 110.0
        assert "hippo.rds" in result.file_path
        mock_fitted_model.save.assert_called_once()

    def test_load_model(
        self, persistence_service, test_session, test_dataset_db, tmp_path, monkeypatch
    ):
        """Verify load_model queries the DB record and delegates to GAMLSS.load_model."""
        from app.fastapi.db.models import FittedModel

        # Create a dummy file on disk
        model_file = tmp_path / "hippo.rds"
        model_file.write_text("dummy")

        db_model = FittedModel(
            dataset_id=test_dataset_db.id,
            structure="hippo",
            family="NO",
            aic=100.0,
            bic=110.0,
            file_path=str(model_file),
        )
        test_session.add(db_model)
        test_session.commit()

        # Mock GAMLSS.load_model to avoid R
        mock_loaded = Mock()
        monkeypatch.setattr(
            "app.fastapi.services.model_persistence.GAMLSS.load_model",
            Mock(return_value=mock_loaded),
        )

        result = persistence_service.load_model(
            dataset_id=test_dataset_db.id,
            structure="hippo",
            source_data=pd.DataFrame(),
            x_column="AgeYears",
            percentiles=[0.5],
        )

        assert result is mock_loaded

    def test_load_model_file_missing(
        self, persistence_service, test_session, test_dataset_db, tmp_path
    ):
        """Verify load_model returns None when DB record exists but file is missing."""
        from app.fastapi.db.models import FittedModel

        db_model = FittedModel(
            dataset_id=test_dataset_db.id,
            structure="hippo",
            family="NO",
            aic=100.0,
            bic=110.0,
            file_path=str(tmp_path / "nonexistent.rds"),
        )
        test_session.add(db_model)
        test_session.commit()

        result = persistence_service.load_model(
            dataset_id=test_dataset_db.id,
            structure="hippo",
            source_data=pd.DataFrame(),
            x_column="AgeYears",
            percentiles=[0.5],
        )

        assert result is None

    def test_load_model_not_found(self, persistence_service, test_dataset_db):
        """Verify load_model returns None when no DB record exists."""
        result = persistence_service.load_model(
            dataset_id=test_dataset_db.id,
            structure="nonexistent",
            source_data=pd.DataFrame(),
            x_column="AgeYears",
            percentiles=[0.5],
        )

        assert result is None

    def test_get_dataset_models(
        self, persistence_service, test_session, test_dataset_db
    ):
        """Verify get_dataset_models returns inserted model records."""
        from app.fastapi.db.models import FittedModel

        test_session.add(
            FittedModel(
                dataset_id=test_dataset_db.id,
                structure="hippo",
                family="NO",
                aic=100.0,
                bic=110.0,
                file_path="/tmp/hippo.rds",
            )
        )
        test_session.add(
            FittedModel(
                dataset_id=test_dataset_db.id,
                structure="amygdala",
                family="NO",
                aic=200.0,
                bic=210.0,
                file_path="/tmp/amygdala.rds",
            )
        )
        test_session.commit()

        models = persistence_service.get_dataset_models(test_dataset_db.id)

        assert len(models) == 2
        assert {m.structure for m in models} == {"hippo", "amygdala"}

    def test_get_dataset_models_empty(self, persistence_service, test_dataset_db):
        """Verify get_dataset_models returns empty list for dataset with no models."""
        models = persistence_service.get_dataset_models(test_dataset_db.id)

        assert models == []

    def test_has_fitted_models(
        self, persistence_service, test_session, test_dataset_db
    ):
        """Verify has_fitted_models returns True when models exist."""
        from app.fastapi.db.models import FittedModel

        test_session.add(
            FittedModel(
                dataset_id=test_dataset_db.id,
                structure="hippo",
                family="NO",
                aic=100.0,
                bic=110.0,
                file_path="/tmp/hippo.rds",
            )
        )
        test_session.commit()

        assert persistence_service.has_fitted_models(test_dataset_db.id) is True

    def test_has_fitted_models_empty(self, persistence_service, test_dataset_db):
        """Verify has_fitted_models returns False for dataset with no models."""
        assert persistence_service.has_fitted_models(test_dataset_db.id) is False

    def test_delete_dataset_models(
        self, persistence_service, test_session, test_dataset_db, tmp_path
    ):
        """Verify delete_dataset_models removes DB records and model files from disk."""
        from app.fastapi.db.models import FittedModel

        # Create model directory and dummy file
        model_dir = (
            tmp_path
            / f"user_{test_dataset_db.user_id}"
            / f"dataset_{test_dataset_db.id}"
        )
        model_dir.mkdir(parents=True)
        (model_dir / "hippo.rds").write_text("dummy")

        test_session.add(
            FittedModel(
                dataset_id=test_dataset_db.id,
                structure="hippo",
                family="NO",
                aic=100.0,
                bic=110.0,
                file_path=str(model_dir / "hippo.rds"),
            )
        )
        test_session.commit()

        deleted = persistence_service.delete_dataset_models(
            user_id=test_dataset_db.user_id,
            dataset_id=test_dataset_db.id,
        )

        assert deleted == 1
        assert persistence_service.get_dataset_models(test_dataset_db.id) == []
        assert not model_dir.exists()

    def test_delete_dataset_models_empty(self, persistence_service, test_dataset_db):
        """Verify deleting models for a dataset with none returns zero and does not error."""
        deleted = persistence_service.delete_dataset_models(
            user_id=test_dataset_db.user_id,
            dataset_id=test_dataset_db.id,
        )

        assert deleted == 0
