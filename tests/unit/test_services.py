import pandas as pd

from app.fastapi.services.reference_data import ReferenceDataService


class TestReferenceDataService:
    def test_save_reference_data(self, test_session, test_dataset_db):
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
