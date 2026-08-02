"""Raw and shipment-grain data validation tests."""

from supplylens.data import parse_source_date
from supplylens.validation import validate_processed_data, validate_raw_data


def test_required_schema_and_checksum_pass(raw_data, raw_path, config):
    result = validate_raw_data(
        raw_data,
        source_path=raw_path,
        expected_sha256=config["data"]["sha256"],
        expected_rows=int(config["data"]["expected_rows"]),
        expected_columns=int(config["data"]["expected_columns"]),
    )
    assert result.passed, result.errors
    assert result.metrics["duplicate_rows"] == 0
    assert result.metrics["duplicate_ids"] == 0


def test_source_date_parsing_uses_real_rows(raw_data):
    parsed = parse_source_date(raw_data["Scheduled Delivery Date"])
    assert parsed.notna().all()
    assert parsed.min().year == 2006
    assert parsed.max().year == 2015


def test_processed_shipment_identifiers_are_unique(shipments):
    result = validate_processed_data(shipments)
    assert result.passed, result.errors
    assert shipments["shipment_id"].is_unique
    assert len(shipments) == 7030
