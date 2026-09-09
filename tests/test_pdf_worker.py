import pytest

from quake_agent.pdf_worker import load_pdf_bounded


def test_pdf_parse_timeout_terminates_worker(tmp_path):
    path = tmp_path / "test.pdf"
    path.write_bytes(b"%PDF invalid")
    with pytest.raises(TimeoutError):
        load_pdf_bounded(path, timeout=0)


def test_pdf_worker_returns_parse_errors(tmp_path):
    path = tmp_path / "test.pdf"
    path.write_bytes(b"%PDF invalid")
    with pytest.raises(ValueError):
        load_pdf_bounded(path)
