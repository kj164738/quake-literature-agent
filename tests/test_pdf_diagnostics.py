from quake_agent.pdf_diagnostics import build_pdf_report


def test_pdf_report_marks_extractable_text_as_ok():
    pages = [
        (1, "Earthquake early warning estimates magnitude and source location."),
        (2, "Seismic risk depends on hazard, exposure, and vulnerability."),
    ]

    report = build_pdf_report("paper.pdf", pages, low_text_threshold=10)

    assert report.status == "ok"
    assert report.page_count == 2
    assert report.pages_with_text == 2
    assert report.chunk_count >= 1
    assert report.low_text_pages == []


def test_pdf_report_detects_scanned_or_empty_pdf():
    report = build_pdf_report("scan.pdf", [(1, ""), (2, "")])

    assert report.status == "no_extractable_text"
    assert report.low_text_pages == [1, 2]
