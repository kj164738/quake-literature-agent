from pathlib import Path

from streamlit.testing.v1 import AppTest


def launch(monkeypatch, tmp_path):
    monkeypatch.setenv("PAPER_LIBRARY_DIR", str(tmp_path / "library"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("SAMPLE_PAPER_DIR", str(Path("sample_papers").resolve()))
    return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=20).run()


def test_offline_question_survives_settings_rerun(monkeypatch, tmp_path):
    app = launch(monkeypatch, tmp_path)
    assert not app.exception
    app.sidebar.radio[0].set_value("仅本地资料")
    app.text_area[0].set_value("地震预警为什么需要快速估计震级？")
    app.button(key="FormSubmitter:research_question-检索并回答").click().run()
    assert not app.exception
    assert len(app.session_state["research_history"]) == 1
    assert app.session_state["research_history"][0]["result"].trace.status == "demo"
    app.sidebar.checkbox[0].set_value(False).run()
    assert not app.exception
    assert len(app.session_state["research_history"]) == 1
    assert any("资料范围已变化" in x.value for x in app.warning)


def test_empty_input_and_empty_corpus_are_handled(monkeypatch, tmp_path):
    app = launch(monkeypatch, tmp_path)
    app.button(key="FormSubmitter:research_question-检索并回答").click().run()
    assert any("请先输入" in x.value for x in app.warning)
    app.sidebar.checkbox[0].set_value(False)
    app.sidebar.radio[0].set_value("仅本地资料")
    app.text_area[0].set_value("地震预警？")
    app.button(key="FormSubmitter:research_question-检索并回答").click().run()
    assert not app.exception
    assert app.session_state["research_history"][-1]["result"].trace.refused
