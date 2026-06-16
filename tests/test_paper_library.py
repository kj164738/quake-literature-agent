from quake_agent.paper_library import delete_paper, list_papers, safe_filename, save_uploaded_papers


class FakeUpload:
    def __init__(self, name: str, content: bytes):
        self.name = name
        self._content = content

    def getbuffer(self):
        return memoryview(self._content)


def test_save_uploaded_papers_persists_supported_files(tmp_path):
    saved = save_uploaded_papers(
        [
            FakeUpload("quake early warning.md", b"earthquake warning"),
            FakeUpload("../unsafe.pdf", b"%PDF"),
            FakeUpload("ignore.exe", b"bad"),
        ],
        tmp_path,
    )

    names = sorted(paper.name for paper in saved)

    assert names == ["quake_early_warning.md", "unsafe.pdf"]
    assert (tmp_path / "quake_early_warning.md").read_bytes() == b"earthquake warning"
    assert not (tmp_path / "ignore.exe").exists()


def test_save_uploaded_papers_uses_unique_names(tmp_path):
    save_uploaded_papers([FakeUpload("paper.md", b"first")], tmp_path)
    save_uploaded_papers([FakeUpload("paper.md", b"second")], tmp_path)

    names = sorted(paper.name for paper in list_papers(tmp_path))

    assert names == ["paper.md", "paper_2.md"]


def test_delete_paper_removes_only_managed_file(tmp_path):
    save_uploaded_papers([FakeUpload("paper.md", b"text")], tmp_path)

    assert delete_paper(tmp_path, "paper.md") is True
    assert delete_paper(tmp_path, "../paper.md") is False
    assert list_papers(tmp_path) == []


def test_safe_filename_keeps_cjk_and_supported_suffix():
    assert safe_filename(" 地震 论文.md ") == "地震_论文.md"
