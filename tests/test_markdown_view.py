from quake_agent.markdown_view import render_answer_html


def test_answer_cannot_exfiltrate_through_html_images_or_links():
    answer = '<img src="https://evil.test/raw">\n\n![tracking](https://evil.test/secret) [click](https://evil.test)'
    html = render_answer_html(answer, set())
    assert '<img' not in html
    assert '<a ' not in html
    assert 'src="https://' not in html


def test_only_vetted_source_links_survive_and_markdown_renders():
    url = "https://arxiv.org/abs/1234.5678"
    html = render_answer_html(f"**Claim** [paper]({url})\n\n- first\n- second", {url})
    assert '<strong>Claim</strong>' in html
    assert f'href="{url}"' in html
    assert '<ul>' in html


def test_reference_style_image_is_also_removed():
    html = render_answer_html('![x][remote]\n\n[remote]: https://evil.test/a', set())
    assert '<img' not in html
    assert 'https://evil.test' not in html


def test_exported_markdown_cannot_restore_remote_images():
    from markdown_it import MarkdownIt
    from quake_agent.agent import AgentAnswer, AgentTrace, Source
    from quake_agent.service import export_answer
    result = AgentAnswer("![secret](https://evil.test/a) Claim [1]", [],
                         [Source("![title](https://evil.test)", "```\n![body](https://evil.test)")], AgentTrace())
    rendered = MarkdownIt("commonmark", {"html": True}).render(export_answer("Question", result))
    assert "<img" not in rendered
    assert 'href="https://evil.test' not in rendered
