from markdown_it import MarkdownIt
import string


def escape_label(text: str) -> str:
    return text.replace("\n", " ").translate(str.maketrans({char: "\\" + char for char in string.punctuation}))


def render_answer_html(markdown: str, allowed_urls: set[str]) -> str:
    """Render untrusted model text without remote images, raw HTML, or arbitrary links."""
    parser = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = parser.parse(markdown)

    def clean(items):
        links = []
        for token in items:
            if token.type == "image":
                token.type = "text"
                token.tag = ""
                token.content = token.content or "[图片已省略]"
                token.children = None
                token.attrs = {}
            elif token.type == "link_open":
                keep = token.attrGet("href") in allowed_urls
                links.append(keep)
                if keep:
                    token.attrs = {"href": token.attrGet("href"), "rel": "noreferrer noopener"}
                else:
                    token.type, token.tag, token.nesting = "text", "", 0
                    token.content, token.attrs = "", {}
            elif token.type == "link_close" and links:
                if not links.pop():
                    token.type, token.tag, token.nesting = "text", "", 0
                    token.content = ""
            if token.children:
                clean(token.children)

    clean(tokens)
    return parser.renderer.render(tokens, parser.options, {})
