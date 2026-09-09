import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from quake_agent.agent import LiteratureAgent
from quake_agent.config import load_settings
from quake_agent.document_loader import PaperChunk
from quake_agent.llm import build_chat_llm
from quake_agent.vector_store import LocalKnowledgeBase


def test_real_chat_client_against_local_protocol_server(monkeypatch):
    captured = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            captured.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            payload = json.dumps({"id": "chatcmpl-local-test", "object": "chat.completion",
                "created": 1, "model": "local-test",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Magnitude is estimated quickly [1]."},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setenv("MODEL_PROVIDER", "deepseek")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "local-test-only")
        monkeypatch.setenv("DEEPSEEK_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
        llm = build_chat_llm(load_settings())
        kb = LocalKnowledgeBase(use_chroma=False)
        kb.build([PaperChunk("Magnitude is estimated quickly.", "paper.md")])
        result = LiteratureAgent(kb, llm, lambda q, k: [], arxiv_mode="off").answer("magnitude")
        assert result.trace.status == "answered"
        assert result.trace.model_calls == 1
        assert len(captured) == 1
        assert captured[0]["messages"][0]["role"] == "system"
        assert "evidence" in captured[0]["messages"][1]["content"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
