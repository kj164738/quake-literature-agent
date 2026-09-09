from __future__ import annotations

import multiprocessing
from pathlib import Path


def _parse(path: str, connection):
    try:
        from quake_agent.document_loader import load_documents
        connection.send((True, load_documents([path])))
    except Exception:
        connection.send((False, "PDF 解析失败。"))
    finally:
        connection.close()


def load_pdf_bounded(path: Path, timeout: float = 25):
    # A malformed PDF must not occupy the Streamlit session indefinitely.
    context = multiprocessing.get_context("spawn")
    receiving, sending = context.Pipe(duplex=False)
    process = context.Process(target=_parse, args=(str(path), sending), daemon=True)
    try:
        process.start()
        sending.close()
        if not receiving.poll(timeout):
            raise TimeoutError("PDF 解析超时，请拆分或重新导出。")
        success, payload = receiving.recv()
        if not success:
            raise ValueError(payload)
        return payload
    finally:
        receiving.close()
        sending.close()
        if process.pid is not None:
            process.join(timeout=0.2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
            if process.is_alive():
                process.kill()
                process.join(timeout=3)
