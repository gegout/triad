"""End-to-end harness test. All model responses are fixtures, not real intelligence."""
import json
import pathlib
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

binary = pathlib.Path(sys.argv[1]).resolve()
lock = threading.Lock()
state = {"active": 0, "peak": 0, "peer_seen": False, "invalid": False, "ballot": False, "inputs": [], "chat_bodies": []}


class Fixture(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        if self.path != "/v1/models":
            self.send_error(404)
            return
        body = json.dumps({"data": [{"id": m} for m in ["smollm2", "minilm", "smollm3-3b", "nomic-embed-text-v1-5"]]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/embeddings":
            state["inputs"].append(request["input"])
            result = {"data": [{"embedding": [1.0, 0.5, 0.25]}]}
        elif self.path == "/v1/chat/completions":
            state["chat_bodies"].append(request)
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.05)
            messages = request["messages"]
            last = messages[-1]["content"]
            ballot = "CONSENSUS VOTE" in last
            with lock:
                state["peer_seen"] |= any("PEER CONTRIBUTIONS" in m["content"] for m in messages)
                state["ballot"] |= ballot
                state["active"] -= 1
            ids = list(dict.fromkeys(map(int, re.findall(r"ID=(\d+)", last))))
            agree = ids if ballot else []
            # The third agent abstains. Exactly two distinct votes must suffice.
            if messages[0]["content"].startswith("You are practical"):
                agree = []
            content = {
                "statement": "Compare answers with and without retrieved examples on unseen tasks.",
                "evidence_ids": [],
                "agree_memory_ids": agree,
                "uncertainty": "No experiment has been executed.",
                "question": "Should the evaluation focus on factual accuracy?",
            }
            result = {"choices": [{"finish_reason": "stop", "message": {
                "content": "invalid JSON" if state["invalid"] else json.dumps(content)
            }}]}
        else:
            self.send_error(404)
            return
        body = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


server = ThreadingHTTPServer(("127.0.0.1", 0), Fixture)
threading.Thread(target=server.serve_forever, daemon=True).start()
try:
    with tempfile.TemporaryDirectory(prefix="triad-smoke-") as tmp:
        data = pathlib.Path(tmp)

        def run(*args, ok=True):
            p = subprocess.run([str(binary), "--data", str(data), *args],
                               text=True, capture_output=True, timeout=30)
            assert (p.returncode == 0) == ok, (args, p.stdout, p.stderr)
            return p.stdout + p.stderr

        run("init")
        config = json.loads((data / "config.json").read_text())
        config["chat_url"] = config["embedding_url"] = f"http://127.0.0.1:{server.server_port}/v1"
        (data / "config.json").write_text(json.dumps(config))
        run("init")
        assert json.loads((data / "config.json").read_text()) == config
        output = run("discuss", "Test shared learning")
        assert "6 valid contributions" in output
        assert "now trusted" in output
        assert state["peak"] == 3 and state["peer_seen"] and state["ballot"]
        db = sqlite3.connect(data / "memory.sqlite3")
        assert db.execute("SELECT COUNT(*) FROM memories WHERE status='accepted'").fetchone()[0] == 3
        counts = db.execute("SELECT COUNT(DISTINCT agent) FROM endorsements GROUP BY memory_id").fetchall()
        assert counts == [(2,), (2,), (2,)], counts
        assert "#1:" in run("questions")
        run("answer", "1", "Yes, prioritize accuracy.")
        assert "#1:" not in run("questions")
        assert "User clarification" in run("search", "accuracy")
        run("feedback", "1", "4", "Useful proposed test.")
        assert "4/5" in run("status")
        run("review", "1", "reject")
        assert "[rejected]" in run("memories")
        transcript = json.loads(db.execute("SELECT transcript FROM sessions WHERE id=1").fetchone()[0])
        assert len(transcript) == 9
        assert "Test shared learning" in run("session", "1")
        state["invalid"] = True
        run("discuss", "Invalid outputs should persist", ok=False)
        assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2
        assert "invalid contribution JSON" in run("session", "2")
        run("feedback", "9999", "4", "Unknown session", ok=False)
        run("ingest", "https://127.0.0.1/", ok=False)
        run("ingest", "http://example.com/", ok=False)
        state["invalid"] = False
        run("configure-snaps")
        snap_config = json.loads((data / "config.json").read_text())
        assert snap_config["chat_url"] == "http://localhost:8366/v1"
        assert snap_config["embedding_url"] == "http://127.0.0.1:8348/v1"
        assert snap_config["embedding_document_prefix"] == "search_document: "
        assert (data / "config.before-snaps.json").exists()
        snap_config["chat_url"] = snap_config["embedding_url"] = config["chat_url"]
        (data / "config.json").write_text(json.dumps(snap_config))
        run("doctor")
        body = state["chat_bodies"][-1]
        assert "response_format" not in body
        assert "/no_think" in body["messages"][0]["content"]
        run("search", "accuracy", ok=False)  # Cannot mix old and Nomic embedding spaces.
        before = db.execute("SELECT id,status FROM memories ORDER BY id").fetchall()
        run("reindex")
        after = db.execute("SELECT id,status FROM memories ORDER BY id").fetchall()
        assert before == after
        run("search", "accuracy")
        assert state["inputs"][-1].startswith("search_query: ")
        assert any(x.startswith("search_document: ") for x in state["inputs"])
        print("PASS: concurrent agents, peer exchange, two-of-three automatic trust, vote provenance,")
        print("semantic reuse, questions, answers, feedback, persistence, rejection, and error paths.")
        print("PASS: snap preset, doctor/model IDs, Nomic prefixes, prompt-only JSON, and reindex migration.")
finally:
    server.shutdown()
