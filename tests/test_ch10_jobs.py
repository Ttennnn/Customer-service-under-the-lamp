from app.core import jobs


class FakeProc:
    def __init__(self):
        self.pid = 1234
        self.returncode = None

    def poll(self):
        return self.returncode

    def finish(self, code):
        self.returncode = code


def test_unknown_job_rejected_and_reentry_rejected(monkeypatch):
    assert jobs.start_job("not-a-job") is None
    fake = FakeProc()
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: fake)
    assert jobs.start_job("ch10-eval") is not None
    assert jobs.start_job("ch10-eval") is None
    fake.finish(0)
    assert jobs.job_status("ch10-eval")["exit_code"] == 0


def test_stop_windows_kills_tree(monkeypatch):
    called = []
    fake = FakeProc()
    monkeypatch.setattr(jobs, "_platform_is_windows", lambda: True)
    monkeypatch.setattr(jobs.subprocess, "Popen", lambda *a, **k: fake)
    monkeypatch.setattr(
        jobs.subprocess, "run", lambda argv, **k: called.append(argv)
    )
    jobs.start_job("classify-pool")
    jobs.stop_job("classify-pool")
    assert called and "taskkill" in called[0][0]


def test_job_env_forces_utf8(monkeypatch):
    seen = {}
    fake = FakeProc()

    def popen(argv, **kwargs):
        seen.update(kwargs.get("env") or {})
        return fake

    monkeypatch.setattr(jobs.subprocess, "Popen", popen)
    jobs.start_job("ch10-eval")
    assert seen.get("PYTHONUTF8") == "1"
    assert seen.get("PYTHONIOENCODING") == "utf-8"
