import importlib
import sys


def test_dify_app_factory_capture_registers_create_app_result(monkeypatch, tmp_path):
    app_factory = tmp_path / "app_factory.py"
    app_factory.write_text(
        "\n".join(
            [
                "class FakeApp:",
                "    def app_context(self):",
                "        return self",
                "",
                "def create_app():",
                "    return object(), FakeApp()",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "app_factory", raising=False)

    import agentguard.adapters.agent.dify_bootstrap as bootstrap
    import agentguard.adapters.agent.dify_flask as dify_flask

    bootstrap = importlib.reload(bootstrap)
    dify_flask = importlib.reload(dify_flask)

    status = bootstrap.install_dify_app_factory_capture()

    assert status == {"installed": True, "patched": False, "reason": "import_hook_installed"}

    module = importlib.import_module("app_factory")
    result = module.create_app()

    assert isinstance(result, tuple)
    assert dify_flask.get_dify_flask_app() is result[1]


def test_dify_app_factory_capture_notifies_app_ready_callbacks(monkeypatch, tmp_path):
    app_factory = tmp_path / "app_factory.py"
    app_factory.write_text(
        "\n".join(
            [
                "class FakeApp:",
                "    def app_context(self):",
                "        return self",
                "",
                "def create_app():",
                "    return object(), FakeApp()",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "app_factory", raising=False)

    import agentguard.adapters.agent.dify_bootstrap as bootstrap
    import agentguard.adapters.agent.dify_flask as dify_flask

    bootstrap = importlib.reload(bootstrap)
    dify_flask = importlib.reload(dify_flask)
    seen = []

    dify_flask.on_dify_flask_app_ready(lambda app: seen.append(app))
    bootstrap.install_dify_app_factory_capture()

    module = importlib.import_module("app_factory")
    result = module.create_app()

    assert seen == [result[1]]
