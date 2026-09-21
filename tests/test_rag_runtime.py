import importlib
import sys
from types import SimpleNamespace


def test_main_import_does_not_load_whisper():
    sys.modules.pop('backend.main', None)
    main = importlib.import_module('backend.main')
    assert 'faster_whisper' not in sys.modules
    assert callable(main.get_whisper_model)
    paths = {route.path for route in main.app.routes}
    assert "/api/v1/commands" in paths
    assert "/api/v1/stt" in paths
    assert "/api/v1/voice_command" in paths


def test_runtime_reuses_index_and_rebuilds_for_changed_corpus(tmp_path):
    from backend.rag.runtime import build_service
    knowledge = tmp_path / 'knowledge'
    knowledge.mkdir()
    doc = knowledge / 'guide.md'
    doc.write_text('# Guide\n예약 버튼', encoding='utf-8')
    calls = []
    class Store:
        def __init__(self, **kwargs):
            self.name = kwargs['collection_name']
        def get(self):
            return {'ids': []}
        def add_documents(self, documents, ids):
            calls.append((self.name, ids))
    kwargs = dict(knowledge_dir=knowledge, persist_dir=tmp_path / 'index', store_factory=Store,
                  embedding_factory=lambda **kw: None, chat_factory=lambda **kw: None)
    first = build_service(**kwargs)
    second = build_service(**kwargs)
    assert first.vector_store.name == second.vector_store.name
    doc.write_text('# Guide\n예약 취소 버튼', encoding='utf-8')
    third = build_service(**kwargs)
    assert third.vector_store.name != first.vector_store.name
    assert len(calls[0][1]) == 1


def test_cached_runtime_refreshes_when_knowledge_changes(monkeypatch, tmp_path):
    from backend.rag import runtime

    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    document = knowledge / "guide.md"
    document.write_text("# Guide\n예약 버튼", encoding="utf-8")
    built_versions = []

    monkeypatch.setattr(runtime, "KNOWLEDGE_DIR", knowledge)
    monkeypatch.setattr(
        runtime,
        "build_service",
        lambda: built_versions.append(runtime.knowledge_version(knowledge)) or object(),
    )
    runtime.clear_rag_service_cache()

    first = runtime.get_rag_service()
    same = runtime.get_rag_service()
    document.write_text("# Guide\n예약 취소 버튼", encoding="utf-8")
    changed = runtime.get_rag_service()

    assert first is same
    assert changed is not first
    assert len(built_versions) == 2
