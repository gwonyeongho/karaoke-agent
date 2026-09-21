import importlib
import sys
from types import SimpleNamespace


def test_main_import_does_not_load_whisper():
    sys.modules.pop('backend.main', None)
    main = importlib.import_module('backend.main')
    assert 'faster_whisper' not in sys.modules
    assert callable(main.get_whisper_model)


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
