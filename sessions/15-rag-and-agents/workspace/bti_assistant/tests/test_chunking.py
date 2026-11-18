import pytest

from bti_assistant.chunking import chunk_documents, chunk_words


def test_windows_have_the_right_length_and_overlap():
    text = " ".join(f"w{i}" for i in range(250))
    chunks = chunk_words(text, size=120, overlap=20)
    assert [len(c.split()) for c in chunks] == [120, 120, 50]
    assert chunks[0].split()[-20:] == chunks[1].split()[:20]  # the overlap appears in both


def test_short_and_empty_texts():
    assert chunk_words("only five words in here", size=120) == ["only five words in here"]
    assert chunk_words("   ") == []


@pytest.mark.parametrize("size, overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_settings_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_words("a b c", size=size, overlap=overlap)


def test_chunks_keep_their_document_id():
    chunks = chunk_documents({"r1": "a b c d e f", "r2": "g h"}, size=4, overlap=1)
    assert [c.chunk_id for c in chunks] == ["r1#0", "r1#1", "r2#0"]
    assert {c.doc_id for c in chunks} == {"r1", "r2"}
