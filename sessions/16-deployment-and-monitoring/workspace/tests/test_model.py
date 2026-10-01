import json

import pytest

from sentiment_service.model import (
    FIXTURE_REVIEWS,
    build_pipeline,
    load_model,
    reference_stats,
    review_text,
    save_model,
)


def test_save_and_load_round_trip(model_dir):
    pipe, meta = load_model(model_dir)
    assert meta["model_version"] == "0.0.0-test"
    assert {"scikit_learn", "python", "created_at", "model_sha256", "reference"} <= set(meta)
    assert list(pipe.classes_) == ["neg", "neu", "pos"]
    assert pipe.predict(["Love it! Works perfectly."])[0] == "pos"


def test_a_changed_model_file_is_refused(tmp_path):
    texts, labels = zip(*FIXTURE_REVIEWS, strict=True)
    path = save_model(build_pipeline(min_df=1).fit(texts, labels), tmp_path, {"model_version": "x"})
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_model(tmp_path)


def test_skops_format(tmp_path):
    pytest.importorskip("skops")
    texts, labels = zip(*FIXTURE_REVIEWS, strict=True)
    save_model(build_pipeline(min_df=1).fit(texts, labels), tmp_path, {"model_version": "s"}, fmt="skops")
    pipe, meta = load_model(tmp_path)
    assert meta["model_file"] == "model.skops" and pipe.predict(["Waste of money"])[0] == "neg"


def test_reference_stats():
    ref = reference_stats(["a" * 9, "b" * 99, "c"], ["neg", "pos", "pos"])
    assert ref["label_shares"] == {"neg": 0.3333, "neu": 0.0, "pos": 0.6667}
    assert len(ref["log_length_quantiles"]) == 11 and ref["n"] == 3
    json.dumps(ref)  # must be serialisable


def test_review_text_is_the_same_for_training_and_serving():
    assert review_text("Great", "Works.") == "Great. Works."
    assert review_text("", "Works.") == "Works."
    assert review_text(None, None) == ""
