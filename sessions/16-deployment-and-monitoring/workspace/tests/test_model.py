import json

import pytest

from tariff_service.model import (
    FIXTURE_DECISIONS,
    build_pipeline,
    decision_text,
    load_model,
    prune_coefficients,
    reference_stats,
    save_model,
    top_k,
)


def test_save_and_load_round_trip(model_dir):
    pipe, meta = load_model(model_dir)
    assert meta["model_version"] == "0.0.0-test"
    assert {"scikit_learn", "python", "created_at", "model_sha256", "reference", "headings"} <= set(meta)
    assert list(pipe.classes_) == ["6307", "6403", "6404", "9503"]
    assert pipe.predict(["Spielzeugauto aus Kunststoff"])[0] == "9503"


def test_top_k_is_sorted_and_has_k_entries(model_dir):
    pipe, _ = load_model(model_dir)
    ranked = top_k(pipe, ["Sportschuhe aus Textil", "Puppe"], k=3)
    assert [len(r) for r in ranked] == [3, 3]
    assert all(r[0][1] >= r[1][1] >= r[2][1] for r in ranked)


def test_a_changed_model_file_is_refused(tmp_path):
    texts, labels = zip(*FIXTURE_DECISIONS, strict=True)
    path = save_model(build_pipeline(min_df=1).fit(texts, labels), tmp_path, {"model_version": "x"})
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_model(tmp_path)


def test_skops_format(tmp_path):
    pytest.importorskip("skops")
    texts, labels = zip(*FIXTURE_DECISIONS, strict=True)
    save_model(build_pipeline(min_df=1, alpha=1e-4).fit(texts, labels), tmp_path, {"model_version": "s"}, fmt="skops")
    pipe, meta = load_model(tmp_path)
    assert meta["model_file"] == "model.skops" and pipe.predict(["Schutzmaske aus Vliesstoff"])[0] == "6307"


def test_reference_stats():
    ref = reference_stats(["a" * 9, "b" * 99, "c"], ["6403", "6404", "9503"], ["de", "de", "fr"], ["DE", "DE", "FR"])
    assert ref["chapter_shares"] == {"64": 0.6667, "95": 0.3333}
    assert ref["language_shares"] == {"de": 0.6667, "fr": 0.3333}
    assert len(ref["log_length_quantiles"]) == 11 and ref["n"] == 3
    json.dumps(ref)  # must be serialisable


def test_decision_text_is_the_same_for_training_and_serving():
    assert decision_text("  Schuhe\r\n aus   Leder ") == "Schuhe aus Leder"
    assert decision_text(None) == ""


def test_pruning_keeps_predictions_and_makes_the_weights_sparse(tmp_path):
    texts, labels = zip(*FIXTURE_DECISIONS, strict=True)
    pipe = build_pipeline(min_df=1, alpha=1e-4).fit(texts, labels)
    before = list(pipe.predict(texts))
    kept = prune_coefficients(pipe, threshold=0.01)
    assert 0 < kept < 1 and hasattr(pipe.named_steps["clf"].coef_, "nnz")
    assert list(pipe.predict(texts)) == before
    save_model(pipe, tmp_path, {"model_version": "p"})
    assert load_model(tmp_path)[0].predict(["Spielzeugauto"])[0] == "9503"
