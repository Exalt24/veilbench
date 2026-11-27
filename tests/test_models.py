import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from veilbench.detect.ml import chunks
from veilbench.detect.stat import StatDetector, shape, tokenize
from veilbench.corpus.generate import generate_corpus


@settings(max_examples=60, deadline=None)
@given(st.text(alphabet="ab \n.,", min_size=0, max_size=3000), st.integers(80, 900))
def test_chunks_cover_the_text_at_their_offsets_and_respect_the_size(text, size):
    pieces = chunks(text, size)
    for off, piece in pieces:
        assert text[off:off + len(piece)] == piece          # an offset is where the piece really sits
        assert piece.strip()
    covered = {i for off, piece in pieces for i in range(off, off + len(piece))}
    assert all(i in covered for i, ch in enumerate(text) if not ch.isspace())  # nothing that is not blank is left out


def test_token_shape_collapses_long_runs():
    assert shape("Maria") == "Xxx" and shape("0917") == "dd" and shape("A1b2") == "Xdxd" and shape("xxxxxxx") == "xx"


def test_the_statistical_detector_learns_a_cue_it_was_not_given_a_list_for():
    det = StatDetector().fit(generate_corpus(500, 1, "train"), epochs=6)
    text = "Hello, my name is Zorbalina Quintessa. Call 0917 555 0142."
    got = {(s.label, text[s.start:s.end]) for s in det.detect(text)}
    assert any(label == "PERSON" and "Zorbalina" in value for label, value in got), got
    assert any(label == "PHONE" for label, _ in got)
    assert det.detect("") == []
    spans = det.detect(text)
    assert all(0 <= s.start < s.end <= len(text) for s in spans)
    assert [t[0] for t in tokenize("a-b")] == ["a", "-", "b"]


def test_an_unfitted_detector_refuses():
    with pytest.raises(RuntimeError):
        StatDetector().detect("anything")


def test_spacy_detector_maps_its_labels():
    spacy = pytest.importorskip("spacy")
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        pytest.skip("the spaCy English model is not installed")
    from veilbench.detect.ml import SpacyDetector
    text = "Barack Obama visited Cebu City last week."
    spans = SpacyDetector(("en_core_web_sm",)).detect(text)
    assert any(s.label == "PERSON" and "Obama" in text[s.start:s.end] for s in spans)
    assert all(s.source == "spacy" and s.label in ("PERSON", "ADDRESS") for s in spans)


def test_gliner_detector_finds_typed_spans_with_real_offsets():
    pytest.importorskip("gliner")
    from veilbench.detect.ml import GlinerDetector
    try:
        det = GlinerDetector()
    except Exception as e:  # noqa: BLE001 - the weights are a download; offline machines skip
        pytest.skip(f"the GLiNER weights are not available: {type(e).__name__}")
    text = "Maria Santos lives at 12 Mabini Street, Cebu City. Call her on +63 917 555 0142 or email maria.santos@example.com."
    spans = det.detect(text)
    labels = {s.label for s in spans}
    assert {"PHONE", "EMAIL", "ADDRESS"} <= labels, labels
    assert all(text[s.start:s.end].strip() for s in spans)
