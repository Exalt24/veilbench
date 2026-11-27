"""Pretrained model detectors: spaCy NER and a GLiNER PII model. Both are optional installs (`pip install -e .[models]`)."""
from __future__ import annotations


from veilbench.schema import Span

SPACY_MAP = {"PERSON": "PERSON", "PER": "PERSON", "GPE": "ADDRESS", "LOC": "ADDRESS", "FAC": "ADDRESS"}

GLINER_LABELS = {
    "person name": "PERSON",
    "email address": "EMAIL",
    "phone number": "PHONE",
    "street address": "ADDRESS",
    "credit card number": "CREDIT_CARD",
    "national id or tax id number": "NATIONAL_ID",
    "iban bank account number": "IBAN",
    "api key or access token": "API_KEY",
    "password": "PASSWORD",
    "date of birth": "DATE_OF_BIRTH",
    "ip address": "IP_ADDRESS",
}


class SpacyDetector:
    """The small English pipeline plus the small multilingual one, both run on every text and their spans pooled."""

    name = "spacy"

    def __init__(self, models: tuple[str, ...] = ("en_core_web_sm", "xx_ent_wiki_sm")) -> None:
        import spacy
        self.nlps = [spacy.load(m, disable=["parser", "lemmatizer", "tagger", "attribute_ruler"]) for m in models]
        for nlp in self.nlps:
            nlp.max_length = 2_000_000

    def detect(self, text: str) -> list[Span]:
        out: list[Span] = []
        for nlp in self.nlps:
            for ent in nlp(text).ents:
                label = SPACY_MAP.get(ent.label_)
                if label:
                    out.append(Span(ent.start_char, ent.end_char, label, "spacy"))
        return out


def chunks(text: str, size: int = 700) -> list[tuple[int, str]]:
    """Split at line breaks (then at spaces) into pieces of at most `size` characters, keeping each piece's offset."""
    pieces: list[tuple[int, str]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        while len(line) > size:
            cut = line.rfind(" ", 0, size)
            cut = cut + 1 if cut > size // 2 else size
            pieces.append((pos, line[:cut]))
            pos += cut
            line = line[cut:]
        if pieces and pieces[-1][0] + len(pieces[-1][1]) == pos and len(pieces[-1][1]) + len(line) <= size:
            off, prev = pieces[-1]
            pieces[-1] = (off, prev + line)
        else:
            pieces.append((pos, line))
        pos += len(line)
    return [(o, t) for o, t in pieces if t.strip()]


class GlinerDetector:
    name = "gliner"

    def __init__(self, model: str = "knowledgator/gliner-pii-small-v1.0", threshold: float = 0.5) -> None:
        from gliner import GLiNER
        self.model = GLiNER.from_pretrained(model)
        self.threshold = threshold
        self.labels = list(GLINER_LABELS)

    def detect(self, text: str) -> list[Span]:
        out: list[Span] = []
        for off, piece in chunks(text):
            for e in self.model.predict_entities(piece, self.labels, threshold=self.threshold):
                label = GLINER_LABELS.get(e["label"])
                if label:
                    a, b = off + e["start"], off + e["end"]
                    while b > a and not text[b - 1].isalnum():
                        b -= 1
                    out.append(Span(a, b, label, "gliner", ""))
        return out


