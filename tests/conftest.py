import pytest

from veilbench.corpus.generate import generate_corpus


@pytest.fixture(scope="session")
def corpus():
    return generate_corpus(300, 7)


@pytest.fixture(scope="session")
def small_corpus():
    return generate_corpus(120, 3)
