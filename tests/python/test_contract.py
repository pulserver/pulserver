from __future__ import annotations

import pytest
from pulserver import SequencePlugin


class _EmptySequence(SequencePlugin):
    pass


def test_pulseq_sequence_requires_abstract_methods():
    with pytest.raises(TypeError):
        _EmptySequence()
