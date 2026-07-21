from __future__ import annotations

import pytest
from pulserver import PulseqSequence


class _EmptySequence(PulseqSequence):
    pass


def test_pulseq_sequence_requires_abstract_methods():
    with pytest.raises(TypeError):
        _EmptySequence()
