from __future__ import annotations

import pytest
from pulserver.core import UIParam, dict_to_protocol, protocol_to_dict


def test_protocol_roundtrip(sample_protocol):
    encoded = protocol_to_dict(sample_protocol)
    decoded = dict_to_protocol(encoded)

    assert set(decoded.keys()) == {UIParam.TE.value, UIParam.TR.value}
    assert decoded[UIParam.TE.value].value == pytest.approx(sample_protocol[UIParam.TE].value)
    assert decoded[UIParam.TR.value].value == pytest.approx(sample_protocol[UIParam.TR].value)


def test_roundtrip_preserves_units(sample_protocol):
    decoded = dict_to_protocol(protocol_to_dict(sample_protocol))
    assert decoded[UIParam.TE.value].unit == "ms"
    assert decoded[UIParam.TR.value].unit == "ms"
