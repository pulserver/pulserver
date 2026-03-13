from __future__ import annotations


def test_bridge_test_plugin_contract(bridge_test_plugin_module):
    assert hasattr(bridge_test_plugin_module, "get_default_protocol")
    assert hasattr(bridge_test_plugin_module, "validate_protocol")
    assert hasattr(bridge_test_plugin_module, "make_sequence")


def test_bridge_test_plugin_validation_semantics(bridge_test_plugin_module):
    protocol = bridge_test_plugin_module.get_default_protocol(opts=None)
    result = bridge_test_plugin_module.validate_protocol(opts=None, protocol=protocol)
    assert result["valid"] is True
    assert result["duration"] is not None
    assert "TA" in result["info"]
