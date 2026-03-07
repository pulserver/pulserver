"""Tests for lightweight plugin registry/discovery."""

from __future__ import annotations

from pulserver.recon_server.plugin_loader import PluginLoadError, PluginRegistry


def test_register_and_resolve_explicit_class_spec():
    registry = PluginRegistry()
    registry.register_spec('local_test', 'pulserver.recon_server.plugins.simplefft:SimpleFFTPlugin')
    plugin = registry.resolve('local_test')
    assert plugin.__class__.__name__ == 'SimpleFFTPlugin'


def test_resolve_legacy_function_plugin():
    registry = PluginRegistry()
    plugin = registry.resolve('pulserver.recon_server.plugins.simplefft:process')
    assert plugin.name == 'pulserver.recon_server.plugins.simplefft:process'


def test_discover_directory_registers_plugins(tmp_path):
    plugin_file = tmp_path / 'demo_plugin.py'
    plugin_file.write_text(
        '\n'.join(
            [
                'from pulserver.recon_server.plugin_api import ReconPlugin',
                'class DemoPlugin(ReconPlugin):',
                '    def process(self, connection, config, metadata):',
                '        return None',
            ]
        ),
        encoding='utf-8',
    )

    registry = PluginRegistry()
    registry.discover_directory(tmp_path)

    plugin = registry.resolve('DemoPlugin')
    assert plugin.name == 'DemoPlugin'


def test_resolve_raises_for_missing_target():
    registry = PluginRegistry()
    try:
        registry.resolve('pulserver.recon_server.plugins.simplefft:Missing')
    except PluginLoadError:
        pass
    else:
        raise AssertionError('Expected PluginLoadError for missing target')
