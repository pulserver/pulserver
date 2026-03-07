"""Tests for recon server config parsing and plugin resolution helpers."""

from pulserver.recon_server.config_loader import (
    load_config_file,
    load_config_text,
    resolve_plugin_spec,
)


def test_load_config_text_toml():
    cfg = load_config_text("cmd = 'simplefft'\n[parameters]\nplugin = 'pkg:Plugin'")
    assert cfg['cmd'] == 'simplefft'
    assert cfg['parameters']['plugin'] == 'pkg:Plugin'


def test_load_config_text_json():
    cfg = load_config_text('{"parameters": {"config": "plugin_x"}}')
    assert cfg['parameters']['config'] == 'plugin_x'


def test_load_config_text_plain_alias():
    cfg = load_config_text('simplefft')
    assert cfg['plugin'] == 'simplefft'


def test_resolve_plugin_spec_priority():
    cfg = {'plugin': 'a:b', 'parameters': {'config': 'x'}}
    assert resolve_plugin_spec(cfg, fallback='fallback') == 'a:b'


def test_resolve_plugin_spec_from_parameters():
    cfg = {'parameters': {'customconfig': 'pkg:CustomPlugin'}}
    assert resolve_plugin_spec(cfg, fallback='fallback') == 'pkg:CustomPlugin'


def test_load_config_file_json(tmp_path):
    p = tmp_path / 'test.json'
    p.write_text('{"plugin": "simplefft"}', encoding='utf-8')
    cfg = load_config_file(p)
    assert cfg['plugin'] == 'simplefft'


def test_load_config_file_toml(tmp_path):
    p = tmp_path / 'test.toml'
    p.write_text("plugin = 'simplefft'", encoding='utf-8')
    cfg = load_config_file(p)
    assert cfg['plugin'] == 'simplefft'
