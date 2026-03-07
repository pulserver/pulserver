"""Minimal config parsing helpers for recon server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]


def load_config_text(text: str) -> dict[str, Any]:
    """Parse config text as TOML first, then JSON, then plain plugin alias."""
    data = text.strip()
    if not data:
        return {}

    # Try TOML first to support existing entry-file patterns.
    try:
        loaded = tomllib.loads(data)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass

    try:
        loaded = json.loads(data)
        if isinstance(loaded, dict):
            return loaded
    except Exception:
        pass

    return {'plugin': data}


def load_config_file(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path)
    if path.suffix.lower() == '.json':
        return json.loads(path.read_text(encoding='utf-8'))

    if path.suffix.lower() in {'.toml', '.entry'}:
        return tomllib.loads(path.read_text(encoding='utf-8'))

    return load_config_text(path.read_text(encoding='utf-8'))


def resolve_plugin_spec(config: dict[str, Any], fallback: str = 'simplefft') -> str:
    """Resolve plugin spec from parsed config using a small compatibility matrix."""
    if 'plugin' in config and config['plugin']:
        return str(config['plugin'])

    params = config.get('parameters', {})
    if isinstance(params, dict):
        for key in ('plugin', 'customconfig', 'config'):
            value = params.get(key)
            if value:
                return str(value)

    cmd = config.get('cmd')
    if cmd:
        return str(cmd)

    return fallback
