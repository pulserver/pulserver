"""Lightweight plugin registry and discovery for recon plugins."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import inspect
import logging
from pathlib import Path
from types import ModuleType
from typing import Any

from .plugin_api import ReconPlugin


class PluginLoadError(RuntimeError):
    """Raised when plugin resolution or construction fails."""


@dataclass
class _LegacyFunctionPlugin(ReconPlugin):
    """Adapter for legacy `process(connection, config, metadata)` functions."""

    _plugin_name: str
    _fn: Any

    @property
    def name(self) -> str:
        return self._plugin_name

    def process(self, connection: Any, config: Any, metadata: Any) -> None:
        self._fn(connection, config, metadata)


class PluginRegistry:
    """Registry with explicit loading plus optional directory discovery."""

    def __init__(self) -> None:
        self._registry: dict[str, Any] = {}

    def register_spec(self, name: str, spec: str) -> None:
        self._registry[name] = spec

    def register_target(self, name: str, target: Any) -> None:
        self._registry[name] = target

    def discover_directory(self, directory: str | Path) -> None:
        root = Path(directory)
        if not root.exists():
            logging.info('Plugin directory does not exist: %s', root)
            return

        for py_file in sorted(root.glob('*.py')):
            if py_file.name.startswith('_'):
                continue
            module_name = py_file.stem
            discovered = self._load_module_from_file(module_name, py_file)
            self._register_module_plugins(discovered, default_module=module_name)

    def resolve(self, plugin_spec: str) -> ReconPlugin:
        spec = self._registry.get(plugin_spec, plugin_spec)
        if not isinstance(spec, str):
            return self._coerce_plugin(spec, plugin_spec)

        module_name, target_name = self._split_spec(spec)
        module = importlib.import_module(module_name)

        if target_name:
            target = getattr(module, target_name, None)
            if target is None:
                raise PluginLoadError(f'Plugin target not found: {spec}')
            return self._coerce_plugin(target, plugin_spec)

        target = self._find_default_target(module)
        return self._coerce_plugin(target, plugin_spec)

    @staticmethod
    def _split_spec(spec: str) -> tuple[str, str | None]:
        if ':' in spec:
            module_name, target_name = spec.split(':', 1)
            return module_name, target_name
        return spec, None

    def _register_module_plugins(self, module: ModuleType, default_module: str) -> None:
        found = False
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, ReconPlugin) and obj is not ReconPlugin:
                self.register_target(name, obj)
                found = True
            elif inspect.isfunction(obj) and name == 'process':
                self.register_target(default_module, obj)
                found = True

        if not found:
            logging.debug('No plugin entry points discovered in %s', module.__name__)

    @staticmethod
    def _find_default_target(module: ModuleType) -> Any:
        for _, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, ReconPlugin) and obj is not ReconPlugin:
                return obj
        fn = getattr(module, 'process', None)
        if callable(fn):
            return fn
        raise PluginLoadError(f'No ReconPlugin class or process() function found in {module.__name__}')

    @staticmethod
    def _coerce_plugin(target: Any, plugin_name: str) -> ReconPlugin:
        if inspect.isclass(target) and issubclass(target, ReconPlugin):
            return target()
        if callable(target):
            return _LegacyFunctionPlugin(plugin_name, target)
        raise PluginLoadError(f'Unsupported plugin target for {plugin_name}: {target!r}')

    @staticmethod
    def _load_module_from_file(module_name: str, file_path: Path) -> ModuleType:
        spec = importlib.util.spec_from_file_location(f'pulserver_user_plugin_{module_name}', file_path)
        if spec is None or spec.loader is None:
            raise PluginLoadError(f'Cannot create module spec for {file_path}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
