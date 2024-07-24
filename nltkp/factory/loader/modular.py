"""Module for loading all modules and single module based on configuration files.

This module includes:
- AllModulesLoader: A class that inherits from ModuleLoader and loads all modules based on configuration files.
- SingleModuleLoader: A class that inherits from ModuleLoader and loads a specific module based on configuration files.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic.main import BaseModel

from .base import ModuleLoader

if TYPE_CHECKING:
    from pydantic import BaseModel


class AllModulesLoader(ModuleLoader):
    """Loads all modules based on the configuration files."""

    def __init__(
        self: AllModulesLoader,
        config_files: dict[str, Any],
    ) -> None:
        """Initialize the AllModulesLoader and load all modules."""
        super().__init__(config_files=config_files)
        self._load_all_modules()

    def _load_all_modules(self: AllModulesLoader) -> None:
        """Load all modules based on the configuration files."""
        for module_name in self.config_files:
            self.get_module(module_name=module_name)


class SingleModuleLoader(ModuleLoader):
    """Loads a single module based on the configuration files."""

    def __init__(
        self: SingleModuleLoader,
        config_files: dict[str, Any],
    ) -> None:
        """Initialize the SingleModuleLoader without loading all modules."""
        super().__init__(config_files=config_files)

    def load_module(self: SingleModuleLoader, module_name: str) -> BaseModel:
        """Load a specific module by name."""
        module: BaseModel = self.get_module(module_name=module_name)
        return module
