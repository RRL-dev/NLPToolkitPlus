"""Base module for managing and configuring different modules with their respective configurations.

This module includes:
- Configuration models for embeddings, generation, and retrieval modules.
- Module classes for embeddings, generation, and retrieval functionalities.
- A wrapper class to manage and initialize these modules based on configuration files.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic.main import BaseModel

from nltkp.utils import load_yaml

from .utils import has_init_arg

if TYPE_CHECKING:
    from pydantic import BaseModel


class ModuleLoader:
    """Dynamically creates module instances based on configuration files and handles dependencies.

    Manages configurations and instances of required dependencies.
    """

    def __init__(self: ModuleLoader, config_files: dict[str, Any]) -> None:
        """Initialize the ModuleLoader with a dictionary of configuration files.

        Each key in the dictionary is a module name, and the value is a tuple containing:
        - The path to the configuration YAML file
        - The Pydantic BaseModel class for configuration
        - The actual module class
        - A dictionary of dependencies where keys are argument names and values are the class types or other module
        """
        self.config_files: dict[str, Any] = config_files
        self.modules: dict[str, BaseModel] = {}  # Cache for loaded modules
        self.keep_module: str | None = None  # Module to keep during deletion

    def get_module(self: ModuleLoader, module_name: str) -> BaseModel:
        """Recursively load a module and its dependencies."""
        if module_name in self.modules:
            return self.modules[module_name]

        if module_name not in self.config_files:
            msg: str = f"Configuration for module '{module_name}' is not defined."
            raise KeyError(msg)

        file_path: str
        module_class: type[Any]
        dependencies: dict[str, Any]

        file_path, config_class, module_class, dependencies = self.config_files[module_name]
        config_instance: BaseModel | None = self._load_config(
            file_path=file_path,
            config_class=config_class,
            module_name=module_name,
        )
        resolved_dependencies: dict[str, BaseModel] = self._resolve_dependencies(
            dependencies=dependencies,
            config_instance=config_instance,
        )

        module: BaseModel = (
            module_class(**resolved_dependencies)
            if config_instance is None
            else module_class(config_instance, **resolved_dependencies)
        )
        self.modules[module_name] = module
        return module

    @staticmethod
    def _load_config(
        file_path: str,
        config_class: type[BaseModel],
        module_name: str,
    ) -> BaseModel | None:
        """Load and return the configuration instance."""
        config_path = Path(file_path)
        if not config_path.exists():
            msg: str = f"Configuration file {file_path} not found for module '{module_name}'."
            raise FileNotFoundError(msg)

        config_data: dict[str, Any] = load_yaml(file_path=file_path)
        configuration: dict[str, Any] = config_data.get(module_name, {})
        return config_class(**configuration) if configuration else None

    def _resolve_dependencies(
        self: ModuleLoader,
        dependencies: dict[str, Any],
        config_instance: BaseModel | None,
    ) -> dict[str, BaseModel]:
        """Resolve and return the dependencies for a module."""
        resolved_dependencies: dict[str, BaseModel] = {}
        for module_name, module in dependencies.items():
            if isinstance(module, str):
                if module in {"True", True}:
                    resolved_dependencies[module_name] = self.get_module(module_name=module_name)
                else:
                    resolved_dependencies[module_name] = self.get_module(module_name=module)
            else:
                resolved_dependencies[module_name] = self._instantiate_dependency(
                    module=module,
                    config_instance=config_instance,
                )
        return resolved_dependencies

    @staticmethod
    def _instantiate_dependency(
        module: type[BaseModel],
        config_instance: BaseModel | None,
    ) -> BaseModel:
        """Instantiate a dependency, passing configuration if required."""
        if has_init_arg(cls=module, arg_name="config") and config_instance:
            return module(config=config_instance)
        return module()

    def __getattr__(self: ModuleLoader, name: str) -> BaseModel:
        """Override attribute access to dynamically create module instances."""
        try:
            return self.get_module(module_name=name)
        except (FileNotFoundError, KeyError) as e:
            msg: str = f"Failed to create or access the module '{name}': {e!s}"
            raise AttributeError(msg) from e

    def __del__(self: ModuleLoader) -> None:
        """Clear all cached modules except the one to keep."""
        self.modules.clear()
