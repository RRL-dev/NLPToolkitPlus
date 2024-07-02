"""Factory loader utils."""

from inspect import Signature, signature

from pydantic import BaseModel


def has_init_arg(cls: type[BaseModel], arg_name: str) -> bool:
    """Check if the `__init__` method of a class has a specific argument.

    Args:
    ----
        cls (Type): The class to inspect.
        arg_name (str): The name of the argument to check for.

    Returns:
    -------
        bool: True if the argument exists, False otherwise.

    """
    init_signature: Signature = signature(cls.__init__)
    return arg_name in init_signature.parameters
