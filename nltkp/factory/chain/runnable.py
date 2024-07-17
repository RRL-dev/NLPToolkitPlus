"""Module for chaining callables or other ChainedRunnables.

This module defines the ChainedRunnable class, which allows for chaining
together callables and other ChainedRunnable instances to create
composable units of work. This chaining enables the creation of complex
pipelines by combining simpler functions.

Classes:
    ChainedRunnable: A generic class for chaining callables or other ChainedRunnables.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

Input = TypeVar("Input")
Output = TypeVar("Output")
Other = TypeVar("Other")


class ChainedRunnable(Generic[Input, Output]):
    """A class for chaining callables or other ChainedRunnables."""

    def __init__(self: ChainedRunnable, func: Callable[[Input], Output]) -> None:
        """Initialize with a callable function.

        Args:
        ----
            func: A callable that takes an input of type Input and returns an output of type Output.

        """
        self.func: Callable[[Input], Output] = func

    def invoke(self: ChainedRunnable, input_value: Input) -> Output:
        """Invoke the stored function with the given input.

        Args:
        ----
            input_value: The input value to pass to the function.

        Returns:
        -------
            The result of the function.

        """
        return self.func(input_value)

    def __or__(
        self: ChainedRunnable,
        other: ChainedRunnable[Output, Other] | Callable[[Output], Other],
    ) -> ChainedRunnable[Input, Other]:
        """Chain this runnable with another runnable or callable.

        Args:
        ----
            other: Another ChainedRunnable or callable to chain with.

        Returns:
        -------
            A new ChainedRunnable representing the chained operation.

        Raises:
        ------
            TypeError: If the type of `other` is unsupported.

        """
        if isinstance(other, ChainedRunnable):

            def chained_func(input_value: Input) -> Other:
                intermediate_result: Any = self.invoke(input_value=input_value)
                return other.invoke(intermediate_result)

            return ChainedRunnable(func=chained_func)

        if callable(other):

            def chained_callable(input_value: Input) -> Other:
                intermediate_result: Any = self.invoke(input_value)
                return other(intermediate_result)

            return ChainedRunnable(func=chained_callable)

        msg = "Unsupported type for chaining"
        raise TypeError(msg)
