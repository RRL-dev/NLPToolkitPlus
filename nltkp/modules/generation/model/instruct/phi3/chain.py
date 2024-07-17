"""The module defines the Phi3InstructChain class, which extends BasePhi3Instruct and ChainedRunnable."""

from typing import Any

from nltkp.factory import ChainedRunnable

from .base import BasePhi3Instruct


class Phi3InstructChain(BasePhi3Instruct, ChainedRunnable[list[dict[str, str]], str]):
    """A chainable class for generating text responses using a pre-trained Phi3 instruct model.

    This class integrates the functionalities of BasePhi3Instruct and ChainedRunnable to
    provide a component that can be used in a chained processing pipeline. It handles
    the preparation of inputs, generation of responses, and post-processing of outputs.
    """

    def __init__(self) -> None:
        """Initialize Phi3InstructChain with configuration and setup for chaining.

        Args:
        ----
            config (Phi3InstructConfig): Configuration settings for the Phi3 instruct model.

        """
        super().__init__()
        ChainedRunnable.__init__(self=self, func=self.forward)

    def extract_generated_text(self, response: list[dict[str, Any]]) -> str:
        """Extract the generated text from the model response.

        Args:
        ----
            response (list[dict[str, Any]]): The response from the model.

        Returns:
        -------
            str: The extracted generated text.

        """
        for result in response:
            if "generated_text" in result:
                return result["generated_text"]
        return ""
