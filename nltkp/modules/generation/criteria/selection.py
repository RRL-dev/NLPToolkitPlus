"""Module provides a class for performing top-k logits warping in text generation processes."""

from torch import Tensor, topk


class TopKLogits:
    """Class performs top-k, i.e., restricting to the k highest probability elements.

    Often used together with temperature and top-p (nucleus) sampling methods.

    Args:
    ----
        top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (float, optional, defaults to -inf): All filtered values will be set to this float value.
        min_tokens_to_keep (int, optional, defaults to 1): Minimum number of tokens that cannot be filtered.

    """

    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1) -> None:
        """Class performs top-k, i.e., restricting to the k highest probability elements.

        Often used together with temperature and top-p (nucleus) sampling methods.

        Args:
        ----
            top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
            filter_value (float, optional, defaults to -inf): All filtered values will be set to this float value.
            min_tokens_to_keep (int, optional, defaults to 1): Minimum number of tokens that cannot be filtered.

        """
        if not isinstance(top_k, int) or top_k <= 0:
            msg = f"`top_k` has to be a strictly positive integer, but is {top_k}"
            raise ValueError(msg)

        self.top_k: int = max(top_k, min_tokens_to_keep)
        self.filter_value: float = filter_value

    def __call__(self, scores: Tensor) -> Tensor:
        """Apply top-k filtering to the given scores.

        Args:
        ----
            scores: The scores to be filtered.

        Returns:
        -------
            The processed scores with values less than the top-k set to the filter value.

        """
        top_k: int = min(self.top_k, scores.size(dim=-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove: Tensor = scores < topk(input=scores, k=top_k)[0][..., -1, None]
        scores_processed: Tensor = scores.masked_fill(mask=indices_to_remove, value=self.filter_value)
        return scores_processed
