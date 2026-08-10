"""Optional transformer-attention inspection for curriculum demonstrations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionSummary:
    """A compact description of attention tensors returned by a Transformer model."""

    tokens: list[str]
    layers: int
    heads: int
    sequence_length: int


class TransformerAttentionInspector:
    """Inspect attention from a locally cached or explicitly downloaded Hugging Face model."""

    def __init__(self, model_name: str = "distilbert-base-uncased") -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise RuntimeError("Install the transformer extra: pip install -e './backend[transformer]'.") from error
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name, output_attentions=True)

    def inspect(self, text: str) -> AttentionSummary:
        """Run one forward pass and summarize attention dimensions, not model predictions."""

        encoded = self._tokenizer(text, return_tensors="pt")
        outputs = self._model(**encoded, output_attentions=True)
        attentions = outputs.attentions
        if not attentions:
            raise RuntimeError("The selected transformer model did not return attention tensors.")
        first_layer = attentions[0]
        return AttentionSummary(
            tokens=self._tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]),
            layers=len(attentions),
            heads=first_layer.shape[1],
            sequence_length=first_layer.shape[-1],
        )
