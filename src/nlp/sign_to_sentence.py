"""
T5-Based Sign-to-Sentence Translator
=====================================
Uses a fine-tuned T5-small model (or fallback rule-based grammar corrector)
to translate ISL sign gloss sequences into fluent English sentences.

Example:
    Input:  ["I", "WATER", "WANT"]
    Output: "I want water."
"""

import logging
from typing import List, Union, Optional
from src.nlp.grammar_corrector import GrammarCorrector

logger = logging.getLogger("MTAN_NLP")


class SignToSentenceTranslator:
    """Translates sign gloss sequences into natural spoken language sentences."""

    def __init__(
        self,
        model_name: str = "t5-small",
        model_path: Optional[str] = None,
        use_fallback: bool = True,
        device: str = "cpu",
    ):
        self.use_fallback = use_fallback
        self.device = device
        self.grammar_corrector = GrammarCorrector()

        self.t5_model = None
        self.tokenizer = None

        # Attempt to load T5 model if transformers is available
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            load_target = model_path if (model_path and os.path.exists(model_path)) else model_name
            logger.info(f"Loading T5 translator from: {load_target}")
            self.tokenizer = T5Tokenizer.from_pretrained(load_target)
            self.t5_model = T5ForConditionalGeneration.from_pretrained(load_target).to(device)
            self.t5_model.eval()
        except Exception as e:
            logger.warning(f"Could not load T5 model ({e}). Using rule-based grammar corrector fallback.")

    def translate(self, sign_sequence: Union[List[str], str]) -> str:
        """Translate a sequence of recognized sign labels into a full sentence."""
        if isinstance(sign_sequence, list):
            gloss_str = " ".join(sign_sequence)
        else:
            gloss_str = sign_sequence

        if not gloss_str.strip():
            return ""

        if self.t5_model is not None and self.tokenizer is not None:
            import torch
            input_text = f"translate signs: {gloss_str}"
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.t5_model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
            sentence = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return sentence

        # Fallback to rule-based grammar corrector
        return self.grammar_corrector.correct(sign_sequence)

    def __call__(self, sign_sequence: Union[List[str], str]) -> str:
        return self.translate(sign_sequence)
