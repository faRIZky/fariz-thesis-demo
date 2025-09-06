# NER_Modeling.py
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import re
from collections import defaultdict


class NER_Modeling:
    def __init__(self, model_name: str, cache_dir: str):
        """Initialize the tokenizer and model."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir=cache_dir)
        except Exception as e:
            raise RuntimeError(f"Error loading model or tokenizer: {e}")

    def run_ner(self, text: str):
        """
        Perform Named Entity Recognition (NER) on the input text.

        Args:
            text (str): Input text to process.

        Returns:
            dict: Dictionary containing token-label-score results and description.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=-1)
        token_scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        labels = predictions[0].tolist()

        token_label_scores = []
        for token, label_id, scores in zip(tokens, labels, token_scores[0]):
            if token not in ["[CLS]", "[SEP]"]:
                label = self.model.config.id2label[label_id]
                score_percent = f"{scores[label_id] * 100:.1f}%"
                token_label_scores.append((token, label, score_percent))

        description = """
                BIO Format:
                - B- (Beginning): Start of an entity
                - I- (Inside): Continuation of the same entity
                - O (Outside): Not an entity

                Entity Labels:
                - B-PER, I-PER: Person
                - B-LOC, I-LOC: Location
                - B-GEO, I-GEO: Geographical Entity
                - B-ORG, I-ORG: Organization
                - B-GPE, I-GPE: Geopolitical Entity
                - B-TIM, I-TIM: Time Expression
                - B-ART, I-ART: Artifact
                - B-EVE, I-EVE: Event
                - B-NAT, I-NAT: Natural Phenomenon
        """
        return {
            "results": token_label_scores,
            "description": description
        }

    def generate_questions(self, text: str, ner_results: list, max_per_entity_type=2):
        """
        Generate fill-in-the-blank questions from NER output.

        Args:
            text (str): Original input text.
            ner_results (list): List of (token, label, score).
            max_per_entity_type (int): Maximum number of questions per entity type.

        Returns:
            dict: Mapping from entity type to list of questions.
        """
        questions = defaultdict(list)
        entity_buffer = []
        current_label = None
        entities = []

        for token, label, _ in ner_results:
            if label.startswith("B-"):
                if entity_buffer:
                    entities.append((current_label, entity_buffer))
                entity_buffer = [token]
                current_label = label[2:]
            elif label.startswith("I-") and current_label:
                entity_buffer.append(token)
            else:
                if entity_buffer:
                    entities.append((current_label, entity_buffer))
                entity_buffer = []
                current_label = None
        if entity_buffer:
            entities.append((current_label, entity_buffer))

        for label_type, tokens in entities:
            # Convert subword tokens back to readable text
            ent_text = self.tokenizer.convert_tokens_to_string(tokens).replace(" ##", "")
            pattern = re.escape(ent_text.strip())
            if len(questions[label_type]) < max_per_entity_type:
                blank_text = re.sub(pattern, "______", text, flags=re.IGNORECASE)
                questions[label_type].append(blank_text)

        return questions
