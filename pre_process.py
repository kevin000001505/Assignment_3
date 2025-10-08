import os
import re
import logging
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("main.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class Preprocessor:
    def __init__(self):
        self.dictionary = {}
        self.word_to_index = {}
        self.target_to_idx = {"<pad>": 0}

    def load_data(self, filepath: str = None) -> List[List[List[str]]]:
        """Load and preprocess data from the specified file."""
        training = self.detect_train_or_test(filepath)
        maximize_sentence_length = 0
        sentences = []
        if training:
            logger.info("Loading training data...")
            word_set = set()
            target_set = set()
        else:
            logger.info("Loading validation/test data...")

        with open(filepath, "rt", encoding="utf-8") as f:
            end = False
            for line in f:
                if line.startswith("-DOCSTART-"):
                    continue
                if line == "\n" and not end:
                    end = True
                    data = []
                elif line == "\n" and end:
                    end = False
                    maximize_sentence_length = max(maximize_sentence_length, len(data))
                    while maximize_sentence_length >= len(data):
                        data.append(["<PAD>", "<pad>"])

                    sentences.append(data)

                else:
                    tokens = line.split()
                    word = self.remain_capital_words(tokens[0])
                    target = tokens[-1]

                    data.append([word, target])
                    if training:
                        self.updated_dictionary(word, target)
                        word_set.add(word)
                        target_set.add(target)
        if training:
            self.word_to_index = {
                word: idx for idx, word in enumerate(sorted(word_set))
            }
            self.word_to_index["<PAD>"] = len(self.word_to_index)
            self.target_to_idx = {
                target: idx
                for idx, target in enumerate(
                    sorted(target_set) if target != "<pad>" else 0, start=1
                )
            }

        logger.info(
            f"Loaded {len(sentences)} sentences. Max sentence length: {maximize_sentence_length}"
        )

        return sentences

    def remain_capital_words(self, content: str) -> str:
        result = re.sub(
            r"\b(?![A-Z]{2,}\b)([A-Za-z]+)\b", lambda m: m.group(1).lower(), content
        )
        return result

    def updated_dictionary(self, word: str, target: str):
        if word not in self.dictionary:
            self.dictionary[word] = list(target)
        elif target not in self.dictionary[word]:
            self.dictionary[word].append(target)

    def detect_train_or_test(self, file_path: str) -> bool:
        if "train" in file_path:
            return True
        elif "valid" in file_path:
            return False
        elif "test" in file_path:
            return False
        else:
            raise ValueError("File path must contain 'train', 'valid', or 'test'")


if __name__ == "__main__":
    preprocessor = Preprocessor()
    train = preprocessor.load_data("conll2003/train.txt")
    valid = preprocessor.load_data("conll2003/valid.txt")
    test = preprocessor.load_data("conll2003/test.txt")
