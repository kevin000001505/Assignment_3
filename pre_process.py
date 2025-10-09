import os
import re
import logging
from typing import List, Tuple

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
        self.target_to_idx = {"<pad>": 0, "O": 1}

    def load_data(self, filepath: str = None) -> Tuple[List[List[str]]]:
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
            data = []
            for line in f:
                if line.startswith("-DOCSTART-"):
                    continue

                if line == "\n":
                    if data:
                        sentences.append(data)
                        maximize_sentence_length = max(
                            maximize_sentence_length, len(data)
                        )
                    data = []

                else:
                    tokens = line.split()
                    word = self.remain_capital_words(tokens[0])
                    target = tokens[-1]

                    data.append([word, target])
                    if training:
                        self.updated_dictionary(word, target)
                        word_set.add(word)
                        target_set.add(target)

        sentences = self.make_every_sentence_same_length(
            sentences, maximize_sentence_length
        )

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
        output = self.return_training(sentences)
        return output

    def remain_capital_words(self, content: str) -> str:
        result = re.match(r"^[A-Z]{1}[a-z]+$", content)
        return content.lower() if result else content

    def make_every_sentence_same_length(
        self, sentences: List[List[List[str]]], max_length: int
    ) -> List[List[List[str]]]:

        for sentence in sentences:
            current_length = len(sentence)

            if current_length < max_length:
                num_padding = max_length - current_length
                padding = [["<PAD>", "<pad>"]] * num_padding
                sentence.extend(padding)

        return sentences

    def return_training(self, sentences: List[List[List[str]]]) -> List[List[str]]:
        response = []
        for sentence in sentences:

            training = []
            targets = []
            for word, target in sentence:
                training.append(word)
                targets.append(target)
            response.append((training, targets))
        return response

    def updated_dictionary(self, word: str, target: str):
        if word not in self.dictionary:
            self.dictionary[word] = [target]
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
