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
    """Preprocess CONLL-style data into fixed-length sentences and build mappings."""

    def __init__(self):
        """Initialize internal mappings, sets, and max sentence length."""
        # Explanation: Set up containers for vocabulary/tags, index mappings, and tracking
        # of the maximum sentence length found in training data.
        self.dictionary = {}
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.target_to_idx = {}
        self.idx_to_target = {}
        self.maximize_sentence_length = 0
        self.word_set = set()
        self.target_set = set()
        self.target_set.add("<pad>")

    def load_data(self, filepath: str) -> List[Tuple[List[str], List[str]]]:
        """Load and preprocess data from the specified file.

        Returns a list of (words_list, targets_list) where each sentence has been
        padded or truncated to the maximum sentence length (for training files the
        maximum is inferred from the data). Also builds word/target mappings for
        training files.
        """
        # Explanation: Determine if file is training (to build mappings and compute
        # max length). Read CONLL-style lines, split sentences on blank lines,
        # normalize words, collect tags, track max length, then pad/truncate all
        # sentences to that length. Finally build index mappings for words/tags
        # for training files and return parallel (words, tags) per sentence.
        training = self.detect_train_or_test(filepath)
        sentences = []
        logger.info(f'Loading {"training" if training else "validation/test"} data...')

        with open(filepath, "rt", encoding="utf-8") as f:
            data = []
            for line in f:
                if line.startswith("-DOCSTART-"):
                    continue

                if line == "\n":
                    if data:
                        sentences.append(data)
                        if training:
                            self.maximize_sentence_length = max(
                                self.maximize_sentence_length, len(data)
                            )
                    data = []

                else:
                    tokens = line.split()
                    word = self.remain_capital_words(tokens[0])
                    target = tokens[-1]

                    data.append([word, target])
                    if training:
                        self.updated_dictionary(word, target)
                        self.word_set.add(word)
                        self.target_set.add(target)

        sentences = self.make_every_sentence_same_length(
            sentences, self.maximize_sentence_length
        )

        if training:
            self.word_to_idx = {
                word: idx for idx, word in enumerate(sorted(self.word_set))
            }
            self.word_to_idx["<PAD>"] = len(self.word_to_idx)
            self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

            self.target_to_idx = {
                target: idx
                for idx, target in enumerate(sorted(t for t in self.target_set))
            }
            self.idx_to_target = {idx: tag for tag, idx in self.target_to_idx.items()}

        logger.info(
            f"Loaded {len(sentences)} sentences. Max sentence length: {self.maximize_sentence_length}"
        )
        output = self.return_training(sentences)
        return output

    def remain_capital_words(self, content: str) -> str:
        """Return the word lowercased only if it matches Capitalized format (e.g., 'London').

        Leaves other tokens unchanged.
        """
        # Explanation: If the token matches a single capital letter followed by lowercase
        # letters (Proper Noun style), convert it to lowercase; otherwise keep as-is.
        result = re.match(r"^[A-Z]{1}[a-z]+$", content)
        return content.lower() if result else content

    def make_every_sentence_same_length(
        self, sentences: List[List[List[str]]], max_length: int
    ) -> List[List[List[str]]]:
        """Pad or truncate each sentence to the given max_length.

        Pads with ["<PAD>", "<pad>"] pairs. Ensures all sentences returned have
        length == max_length.
        """
        # Explanation: For each sentence, append PAD tokens up to max_length or truncate
        # longer sentences. Assert that all sentences end up exactly max_length.
        for i, sentence in enumerate(sentences):
            current_length = len(sentence)

            if current_length < max_length:
                num_padding = max_length - current_length
                padding = [["<PAD>", "<pad>"]] * num_padding
                sentence.extend(padding)
            else:
                sentences[i] = sentence[:max_length]

        assert all(
            len(s) == max_length for s in sentences
        ), "Not all sentences are the same length"

        return sentences

    def return_training(
        self, sentences: List[List[List[str]]]
    ) -> List[Tuple[List[str], List[str]]]:
        """Convert list of [word, target] pairs per sentence into tuple (words, targets)."""
        # Explanation: Split each sentence’s [word, tag] pairs into two parallel lists:
        # one containing words and one containing tags.
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
        """Record a mapping from a word to possible target tags (avoids duplicates)."""
        # Explanation: Maintain a dictionary from word -> list of unique tags seen
        # for that word in the training data.
        if word not in self.dictionary:
            self.dictionary[word] = [target]
        elif target not in self.dictionary[word]:
            self.dictionary[word].append(target)

    def detect_train_or_test(self, file_path: str) -> bool:
        """Detect whether the file is training data based on filename.

        Returns True for 'train' in path, False for 'valid' or 'test'. Raises
        ValueError if none of these substrings are present.
        """
        # Explanation: Use filename substring matching to decide processing mode and
        # raise if the path does not include an expected split indicator.
        if "train" in file_path:
            return True
        elif "valid" in file_path:
            return False
        elif "test" in file_path:
            return False
        else:
            raise ValueError("File path must contain 'train', 'valid', or 'test'")


if __name__ == "__main__":
    # Explanation: Example usage that loads train/valid/test from a local conll2003 folder.
    preprocessor = Preprocessor()
    train = preprocessor.load_data("conll2003/train.txt")
    valid = preprocessor.load_data("conll2003/valid.txt")
    test = preprocessor.load_data("conll2003/test.txt")
