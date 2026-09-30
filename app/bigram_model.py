import random
import re
from collections import Counter, defaultdict


class BigramModel:
    def __init__(self, corpus):
        self.bigram_probs = defaultdict(dict)
        bigram_counts = Counter()

        for text in corpus:
            words = re.findall(r"\b\w+\b", text.lower())
            bigram_counts.update(zip(words[:-1], words[1:]))

        # Count how often each word has a next word.
        next_word_counts = Counter()
        for (word1, word2), count in bigram_counts.items():
            next_word_counts[word1] += count

        for (word1, word2), count in bigram_counts.items():
            self.bigram_probs[word1][word2] = (
                count / next_word_counts[word1]
            )

    def generate_text(self, start_word, length=20):
        current_word = start_word.strip().lower()
        generated_words = [current_word]

        for _ in range(length - 1):
            next_words = self.bigram_probs.get(current_word)
            if not next_words:
                break

            current_word = random.choices(
                list(next_words.keys()),
                weights=list(next_words.values()),
            )[0]
            generated_words.append(current_word)

        return " ".join(generated_words)