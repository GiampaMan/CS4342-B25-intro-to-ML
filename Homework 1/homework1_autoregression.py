from collections import Counter
from datasets import load_dataset
import numpy as np

# Load dataset of children's stories
dataset = load_dataset("roneneldan/TinyStories", split="train").shuffle(seed=42)
counter = Counter()  # Keep track of most common words
ENOUGH_EXAMPLES = 100000
unigram_counts = Counter()
bigram_counts = Counter()
trigram_counts = Counter()
for i, story in enumerate(dataset):
    if i == ENOUGH_EXAMPLES:
        break
    # Ignore case and punctuation such as commas and quotation marks.
    words = story['text'].upper().replace(",", "").replace("\n", " ").replace('"', '').replace("!", " ").replace(".", " ").split(' ')
    filteredWords = [ w for w in words if w != "" ]
    counter.update(filteredWords)

# Select the most common words
NUM_WORDS = 501
topWords = [ w[0] for w in counter.most_common(NUM_WORDS-1) ]  # leave room for "."
topWords.append(".")  # end-of-sentence symbol "."
wordToIdxMap = { topWords[i]:i for i in range(NUM_WORDS) }  # map from a word to its index in the topWords list

# TASK 1 (training): estimate the three probability distributions P(x_1), P(x_2 | x_1), and P(x_{t+2} | x_t, x_{t+1}).
# TODO: initialize np.array's to represent the probability distributions
for i, story in enumerate(dataset):
    if i == ENOUGH_EXAMPLES:
        break
    # Split each story into sentences, ignoring case and punctuation.
    sentences = story['text'].upper().replace(",", "").replace("\n", " ").replace('"', '').replace("!", ".").split('. ')
    for sentence in sentences:
        # Convert each sentence into a word sequence
        sentence = sentence.replace(".", "")
        words = sentence.split(" ") + ["."]
        if not set(words).issubset(topWords):  # Ignore any sentence that contains ineligible words
            continue
        # Count for unigram, bigram, and trigram 
        for j in range(len(words)):
            unigram_counts[words[j]] += 1
            if j >= 1:
                bigram_counts[(words[j-1], words[j])] += 1
            if j >= 2:
                trigram_counts[(words[j-2], words[j-1], words[j])] += 1


# TODO: normalize the probability distributions.
# Unigram prob
total_unigrams = sum(unigram_counts.values())
P1 = {w: unigram_counts[w] / total_unigrams for w in unigram_counts}

# Bigram prob
# Need count of each first word for denominator
first_word_counts = Counter()
for (w1, w2), count in bigram_counts.items():
    first_word_counts[w1] += count
P2 = {(w1, w2): bigram_counts[(w1, w2)] / first_word_counts[w1] for (w1, w2) in bigram_counts}

# Trigram prob
# Need count of each (w1, w2) pair for denominator
pair_counts = Counter()
for (w1, w2, w3), count in trigram_counts.items():
    pair_counts[(w1, w2)] += count
P3 = {(w1, w2, w3): trigram_counts[(w1, w2, w3)] / pair_counts[(w1, w2)] for (w1, w2, w3) in trigram_counts}

# TASK 2 (testing/inference): use the probability distributions to generate 100 new "sentences".
# Note: given this relatively weak 3-gram model, not all sentences will be grammatical.
# This is ok for this assignment.
# To select from any probability distribution, you can use np.random.choice.
# TODO

