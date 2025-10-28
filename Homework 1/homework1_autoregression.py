from collections import Counter
from datasets import load_dataset
import numpy as np

# Load dataset of children's stories
dataset = load_dataset("roneneldan/TinyStories", split="train").shuffle(seed=42)
counter = Counter()  # Keep track of most common words
ENOUGH_EXAMPLES = 100000
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
        # TODO: process all the 3-grams in each sentence, but ignore any 3-gram if it contains 
        # a word that is not in topWords.

# TODO: normalize the probability distributions.

# TASK 2 (testing/inference): use the probability distributions to generate 100 new "sentences".
# Note: given this relatively weak 3-gram model, not all sentences will be grammatical.
# This is ok for this assignment.
# To select from any probability distribution, you can use np.random.choice.
# TODO ...
