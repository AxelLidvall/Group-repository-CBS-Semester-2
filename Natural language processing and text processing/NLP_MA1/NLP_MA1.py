# Import the required libraries
import pandas as pd 
import spacy
from spacy.pipeline import EntityRuler
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter
from prettytable import PrettyTable

# Define the path to the Excel file and stored the specified column name in a variable
path = "sample.xlsx"
col = "SOS Tweet / SOS Message"

# 1) Load the Excel file "sample.xlsx"
df = pd.read_excel(path, skiprows= 8, skipfooter=5, usecols=[col])

# 2) Nested list with each tweet as a separate list
tw = (
    df[col]
    .dropna() 
    .astype(str) 
    .str.strip() 
    .tolist() 
)

# 3) setup spaCy for tokenization
nlp = spacy.load("en_core_web_sm")

# As the tweets anonymizes names with a pattern like "A***B", we can add a custom entity ruler to recognize these as PERSON entities.
ruler = nlp.add_pipe("entity_ruler", before="ner")
patterns = [
    {
        "label": "PERSON",
        "pattern": [{"TEXT": {"REGEX": r"^[A-Za-z]\*+[A-Za-z]$"}}]
    }
]
ruler.add_patterns(patterns)

# Set up the stop words: default + extras
stop_words = set(STOP_WORDS) | {'helpðÿ', '™', '\x8fðÿ', '\x8f', 'ðÿ', '+', '@', "#", "*", 'a+'}
#stop_words.update({'helpðÿ', '™', '\x8fðÿ', '\x8f', 'ðÿ', '+', '@', "#", "*", 'a+'})


# tokenize the tweets using spaCy's nlp.pipe with batch processing for efficiency
tokenized_tweets = [
    [tok.text for tok in doc]
    for doc in nlp.pipe(tw, batch_size=200)
]


# clean the tokenized tweets by removing stop words, punctuation, and whitespace, and converting to lowercase
cleaned_tweets = [
    [
        tok.text.lower().strip("-") # remove leading/trailing "-" but keep internal hyphens
        for tok in doc
        if not tok.is_space 
        and not tok.is_punct 
        and tok.text.lower() not in stop_words 
    ]
    for doc in nlp.pipe(tw, batch_size=200)
]


# POS tagging
cleaned_texts = [" ".join(tokens) for tokens in cleaned_tweets]

pos_counts = Counter(
    tok.pos_ 
    for doc in nlp.pipe(cleaned_texts, batch_size=200) 
    for tok in doc
)

#the POS counts into a pretty table
pos_tab = PrettyTable(["POS", "Count"])
for pos,count in pos_counts.most_common():
    pos_tab.add_row([pos,count])

print(pos_tab)

# NER counts

entity_counts = Counter(
    ent.label_ 
    for doc in nlp.pipe(cleaned_texts, batch_size=200) 
    for ent in doc.ents
)

ner_tab = PrettyTable(["NER", "Count"])
for ner,count in entity_counts.most_common():
    ner_tab.add_row([ner,count])

print(ner_tab)