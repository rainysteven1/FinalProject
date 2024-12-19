import spacy
from pprint import pprint

nlp = spacy.load("en_core_web_sm")
doc = nlp(
    "European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices"
)
pprint([(X.text, X.label_) for X in doc.ents])
