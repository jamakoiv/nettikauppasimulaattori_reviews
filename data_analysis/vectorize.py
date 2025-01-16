# Databricks notebook source
# MAGIC %md
# MAGIC ## Vectorize the lemmatized review texts using bag-of-words model.
# MAGIC
# MAGIC Since ML-models always need numerical input for the algorithms, we must encode the information of individual words/terms/tokens into numbers. Here we use bag-of-words model, which encodes word occurances into a vector counting the amount of different words in sentences.
# MAGIC
# MAGIC E.g. for dataset of two sentences A = "food was good" and B = "food was bad", we can  define word-space
# MAGIC
# MAGIC $$\begin{gather*}
# MAGIC \text{food}, \\
# MAGIC \text{was}, \\
# MAGIC \text{good}, \\
# MAGIC \text{bad},
# MAGIC \end{gather*}$$
# MAGIC
# MAGIC and the two example sentences can be described with vectors
# MAGIC
# MAGIC $$\begin{gather*}
# MAGIC A = [1, 1, 1, 0], \\
# MAGIC B = [1, 1, 0, 1]
# MAGIC \end{gather*}$$
# MAGIC
# MAGIC In this model every word in the input material will create new dimension to the vector, so we end up with a sparse matrix since most words are rarely used.
# MAGIC Only considering single words will lose any information of word order. This is obviously bad for sentences where the order might change the meaning of the entire sentence, e.g. "cat ate dog" and "dog ate cat" would result in exactly same vectors. This can be mitigated taking higher *ngrams*, in other words considering two-word, three-word, etc. phrases. Taking only ngram = 1 of our example sentences gives word space *dog, ate, cat*, and both vectors are *[1,1,1]*. If we take ngrams = (1,2), we instead get
# MAGIC $$
# MAGIC \begin{align*}
# MAGIC \text{cat},
# MAGIC \text{ate},
# MAGIC \text{dog},
# MAGIC \text{cat ate},
# MAGIC \text{ate dog},
# MAGIC \text{dog ate},
# MAGIC \text{ate cat},
# MAGIC \end{align*}
# MAGIC $$
# MAGIC and the resulting vectors are
# MAGIC $$
# MAGIC [1, 0, 1, 1, 1, 1, 0]
# MAGIC $$
# MAGIC and
# MAGIC $$
# MAGIC [1, 1, 0, 1, 0, 1, 1]
# MAGIC $$
# MAGIC showing that we preserve atleast some information from the word order.
# MAGIC
# MAGIC First import the necessary vectorizers and lemmatized review texts from upstream notebooks.
# MAGIC
# MAGIC NOTE: We will not save the vectors. It is more convenient and faster to do the vectorization step as the parameter space can get quite large and
# MAGIC inefficient to save as regular tables. Does spark have special datatypes for sparse dataframes?

# COMMAND ----------

# MAGIC %run ../utils/run_target_helper

# COMMAND ----------

settings = get_settings(dbutils.widgets.get("TARGET"))

upstream_table = "verkkokauppa_reviews_gold" + settings["table_suffix"]

# COMMAND ----------

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.model_selection import train_test_split
from pyspark.sql.functions import col, lit

df_gold = spark.table(upstream_table)
data = df_gold.select("id", "lemmatized", "rating").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC Create the vectorizer object:
# MAGIC
# MAGIC In addition of the earlier NLP-step, we can also prune the input data using the vectorizer. Words which are present in almost all or almost none of the documents are not good indicators for any difference between the documents, so these are useless for analysis purposes and can be thrown away.
# MAGIC
# MAGIC * max_df: Max document frequency. If float, throw away words which are present in higher than specified percentage of input documents. If integer, use absolute document numbers.
# MAGIC * min_df: Min document frequency. Same as *max_df_ but as lower limit.
# MAGIC * max_features: Maximum amount of features permitted.
# MAGIC * ngram_range: Word groups to vectorize. (1,1) only takes single words, (1,2) takes single words and two word phrases, etc. 
# MAGIC
# MAGIC After creating and fitting the model, do a quick check at the bag-size. First number is the amount of texts, and the second in amount of words that survived the cutoff.
# MAGIC
# MAGIC NOTE: After checking the resulting dimensions, we can safely say that the min_df number is the dominant parameter.
# MAGIC
# MAGIC NOTE: min_df is set to very small, and ngram is quite high. This is done to generate enough words & phrases to get something out of our rather small dataset.
# MAGIC
# MAGIC TODO: Make *max_df* and *min_df* notebook parameters.

# COMMAND ----------

to_count = CountVectorizer(max_df = 0.80, min_df = 2, ngram_range=(1,4))
count_bag = to_count.fit_transform(data['lemmatized'].values)

# COMMAND ----------

# MAGIC %md
# MAGIC For actual analysis we employ TF-IDF features.
# MAGIC
# MAGIC read and explain....

# COMMAND ----------

to_tfidf = TfidfTransformer()
tfidf_bag = to_tfidf.fit_transform(count_bag)

# COMMAND ----------

# MAGIC %md
# MAGIC When actually using the vectors we can use the *TfidfVectorizer* which does the *CountVectorizer* and *TfidfTransformer* steps in single step.

# COMMAND ----------

to_tfidf = TfidfVectorizer(max_df = 0.80, min_df = 2, ngram_range=(1,4))
tfidf_bag = to_tfidf.fit_transform(data['lemmatized'].values)
