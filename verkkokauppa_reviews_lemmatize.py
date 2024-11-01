# Databricks notebook source
# MAGIC %md
# MAGIC ## Extract individual words lemmas from the reviews and create bag-of-words vectors from these words.
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Import upstream table containing cleaned reviews with language information. Extract necessary information as python-lists for NLP-pipeline.

# COMMAND ----------

df = spark.table("reviews_verkkokauppa_bronze")
df_fi = df.filter(df.language == "fi") # For now we only look Finnish reviews. Would be possible to take list of identified languages, create stanza NPL pipeline for each, and run the lemmatization.

review_rows = df_fi.select("id", "text").collect()
review_ids = [int(row.id) for row in review_rows]
review_texts = [row.text for row in review_rows]

# COMMAND ----------

# MAGIC %md
# MAGIC Import Stanza, define word categories which are to be removed from the text, and create pipeline for tokenization and lemmatization of the text.

# COMMAND ----------

import stanza

remove_upos_categories = [
    "PUNCT",  # Välimerkit
    "NUM",  # Numeraalit
    "CCONJ",  # Konjuktiot
    "SCONJ",  # Alistuskonjuktiot
    "INTJ",  # Interjektiot
    "AUX",  # Apuverbit
    "ADV",  # Adverbit
]

processors = "tokenize,mwt,pos,lemma"
nlp = stanza.Pipeline(lang="fi", processors=processors, use_gpu=True)

review_docs = nlp.bulk_process(review_texts)


# COMMAND ----------

# MAGIC %md
# MAGIC Stanza documents follow hierarchy _Document -> Sentence -> Word_, so we must unravel these to get a list of the lemmatized words. We also remove the not important word categories at this stage.

# COMMAND ----------

review_words = []
for doc in review_docs:
    lemmas = []
    for s in doc.sentences:
        for w in s.words:
            if w.upos in remove_upos_categories:
                continue
            else:
                lemmas.append(w.lemma)

    review_words.append(lemmas)

print("Total words: {}".format(sum([len(w) for w in review_words])))
import itertools
print("Number of different words/tokens: {}".format(len(set(itertools.chain(*review_words)))))

# COMMAND ----------

# MAGIC %md
# MAGIC Now we have all the lemmatized words as python list containing single word per element. Most vectorizers, e.g. sklearn _CountVectorizer_, want input as single string per data-point. So we must join the words back again.

# COMMAND ----------

vectorizer_input = []
for words in review_words:
    vectorizer_input.append(" ".join([w for w in words]))

# COMMAND ----------

# MAGIC %md
# MAGIC From the lemmatized text we create _bag-of-words_ model, which encodes amount of word occurances into a vector.
# MAGIC
# MAGIC * max_df: Ignore words which are present in more than specified proportion of documents. For example if *max_df = 0.80*, the vectorizer ignores words which are present in more than 80% of the input texts. If word is present in too many texts, it is not good indicator for any difference we want to extract from the material.
# MAGIC * min_df: Ignore words which are present in less than specified proportion of documents.
# MAGIC * max_features: Maximum amount of words taken to the bag. Since each word creates a new dimension in the resulting matrix, reducing this can improve performance.

# COMMAND ----------

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)

# TODO: Check how much the parameters affect the results
#to_count = CountVectorizer(max_df = 0.8, min_df = 0.05, max_features = 5000)
to_count = CountVectorizer(max_features = 1500)
count_bag = to_count.fit_transform(vectorizer_input)

# COMMAND ----------

# MAGIC %md
# MAGIC For actual analysis we employ TF-IDF features.
# MAGIC
# MAGIC read and explain....
# MAGIC
# MAGIC NOTE: We could also use TfidfVectorizer, which is equivalent to running CountVectorizer followed by TfidfTransformer. Here we use these separately so we can inspect individual word counts if we want.

# COMMAND ----------

to_tfidf = TfidfTransformer()
tfidf_bag = to_tfidf.fit_transform(count_bag)

# COMMAND ----------

# MAGIC %md
# MAGIC Join the results from this notebook with the original table and save the results.

# COMMAND ----------

from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
    ArrayType,
    ByteType,
    FloatType,
)

schema = StructType(
    [
        StructField("id", IntegerType(), False),
        StructField("lemmatized_text", StringType(), False),
        StructField(
            "count_vector", ArrayType(ByteType(), False), False
        ),  # Assume a single word does not ever occur more than 128 times in any review.
        StructField("tfidf_vector", ArrayType(FloatType(), False), False),
    ]
)

loader = zip(
    review_ids,
    vectorizer_input,
    count_bag.toarray().tolist(),
    tfidf_bag.toarray().tolist(),
)
df_res = spark.createDataFrame(loader, schema=schema)

df_final = df.join(df_res, "id")

assert (
    df_final.count() == df_res.count()
), "Final table is not the same length as the lemmatized text dataset. Probably JOIN failed somehow, duplicate data, or xyz..."

df_final.write.mode("overwrite").partitionBy("brand_name").parquet(
    "tmp/reviews_verkkokauppa_silver"
)
df_final.write.mode("overwrite").option("overwriteSchema", "True").format("delta").saveAsTable("reviews_verkkokauppa_silver")


# COMMAND ----------

# MAGIC %md
# MAGIC Save vocabulary as we need those for the vectorizer in the inference step.

# COMMAND ----------

print(to_count.vocabulary_.values())
print(len(to_count.vocabulary_.values()))

# COMMAND ----------

voc_schema = StructType(
    [StructField("word", StringType(), False), StructField("id", IntegerType(), False)]
)
voc_values = [ int(v) for v in to_count.vocabulary_.values() ] # Convert to python int just in case.
voc_loader = zip(to_count.vocabulary_.keys(), voc_values)

df_voc = spark.createDataFrame(voc_loader, schema=voc_schema)
df_voc.write.mode("overwrite").option("overwriteSchema", "True").format("delta").saveAsTable("reviews_verkkokauppa_vocabulary_fi")
df_voc.write.mode("overwrite").parquet("tmp/reviews_verkkokauppa_vocabulary_fi")


# COMMAND ----------

# MAGIC %md
