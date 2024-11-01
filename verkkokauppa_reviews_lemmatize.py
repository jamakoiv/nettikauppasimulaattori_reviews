# Databricks notebook source
# MAGIC %md
# MAGIC ## Extract individual words lemmas from the reviews.

# COMMAND ----------

# MAGIC %md
# MAGIC Import upstream table containing cleaned reviews with language information. Extract necessary information as python-lists for NLP-pipeline.

# COMMAND ----------

df = spark.table("verkkokauppa_reviews_bronze")
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
    #"CCONJ",  # Konjuktiot
    #"SCONJ",  # Alistuskonjuktiot
    #"INTJ",  # Interjektiot
    #"AUX",  # Apuverbit
    #"ADV",  # Adverbit
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
    ]
)

loader = zip(
    review_ids,
    vectorizer_input
)
df_res = spark.createDataFrame(loader, schema=schema)

df_final = df.join(df_res, "id")

assert (
    df_final.count() == df_res.count()
), "Final table is not the same length as the lemmatized text dataset. Probably JOIN failed somehow, duplicate data, or xyz..."

df_final.write.mode("overwrite").partitionBy("brand_name").parquet(
    "tmp/verkkokauppa_reviews_silver"
)
df_final.write.mode("overwrite").option("overwriteSchema", "True").format("delta").saveAsTable("verkkokauppa_reviews_silver")


# COMMAND ----------

# MAGIC %md
