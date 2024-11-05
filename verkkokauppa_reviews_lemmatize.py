# Databricks notebook source
# MAGIC %md
# MAGIC ## Extract individual words lemmas from the reviews.

# COMMAND ----------

# MAGIC %md
# MAGIC Import upstream table containing cleaned reviews with language information. Extract necessary information as python-lists for NLP-pipeline.

# COMMAND ----------

from pyspark.sql.functions import col

df = spark.table("verkkokauppa_reviews_bronze")
df_fi = df.filter(df.language == "fi") # For now we only look Finnish reviews. Would be possible to take list of identified languages, create stanza NPL pipeline for each, and run the lemmatization.

review_rows = df_fi.select("id", "text", "title").where(col("title").isNotNull()).collect()
review_ids = [int(row.id) for row in review_rows]
review_texts = [row.text for row in review_rows]
review_titles = [row.title for row in review_rows]

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

review_text_docs = nlp.bulk_process(review_texts)
review_title_docs = nlp.bulk_process(review_titles)

# COMMAND ----------

# MAGIC %md
# MAGIC Stanza documents follow hierarchy _Document -> Sentence -> Word_, so we must unravel these to get a list of the lemmatized words. We also remove the not important word categories at this stage.

# COMMAND ----------

def get_lemmas(docs: list[stanza.Document]) -> list[list[str]]:
    words = []
    
    for doc in docs:
        lemmas = []
        for s in doc.sentences:
            for w in s.words:
                if w.upos in remove_upos_categories:
                    continue
                else:
                    lemmas.append(w.lemma)

        words.append(lemmas)
    
    return words

review_text_words = get_lemmas(review_text_docs)
review_title_words = get_lemmas(review_title_docs)

import itertools
print("Total words in texts: {}".format(sum([len(w) for w in review_text_words])))
print("Number of different words/tokens in texts: {}".format(len(set(itertools.chain(*review_text_words)))))
print("Total words in titles: {}".format(sum([len(w) for w in review_title_words])))
print("Number of different words/tokens in titles: {}".format(len(set(itertools.chain(*review_title_words)))))

# COMMAND ----------

# MAGIC %md
# MAGIC Now we have all the lemmatized words as python list containing single word per element. Most vectorizers, e.g. sklearn _CountVectorizer_, want input as single string per data-point. So we must join the words back again.

# COMMAND ----------

vectorizer_input_text = []
for words in review_text_words:
    vectorizer_input_text.append(" ".join([w for w in words]))

vectorizer_input_title = []
for words in review_title_words:
    vectorizer_input_title.append(" ".join([w for w in words]))


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
        StructField("lemmatized_title", StringType(), False),
    ]
)

loader = zip(
    review_ids,
    vectorizer_input_text,
    vectorizer_input_title,
)
df_res = spark.createDataFrame(loader, schema=schema)

df_final = df.join(df_res, "id")

assert (
    df_final.count() == df_res.count()
), "Final table is not the same length as the lemmatized text/title dataset. Probably JOIN failed somehow, duplicate data, or xyz..."

df_final.write.mode("overwrite").partitionBy("brand_name").parquet(
    "tmp/verkkokauppa_reviews_silver"
)
df_final.write.mode("overwrite").option("overwriteSchema", "True").format("delta").saveAsTable("verkkokauppa_reviews_silver")


# COMMAND ----------

# MAGIC %md
