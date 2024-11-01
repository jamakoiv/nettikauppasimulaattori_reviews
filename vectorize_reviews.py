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
# MAGIC
# MAGIC First import the necessary vectorizers and lemmatized review texts from upstream notebooks.

# COMMAND ----------

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfTransformer,
    TfidfVectorizer,
)
from sklearn.model_selection import train_test_split

df = spark.table("verkkokauppa_reviews_silver")

rows = df.select("id", "lemmatized_text", "rating").collect()
reviews_id = [row.id for row in rows]
reviews_rating = [row.rating for row in rows]
reviews_lemmatized_text = [row.lemmatized_text for row in rows]

# COMMAND ----------

# MAGIC %md
# MAGIC Create the vectorizer object:
# MAGIC
# MAGIC In addition of the earlier NLP-step, we can also prune the input data using the vectorizer. Words which are present in almost all or almost none of the documents are not good indicators for any difference between the documents, so these are useless for analysis purposes and can be thrown away.
# MAGIC
# MAGIC * max_df: Max document frequency. If float, throw away words which are present in higher than specified percentage of input documents. If integer, use absolute document numbers.
# MAGIC * min_df: Min document frequency. Same as *max_df_ but as lower limit.
# MAGIC * max_features: Maximum amount of features permitted.
# MAGIC
# MAGIC After creating and fitting the model, do a quick check at the bag-size. First number is the amount of texts, and the second in amount of words that survived the cutoff.

# COMMAND ----------

to_count = CountVectorizer(max_df = 0.50, min_df = 5)
count_bag = to_count.fit_transform(reviews_lemmatized_text)

print(count_bag.shape)

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
# MAGIC Create new dataframe with the results and join with original.

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, ArrayType, FloatType
res_schema = StructType(
    [
    StructField("id", IntegerType(), False),
    StructField("count", ArrayType(IntegerType(), True), False),
    StructField("tfidf", ArrayType(FloatType(), True), False)
    ]
)
res_loader = zip(reviews_id, count_bag.toarray().tolist(), tfidf_bag.toarray().tolist())

df_res = spark.createDataFrame(res_loader, schema=res_schema)
df = df.join(df_res, "id")

# COMMAND ----------

# MAGIC %md
# MAGIC Save table as usual. Save also the vocabulary since we need it for the inference step for creating a vectorizer for this word-set.

# COMMAND ----------

df.write.mode("overwrite").option("overwriteSchema", "True").format("delta").saveAsTable("verkkokauppa_reviews_gold")
df.write.mode("overwrite").parquet("tmp/verkkokauppa_reviews_gold")

voc_schema = StructType(
    [StructField("word", StringType(), False), StructField("id", IntegerType(), False)]
)
voc_values = [ int(v) for v in to_count.vocabulary_.values() ] # Convert to python int just in case.
voc_loader = zip(to_count.vocabulary_.keys(), voc_values)

df_voc = spark.createDataFrame(voc_loader, schema=voc_schema)
df_voc.write.mode("overwrite").option("overwriteSchema", "True").format("delta").saveAsTable("verkkokauppa_reviews_vocabulary_fi")
df_voc.write.mode("overwrite").parquet("tmp/verkkokauppa_reviews_vocabulary_fi")
