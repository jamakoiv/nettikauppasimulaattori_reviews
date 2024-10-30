# Databricks notebook source
# MAGIC %md
# MAGIC ### Language identification
# MAGIC
# MAGIC Since language analysis obviously does not work for a dataset which has mix of languages, we must identify the review language and bin them accordingly.
# MAGIC
# MAGIC For language identification we use [Stanza](https://stanfordnlp.github.io/stanza/), which should be installed on cluster-libraries instead of using pip-magic commands.
# MAGIC
# MAGIC First import the table containing the raw reviews.

# COMMAND ----------

from pyspark.sql.functions import lit

df_tv = spark.table("verkkokauppa_reviews_tv_raw")
df_phones = spark.table("verkkokauppa_reviews_phones_raw")
df_appliances = spark.table("verkkokauppa_reviews_appliances_raw")

df_tv = df_tv.withColumn("category", lit("tv"))
df_phones = df_phones.withColumn("category", lit("phone"))
df_appliances = df_appliances.withColumn("category", lit("appliance"))


# COMMAND ----------

# MAGIC %md
# MAGIC Join the reviews into single table so we don't have to save three tables downstream.

# COMMAND ----------

df = df_tv.union(df_phones).union(df_appliances)

assert df_tv.count() + df_phones.count() + df_appliances.count() == df.count(), "The number of rows in the unioned dataframe should be the sum of the rows in the original dataframes."

df.groupBy("category").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC Stanza must download the necessary language files for each used language. For language identification we only need the _multilingual_ package, and _langid_ processor.
# MAGIC Necessary language files are downloaded automatically.

# COMMAND ----------

import stanza

stanza.download("multilingual")
langid = stanza.Pipeline(lang="multilingual", processors="langid", use_gpu=False)

# COMMAND ----------

# MAGIC %md
# MAGIC We extract the review texts as regular python _list[str]_ which we can then feed to the language identification pipeline.
# MAGIC The pipeline returns stanza.Document which contains the identified language.
# MAGIC
# MAGIC __NOTE:__ With stanza it is best to use the *Pipeline.bulk_process* for processing multiple documents to take advantage of stanza's parallerization.
# MAGIC
# MAGIC __NOTE:__ It would be more spark-like to create new column via *df.withColumn("language", langid_udf)*, where we use udf-function for column transform, but stanza.Pipeline does not survive pickle.

# COMMAND ----------

review_rows = df.select("id", "text").collect()
reviews_text = [row.text for row in review_rows]
reviews_type = [type(txt) for txt in reviews_text]
reviews_id = [int(row.id) for row in review_rows]

reviews_docs = langid.bulk_process(reviews_text)
reviews_lang = [doc.lang for doc in reviews_docs]

print(reviews_lang[:10])

# COMMAND ----------

# MAGIC %md
# MAGIC Combine the original dataframe and obtained language information to create our final dataframe.

# COMMAND ----------

from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    StringType,
)

schema = StructType(
    [
        StructField("id", IntegerType(), False),
        StructField("language", StringType(), False),
    ]
)
loader = zip(reviews_id, reviews_lang)
df_lang = spark.createDataFrame(loader, schema=schema)

df_final = df.join(df_lang, "id")

# COMMAND ----------

df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC Quick check at the results, amount of different languages, and check that there are no glaring errors in language id.

# COMMAND ----------

df_final.groupBy("language").count().show()

df_final.select("language", "text").where(df_final.language == "en").head(2)
df_final.select("language", "text").where(df_final.language == "fi").head(2)

# COMMAND ----------

# MAGIC %md
# MAGIC Save the table again now with the language information added.

# COMMAND ----------

df_final.write.mode("overwrite").partitionBy("brand_name").parquet("/tmp/verkkokauppa_reviews_bronze")
df_final.write.mode("overwrite").option("overwriteSchema", "True").format("delta").saveAsTable("verkkokauppa_reviews_bronze")
