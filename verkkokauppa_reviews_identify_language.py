# Databricks notebook source
# MAGIC %md
# MAGIC ### Language identification
# MAGIC
# MAGIC Since language analysis obviously does not work for a dataset which has mix of languages, we must identify the review language and bin them accordingly.
# MAGIC
# MAGIC For language identification we use [Stanza](https://stanfordnlp.github.io/stanza/)
# MAGIC
# MAGIC __NOTE:__ Remember _%pip_ magic command must be on separate line.
# MAGIC

# COMMAND ----------

# MAGIC %pip install stanza --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC Import necessary spark-functions and the table containing raw-reviews.

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as f

df = spark.table("reviews_verkkokauppa_raw")
# df.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Stanza must download the necessary language files for each used language. For language identification we only need the _multilingual_ package.
# MAGIC
# MAGIC For the identification we must define a _Pipeline_ with _langid_ processor.

# COMMAND ----------

import stanza

stanza.download("multilingual")
langid = stanza.Pipeline(lang="multilingual", processors="langid", use_gpu=False)

# COMMAND ----------

# MAGIC %md
# MAGIC We extract the review texts as regular python _list[str]_ which we can then feed to the language identification pipeline.
# MAGIC The pipeline returns stanza.Document which contains the identified language.
# MAGIC
# MAGIC __NOTE:__ It would be more spark-like to create new column via *df.withColumn("language", langid_udf)*, where we use udf-function for column transform, but stanza.Pipeline does not survive pickle.

# COMMAND ----------

reviews_text = [row.text for row in df.select("text").collect()]
reviews_docs = langid.bulk_process(reviews_text)
reviews_lang = [doc.lang for doc in reviews_docs]

print(reviews_lang[:10])

# COMMAND ----------

# MAGIC %md
# MAGIC Combine the original dataframe and obtained language information to create our final dataframe.
# MAGIC
# MAGIC We do this by adding a new _id_ to both tables, and joining the tables via this _id_. This assumes the *reviews_lang* list has the languages in same order as the input dataframe, so any operation which screws with the order is forbidden.

# COMMAND ----------



df_lang = spark.createDataFrame([(lang,) for lang in reviews_lang], ["language"])

a = df.withColumn("idx",f.row_number().over(Window.orderBy(f.monotonically_increasing_id())))
b = df_lang.withColumn("idx", f.row_number().over(Window.orderBy(f.monotonically_increasing_id())))

df_final = a.join(b, a.idx == b.idx).drop("idx")

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

df_final.write.mode("overwrite").partitionBy("brand_name").parquet("/tmp/reviews_verkkokauppa_bronze")
df_final.write.mode("overwrite").format("delta").saveAsTable("reviews_verkkokauppa_bronze")
