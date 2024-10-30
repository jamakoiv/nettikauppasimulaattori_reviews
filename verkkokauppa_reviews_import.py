# Databricks notebook source
# MAGIC %md
# MAGIC ## Import verkkokauppa reviews from Azure cosmos (mongoDB connection). Perform basic pruning, throw away NULLs etc., finally save reviews to delta-table.
# MAGIC
# MAGIC First define connection parameters for _mongodb_.
# MAGIC
# MAGIC - _connectionString_ can be found from Azure cosmosDB settings.
# MAGIC - _database_ and _collection_ are user defined and depend on how you set up the database.

# COMMAND ----------

connectionString='mongodb+srv://jamakoiv:{passwd}@cosmos-mongo-testi.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000&wtimeoutMS=0'.format(passwd=dbutils.secrets.get(scope="azure_mongodb", key="mongopasswd"))

database = dbutils.widgets.get("DATABASE")
collection = dbutils.widgets.get("COLLECTION")
downstream_table = dbutils.widgets.get("DOWNSTREAM_TABLE") 

# COMMAND ----------

# MAGIC %md ##Read  data from the MongoDB collection running an Aggregation Pipeline.
# MAGIC
# MAGIC When debugging we run the notebook with limited dataset using aggregation component _$limit_. First we get the limit value from the notebook parameters.
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

limit = int(dbutils.widgets.get("LIMIT"))
if limit <= 0: 
    __LIMIT__ = ""
else:
    __LIMIT__ = f"{{$limit: {limit}}},"


# COMMAND ----------

# MAGIC %md
# MAGIC The pipeline transforms the data from form _products -> reviewsById -> [revA, revB, revC]_, which groups reviews by product ID, to form _revA -> [text, rating, ...], revB -> [...], revC -> [...]_ where each review is separate entry. This form is alot easier to handle for analysis.
# MAGIC
# MAGIC __NOTE:__ Elements from 'prod' have to be extracted using arrayElemAt. Elementst from 'review' don't since we unwind them.

# COMMAND ----------

pipeline_raw = '''[
    __LIMIT__ 
    {$project:
        {
        review: {$objectToArray: "$products.reviewsById"},
        prod: {$objectToArray: "$products.detailsByPid"}
        }
    },
    {$unwind: "$review"},
    {$project: 
        {
            id: "$review.k",
            product_id: {$arrayElemAt: ["$prod.v.pid", 0]},
            product_id_alt: "$review.v.pid.value",
            brand_name: { $toLower: {$arrayElemAt: ["$prod.v.brandName", 0]} },
            product_name: {$arrayElemAt: ["$prod.v.name", 0]}, 
            text: "$review.v.reviewText",
            title: "$review.v.title",
            rating: "$review.v.rating",
            reviewerAge: "$review.v.contextDataValues.age.value"
        }
    } 
]'''
pipeline = pipeline_raw.replace("__LIMIT__", __LIMIT__)

df = spark.read.format("com.mongodb.spark.sql.DefaultSource").option("database", database).option("collection", collection).option("pipeline", pipeline).option("spark.mongodb.input.uri", connectionString).load()
df = df.drop("_id")  # Drop mongoDB objectID.

df.show()

# COMMAND ----------

df.select("brand_name").distinct().show()

# df.where(df.brand_name.contains("LG")).show()

# COMMAND ----------

# MAGIC %md
# MAGIC Since AzureMongo does not support __$$NOW__ currently, we must add timestamp-column manually using _current_timestamp()_.

# COMMAND ----------

from pyspark.sql.functions import current_timestamp

df = df.withColumn("lastModified", current_timestamp())

# COMMAND ----------

# MAGIC %md
# MAGIC Quick check at the results. Amount of different *product_ids* should give same count as our pipeline _$limit_, if we are using a limit.

# COMMAND ----------

df.groupBy("product_id").count().show()

print("Amount of different product_ids: {}".format(df.select("product_id").distinct().count()))
print("Amount of reviews in total: {}".format(df.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC Some products are semi-duplicates, e.g. same model of TV in different panel sizes, but the reviews are shown for all different models. This leads to some reviews being introduced to the original database multiple times. The duplicate reviews can be identified by the review-id.
# MAGIC
# MAGIC Also remove all entries where there is no review text. All entries should have one but stuff happens...

# COMMAND ----------

df = df.dropDuplicates(["id"])
df = df.where(df.text.isNotNull())

# COMMAND ----------

# MAGIC %md
# MAGIC Save tables as _parquet_, for exporting to local machine for testing, and in _delta_, for usage in Databricks.

# COMMAND ----------

df.write.mode("overwrite").option("overwriteSchema", "True").format("delta").partitionBy("brand_name").saveAsTable(downstream_table)
