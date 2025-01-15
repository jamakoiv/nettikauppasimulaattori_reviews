# Databricks notebook source
# MAGIC %md
# MAGIC ### Check the input and transformed data for quick look at the dataset.
# MAGIC
# MAGIC Plot basic analytics of the dataset. This should always be done to check various pitfalls we might have in using the dataset.

# COMMAND ----------

# MAGIC %run ../utils/run_target_helper

# COMMAND ----------

settings = get_settings(dbutils.widgets.get("TARGET"))

bronze_table = "verkkokauppa_reviews_bronze" + settings["table_suffix"]
gold_table = "verkkokauppa_reviews_gold" + settings["table_suffix"]

# COMMAND ----------

# MAGIC %md
# MAGIC Import the data. We need both bronze and gold -level data since the gold level has only single language preserved at this time. Dataframes are converted to *Pandas*-dataframes for better usability with *seaborn*.

# COMMAND ----------

df_bronze = spark.table(bronze_table)
df_gold = spark.table(gold_table)
df_bronze = df_bronze.toPandas()
df_gold = df_gold.toPandas()


# COMMAND ----------

# MAGIC %md
# MAGIC Import plotting libraries and create figure for plotting.
# MAGIC
# MAGIC NOTE: If running for test-dataset, these might be seriously skewed.

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

sns.histplot(data=df_bronze, x="language", ax=ax1)
sns.histplot(data=df_gold, x="rating", hue="train_test", multiple="stack", ax=ax2)
sns.histplot(data=df_bronze, x="rating", hue="category", multiple="dodge", ax=ax3, bins=10)

ax2.set_ylabel("")  
ax3.set_ylabel("")
fig.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Most of the reviews should be in finnish, with minority being in english. A tiny fraction is also in swedish, which can be confirmed with manual inspection. There are always some tiny fraction of mis-identified languages.Generally these are texts which are in finnish but identified as something else.
# MAGIC
# MAGIC The biggest problem in the dataset is the large imbalance in the review ratings. Vast majority of the reviews are positive, rating 4 or 5, so we will have problems getting any model to recognize neutral and/or negative reviews from this dataset. We should also be able to see that our test-train -split has preserved this imbalance in the train and test-datasets.
# MAGIC
# MAGIC The rating imbalance is generally the same for all product categories, but not identical. Unfortunately due to the small amount of reviews we cannot do analysis for individual categories or brands, but have to take all reviews as single body.
