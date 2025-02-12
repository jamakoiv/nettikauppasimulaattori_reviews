# Databricks notebook source
# MAGIC %run ../utils/run_target_helper

# COMMAND ----------

from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC Obligatory settings-parsing and retrieve tables.

# COMMAND ----------

settings = get_settings(dbutils.widgets.get("TARGET"))
gold_table = "verkkokauppa_reviews_gold" + settings["table_suffix"]
df_gold = spark.table(gold_table)

data = df_gold.select("id", "lemmatized_text", "lemmatized_title", "positive_review", "train_test").toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC Create the vectorized bag-of-words model, and get the train and test data.

# COMMAND ----------

to_tfidf = TfidfVectorizer(max_df = 0.80, min_df = 2, ngram_range=(1,4))
tfidf_bag = to_tfidf.fit_transform(data['lemmatized_text'].values)

X_train = tfidf_bag.toarray()[data[data["train_test"] == "train"].index]
z_train = data[data["train_test"] == "train"]["positive_review"].values

X_test = tfidf_bag.toarray()[data[data["train_test"] == "test"].index]
z_test = data[data["train_test"] == "test"]["positive_review"].values

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE

# COMMAND ----------

n_estimators = 100
n_jobs = 4

names = [
    "LogReg cw-bal.",
    "RandFor. cw-bal.",
    "RandFor. undersampling, all",
    "RandFor. undersampling, not min.",
]
models = [
    LogisticRegression(class_weight="balanced"),
    RandomForestClassifier(
        n_estimators=n_estimators, class_weight="balanced", n_jobs=n_jobs
    ),
    BalancedRandomForestClassifier(
        n_estimators=n_estimators,
        # bootstrap=True,  # pyright: ignore
        sampling_strategy="all",
        replacement=True,  # pyright: ignore
        n_jobs=n_jobs,
    ),
    BalancedRandomForestClassifier(
        n_estimators=n_estimators,
        # bootstrap=False,  # pyright: ignore
        sampling_strategy="not minority",
        replacement=True,  # pyright: ignore
        n_jobs=n_jobs,
    ),
]

models = [model.fit(X_train, z_train) for model in models]
conf_mat = [confusion_matrix(z_test, model.predict(X_test)) for model in models]

# COMMAND ----------

sns.set_theme()
fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

for ax, mat, name in zip([ax1, ax2, ax3, ax4], conf_mat, names):
    sns.heatmap(mat, annot=True, fmt="d", cbar=False, cmap="flag", ax=ax)
    ax.set_title(name)

fig.supxlabel("Predicted")
fig.supylabel("Actual")
fig.suptitle("Confusion matrices. 0 = negative, 1 = positive.")
