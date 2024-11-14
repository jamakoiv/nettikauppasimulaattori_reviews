# Databricks notebook source
import numpy as np
from pyspark.sql.functions import col
from pyspark.sql import dataframe

def get_training_dataset(df: dataframe) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  rows_train = df.select("rating", "tfidf", "positive_review").where(col("train_test") == "train").collect()

  X_train = np.array([row.tfidf for row in rows_train])
  y_train = np.array([row.rating for row in rows_train])
  z_train = np.array([row.positive_review for row in rows_train])

  rows_test = df.select("rating", "tfidf", "positive_review").where(col("train_test") == "test").collect()

  X_test = np.array([row.tfidf for row in rows_test])
  y_test = np.array([row.rating for row in rows_test])
  z_test = np.array([row.positive_review for row in rows_test])

  return X_train, X_test, y_train, y_test, z_train, z_test

