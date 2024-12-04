# Databricks notebook source
# MAGIC %md
# MAGIC ### Hyperparameter optimization with Optuna.
# MAGIC
# MAGIC Default machine-learning -algorithm parameters are seldom the best possible ones, and we can usually improve the results by performing a hyperparameter optimization for the most promising canditate.
# MAGIC
# MAGIC Most ML frameworks offer some sort of automated tooling for optimizing hyperparameters (e.g. *GridsearchCV* in sklearn), or you can manually loop over possible parameters 

# COMMAND ----------

# MAGIC %run ./data_helpers

# COMMAND ----------

# MAGIC %run ../utils/run_target_helper

# COMMAND ----------

# MAGIC %md
# MAGIC First import necessary libraries. Optuna is not included in the databricks ML runtime and should be configured manually from *compute->cluster->libraries*.
# MAGIC
# MAGIC NOTE: install both *optuna* and *optuna-integration*

# COMMAND ----------

import optuna

from sklearn.metrics import confusion_matrix
from pyspark.sql import SparkSession
from imblearn.ensemble import BalancedRandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# from data_helpers import get_training_dataset

# COMMAND ----------

# MAGIC %md
# MAGIC Optuna works by defining a study and a target function. The target function must define a parameter space for the parameters of interest, and return a score which can be used to track how good the parameters were.
# MAGIC The study will generate trials for the target function, and track the parameters which have already been tried.
# MAGIC
# MAGIC Very simplistic example:

# COMMAND ----------

example_study = optuna.create_study(direction="minimize")

def target_function(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", -10, 10)
    return x**2 - (3*x)

example_study.optimize(target_function, n_trials=50)


# COMMAND ----------

# MAGIC %md
# MAGIC After running the trials we can check the best found parameters.

# COMMAND ----------

print(example_study.best_trial)
print("Best found parameters: {}".format(example_study.best_trial.params))

# COMMAND ----------

# MAGIC %md
# MAGIC In a more complex case it is best practice to define a class, which can handle all the non-optuna based initialization etc. in a separate function (usually *init*), and all the optuna-related things to the class *call* -function.

# COMMAND ----------

class OptunaObjective():
    def __init__(self, X_train, X_test, z_train, z_test):
        self.X_train = X_train
        self.X_test = X_test
        self.z_train = z_train
        self.z_test = z_test

    def __call__(self, trial: optuna.Trial):
        n_estimators = trial.suggest_int("n_estimators", 10, 500, step=10)
        max_leaf_nodes = trial.suggest_categorical("max_leaf_nodes", [None])
        max_depth = trial.suggest_categorical("max_depth", [None])
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10, step=2)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10, step=1)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
        # random_state = trial.suggest_int("random_state", 1, 100000, step=1) # Should we use this as parameter? 
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        replacement = trial.suggest_categorical("replacement", [True, False])
        sampling_strategy = trial.suggest_categorical("sampling_strategy", ["majority", "not minority", "not majority", "all"])
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

        model = BalancedRandomForestClassifier(
            n_estimators=n_estimators,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            max_features=max_features, # pyright: ignore
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            #random_state=random_state,
            bootstrap=bootstrap, # pyright: ignore
            replacement=replacement, # pyright: ignore
            sampling_strategy=sampling_strategy,
            criterion=criterion,
        )

        model = model.fit(self.X_train, self.z_train) 

        pred = model.predict(self.X_test)
        print(confusion_matrix(z_test, pred))
        
        # Guard against models which give same label to all datapoints.
        # These might give best result if the data is imbalanced, 
        # but are very uninformative.
        if (pred == 0).all() or (pred == 1).all():
            score = 0.0
        else:
            score = model.score(self.X_test, self.z_test)

        return score


# COMMAND ----------

# MAGIC %md
# MAGIC Import data as usual.

# COMMAND ----------

settings = get_settings(dbutils.widgets.get("TARGET"))
n_trials = int(dbutils.widgets.get("N_TRIALS"))

gold_table = "verkkokauppa_reviews_gold" + settings["table_suffix"]
df_gold = spark.table(gold_table)

X_train, X_test, y_train, y_test, z_train, z_test = get_training_dataset(df_gold)

# COMMAND ----------

# MAGIC %md
# MAGIC For parallel optimization in spark, we can use joblib with the spark-backend.

# COMMAND ----------

import joblib
from joblibspark import register_spark

register_spark() # register Spark backend for Joblib

# When calling study.optimize, wrap in in the context manager 
# "with joblib.parallel_backend("spark", n_jobs=-1):"

# COMMAND ----------

# MAGIC %md
# MAGIC Databricks MLflow integration does track the scikit-learn model runs automatically, but for better integration it is advised to create a callback funtion using optunas *MLFlowCallback*.

# COMMAND ----------

import mlflow
from optuna.integration.mlflow import MLflowCallback


experiment_name = "/Users/jaakko.m.koivisto@gmail.com/reviews"
mlflow.set_experiment(experiment_name)
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
print(experiment_name, experiment_id)

mlflow_callback = MLflowCallback(
    tracking_uri="databricks",
    metric_name="accuracy",
    create_experiment=False,
    mlflow_kwargs={
        "experiment_id": experiment_id
        }
)

# COMMAND ----------


study = optuna.create_study(direction="maximize")    
obj = OptunaObjective(X_train, X_test, z_train, z_test)

with joblib.parallel_backend("spark", n_jobs=4):
    study.optimize(obj, n_trials=n_trials, callbacks=[mlflow_callback])

# COMMAND ----------

# MAGIC %md
# MAGIC Print out the best found parameters.

# COMMAND ----------

print(study.best_trial.params)
