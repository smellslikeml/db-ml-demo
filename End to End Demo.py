# Databricks notebook source
# MAGIC %md
# MAGIC # End to End ML Demo with Databricks and MLflow
# MAGIC 
# MAGIC This notebook will walk through a sample usecase of predicting customer churn. It features using MLflow to track and log our experiments and serve our final model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import Training Data
# MAGIC 
# MAGIC For this example, we'll be working customer data. 
# MAGIC The data is stored in the Delta Lake format.

# COMMAND ----------

#uncomment to run within this notebook runtime
#%run ./includes/Lakehouse-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### Featurization Logic
# MAGIC 
# MAGIC We'll read in our bronze table and do some basic transformations, including performing one-hot encoding and cleaning up the column names, to create a silver table.

# COMMAND ----------

# Using our database from here forward
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
database_name = 'ml_demo_churn_{}'.format(user.split(".")[0])
spark.sql("USE {}".format(database_name))

# COMMAND ----------

# Read into Spark
df = spark.table("bronze_customers")

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC You can always use `Pandas on Spark` if you prefer pandas syntax while leveraging spark under the hood!

# COMMAND ----------

import pyspark.pandas as ps

def compute_churn_features(data):
  
  # Convert to koalas
  data = data.to_pandas_on_spark()
  
  # OHE
  data = ps.get_dummies(data, 
                        columns=['gender', 'partner', 'dependents',
                                 'phoneService', 'multipleLines', 'internetService',
                                 'onlineSecurity', 'onlineBackup', 'deviceProtection',
                                 'techSupport', 'streamingTV', 'streamingMovies',
                                 'contract', 'paperlessBilling', 'paymentMethod'],dtype = 'int64')
  
  # Clean up column names
  data.columns = data.columns.str.replace(' ', '')
  data.columns = data.columns.str.replace('(', '-')
  data.columns = data.columns.str.replace(')', '')
  
  # Convert churnString into boolean value
  churn_values = {"Yes": 1.0, "No": 0.0}
  data['churn'] = data["churnString"].map(lambda x: churn_values[x])

  
  # Drop missing values
  data = data.dropna()
  
  return data

# COMMAND ----------

# MAGIC %md
# MAGIC After applying our transformations, we'll save the resulting dataframe into a silver Delta Lake table.

# COMMAND ----------

# Set paths
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
silver_tbl_path = '/home/{}/ml_demo_churn/silver/'.format(user)
silver_tbl_name = 'silver_customers'

# COMMAND ----------

# # Write out silver-level data to Delta lake
trainingDF = compute_churn_features(df).to_spark()

trainingDF.write.format('delta').mode('overwrite').save(silver_tbl_path)

# # Create silver table
spark.sql('''
   CREATE TABLE `{}`.{}
   USING DELTA 
   LOCATION '{}'
   '''.format(database_name,silver_tbl_name,silver_tbl_path))

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM silver_customers

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualizations
# MAGIC 
# MAGIC Create quick visualizations and plots simply by toggling the display buttons shown under the results! Here, we'll make a fast bar chart.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM silver_customers

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bonus: Compute and write features to Feature Store!
# MAGIC We can also build a feature store on top of data and then use it to train a model and deploy both the model and features to production. After executing the next cell, the table will be visible and searchable in the [Feature Store](https://e2-demo-field-eng.cloud.databricks.com/?o=1444828305810485#feature-store) -- try it!

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ```python
# MAGIC #remove %md to run
# MAGIC from databricks.feature_store import feature_table
# MAGIC from databricks.feature_store import FeatureStoreClient
# MAGIC 
# MAGIC fs = FeatureStoreClient()
# MAGIC 
# MAGIC churn_features_df = compute_churn_features(df)
# MAGIC 
# MAGIC churn_feature_table = fs.create_feature_table(
# MAGIC   name='ml_demo_churn.silver_features',
# MAGIC   keys='customerID',
# MAGIC   schema=churn_features_df.spark.schema(),
# MAGIC   description='These features are derived from the ml_demo_churn.bronze_customers table in the lakehouse.  We created dummy variables for the categorical columns, cleaned up their names, and added a boolean flag for whether the customer churned or not.  No aggregations were performed.'
# MAGIC )
# MAGIC 
# MAGIC fs.write_table(df=churn_features_df.to_spark(), name='ml_demo_churn.silver_features', mode='overwrite')
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment Tracking
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/eLearning/ML-Part-4/mlflow-tracking.png" width="800px"/>
# MAGIC 
# MAGIC This notebook walks through a basic Machine Learning example. A resulting model from one of the models will be deployed using MLflow APIs.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train using Scikit-Learn
# MAGIC 
# MAGIC The modeling here is simplistic, and just trains a plain scikit-learn random forest classifier. You can expand this example with [hyperopt]() for distributed asynchronous hyperparameter optimization.
# MAGIC 
# MAGIC MLflow is library-agnostic. You can use it with any machine learning library, and in any programming language, since all functions are accessible through a [REST API](https://mlflow.org/docs/latest/rest-api.html#rest-api) and [CLI](https://mlflow.org/docs/latest/cli.html#cli).

# COMMAND ----------

import mlflow
import mlflow.shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Define a method for reuse later
def fit_model(n_estimators=10, max_depth=5):
  
  mlflow.autolog()
  
  training_set = spark.read.table("silver_customers")

  training_pd = training_set.toPandas()
  X = training_pd.drop(["churn", "churnString", "customerID"], axis=1)
  y = training_pd["churn"]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
  # Churn is relatively rare; let's weight equally to start as it's probably at least as important as not-churning
  churn_weight = 1.0 / y_train.sum()
  not_churn_weight = 1.0 / (len(y) - y_train.sum())
  sample_weight = y_train.map(lambda churn: churn_weight if churn else not_churn_weight)

  rf_classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=-1)
  pipeline = Pipeline([("rf_classifier", rf_classifier)])
  pipeline_model = pipeline.fit(X_train, y_train, rf_classifier__sample_weight=sample_weight)

# COMMAND ----------

## Perform a mini-grid search for hyperparameter tuning with model wrapped in a function

nEstimatorsList = [5, 10]
maxDepthList = [5, 10]

for n_estimators, max_depth in [(n_estimators,max_depth) for n_estimators in nEstimatorsList for max_depth in maxDepthList]:
  fit_model(n_estimators, max_depth)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Registry
# MAGIC Typically, data scientists who use MLflow will conduct many experiments, each with a number of runs that track and log metrics and parameters.
# MAGIC 
# MAGIC During the course of this development cycle, they will select the best run within an experiment and register its model with the registry.<br>
# MAGIC Thereafter, the registry will let data scientists track multiple versions over the course of model progression as they assign each version with a lifecycle stage:
# MAGIC - Staging
# MAGIC - Production
# MAGIC - Archived
# MAGIC <br>
# MAGIC <img src="https://databricks.com/wp-content/uploads/2020/04/databricks-adds-access-control-to-mlflow-model-registry_01.jpg" width="1000px"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Deployment
# MAGIC Using MLFlow APIs, models can be deployed:
# MAGIC - In batch or streaming pipelines in Databricks with Python functions as Spark or Pandas UDFs
# MAGIC - As REST Endpoints using built-in MLflow Model Serving
# MAGIC - As Python functions in AWS SageMaker
# MAGIC - As Docker images and deployed on external infrastructure
# MAGIC 
# MAGIC And many other options! Here, we demonstrate creating a REST endpoint with MLflow Model Serving

# COMMAND ----------

# MAGIC %md
# MAGIC You can select any tracked model and load as a spark udf/pandas udf to predict on a dataframe.

# COMMAND ----------

logged_model = 'runs:/2d3f47be972641f6842f86e73ff52780/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# inference data
inference_data = spark.table("silver_customers").toPandas()

# Predict on a Pandas DataFrame.
inference_data["predictions"] = loaded_model.predict(inference_data)

display(inference_data)

# COMMAND ----------

# MAGIC %md
# MAGIC Once a model has been registered, you can construct the `model_uri` with its dedicated model name and stage label.

# COMMAND ----------

model_name = 'ML_churn_demo'
stage = 'Staging'

staged_model = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{model_name}/{stage}")

display(inference_data.withColumn('predictions', staged_model(*columns)))

# COMMAND ----------

# MAGIC %md
# MAGIC Other options for serving include [AWS Sagemaker](https://docs.databricks.com/_static/notebooks/mlflow/mlflow-quick-start-deployment-aws.html), Kubernetes, and many more!
