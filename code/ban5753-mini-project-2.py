import os
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.types import * 
from pyspark.sql import SQLContext
from pyspark.mllib.stat import Statistics
from pyspark.sql.functions import udf
from pyspark.ml.feature import StringIndexer, VectorAssembler,StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import when
from pyspark.ml.classification import LogisticRegression 
from pyspark.ml.evaluation import BinaryClassificationEvaluator 
from pyspark.ml.classification import RandomForestClassifier,DecisionTreeClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import LinearSVC
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import Imputer

# Initiating Spark Session
spark=SparkSession.builder.master ("local[*]").appName("MiniProject").getOrCreate()
sc=spark.sparkContext
sqlContext=SQLContext(sc)

# Accessing the data from the correct data directory
# Linked to the current github repo structure
cwd = os.getcwd()
data_dir = cwd.replace("\code","\data")
df = spark.read.option('header','True').option("InferSchema",'True').csv('{}\XYZ_Bank_Deposit_Data_Classification.csv'.format(data_dir), sep = ";")
df = df.withColumnRenamed("emp.var.rate","emp_var_rate").withColumnRenamed('cons.price.idx','cons_price_idx').withColumnRenamed("cons.conf.idx",'cons_conf_idx')
df = df.withColumnRenamed('nr.employed','nr_employed')

# Explore the outcome variable and its distribution
df.groupby("y").count().show()

# Identifying numerical and categorical features based on their d-types
numeric_features = []
categorical_features = []
for x,y in df.dtypes:
    if y in ['int','double']:
        numeric_features.append(x)
    else:
        categorical_features.append(x)

# For numerical features, describe the mean, standard deviation, minimum and maximum variables
print(df.select(numeric_features).describe().toPandas().transpose())

#Variable distributions
fig = plt.figure(figsize=(25,15)) ## Plot Size 
st = fig.suptitle("Distribution of Features", fontsize=50,
                  verticalalignment='center') # Plot Main Title 

for col,num in zip(df.toPandas().describe().columns, range(1,11)):
    ax = fig.add_subplot(3,4,num)
    ax.hist(df.toPandas()[col])
#     plt.style.use('dark_background') 
    plt.grid(False)
    plt.xticks(rotation=45,fontsize=20)
    plt.yticks(fontsize=15)
    plt.title(col.upper(),fontsize=20)
plt.tight_layout()
st.set_y(0.95)
fig.subplots_adjust(top=0.85,hspace = 0.4)
plt.show()

# Explore the correlations between numerical features
print(df.select(numeric_features).toPandas().corr())

# Using an imputer with default settings on numerical features
imputer=Imputer(
    inputCols=["age","duration","campaign","pdays","previous","emp_var_rate","cons_price_idx","cons_conf_idx","euribor3m","nr_employed"],
    outputCols=["age","duration","campaign","pdays","previous","emp_var_rate","cons_price_idx","cons_conf_idx","euribor3m","nr_employed"]
    )
model=imputer.fit(df)
raw_data=model.transform(df)

# Using a string indexer on categorical features
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index") for col in categorical_features]
pipeline = Pipeline(stages=indexers)
df_converted = pipeline.fit(df).transform(raw_data)
for x in categorical_features:
    df_converted = df_converted.drop(x)

df_x = df_converted.drop('y_index')
# Creating train-test data as 60-40 split on the converted dataframe
train_data, test_data = df_converted.randomSplit([0.6, 0.4]) 

# Logistic Regression
# Standard Scaler used on numerical features
scaler = StandardScaler(inputCol= 'features_s', outputCol="scaled_features", withStd=True, withMean=True)
assembler2 = VectorAssembler().setInputCols(numeric_features).setOutputCol("features_s")
VA = VectorAssembler(inputCols=df_x.columns, outputCol='features') 
log_reg = LogisticRegression(featuresCol='features', labelCol='y_index') 
# Creating the pipeline 
pipex = Pipeline(stages=[assembler2, scaler, VA, log_reg])
# Fitting the model on training data 
fit_model = pipex.fit(train_data)
fit_model.write().overwrite().save('logreg.pkl')
# Running the model on test data
results = fit_model.transform(test_data) 
# Evaluating the model on AUC score
res = BinaryClassificationEvaluator(labelCol="y_index", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
ROC_AUC = res.evaluate(results)
print('ROC-AUC for logistic Regression:',ROC_AUC)

#Random Forest depth 5
rf_classifier = RandomForestClassifier(featuresCol="features", labelCol="y_index", numTrees=100, maxDepth=5)
# Creating the pipeline 
pipe2 = Pipeline(stages=[VA, rf_classifier])
# Fitting the model on training data 
fit_model = pipe2.fit(train_data)
fit_model.write().overwrite().save('RF1.pkl')
# Running the model on test data
results2 = fit_model.transform(test_data) 
# Evaluating the model on AUC score
res2 = BinaryClassificationEvaluator(labelCol="y_index", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
ROC_AUC = res2.evaluate(results2)
print('ROC-AUC for Random Forest:',ROC_AUC)

#Support Vector Classifier
svc = LinearSVC(featuresCol="features", labelCol="y_index", maxIter=10, regParam=0.1)
# Creating the pipeline 
pipe4 = Pipeline(stages=[VA, svc])
# Fitting the model on training data 
fit_model = pipe4.fit(train_data)
fit_model.write().overwrite().save('Linear-SVC.pkl') 
# Running the model on test data
results4 = fit_model.transform(test_data) 
# Evaluating the model on AUC score
res4 = BinaryClassificationEvaluator(labelCol="y_index", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
ROC_AUC = res4.evaluate(results4)
print("ROC-AUC for Support Vector Classifier is:", ROC_AUC)

#Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(featuresCol="features", labelCol="y_index", maxDepth=6)
# Creating the pipeline 
pipe5 = Pipeline(stages=[VA, dt_classifier])
# Fitting the model on training data 
fit_model = pipe5.fit(train_data) 
fit_model.write().overwrite().save('Decision_Tree.pkl') 
# Running the model on test data
results5 = fit_model.transform(test_data) 
# Evaluating the model on AUC score
res5 = BinaryClassificationEvaluator(labelCol="y_index", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
ROC_AUC = res5.evaluate(results5)
print('ROC-AUC for Decision Tree is:',ROC_AUC)

#Gradient Boosted Tree
GBT_classifier =  GBTClassifier(featuresCol="features", labelCol="y_index", maxDepth=5, maxIter=25, stepSize=0.01)
# Creating the pipeline 
pipe3 = Pipeline(stages=[VA, GBT_classifier])
# Fitting the model on training data 
fit_model = pipe3.fit(train_data) 
fit_model.write().overwrite().save('GBT-Classifier,pkl')
# Running the model on test data
results3 = fit_model.transform(test_data) 
# Evaluating the model on AUC score
res3 = BinaryClassificationEvaluator(labelCol="y_index", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
ROC_AUC = res3.evaluate(results3)
print("ROC-AUC for Gradient Boosted Tree is:",ROC_AUC)

# Building the feature importance scores to identify the most important features for prediction
# Best model was Gradient Boosting Model which is the current fit_model after the last run
feature_importances = fit_model.stages[-1].featureImportances
value_dict = {}
for i, importance in enumerate(feature_importances):
    value_dict[train_data.columns[i]] = importance
feature_importance = dict(sorted(value_dict.items(), key=lambda item: item[1], reverse=True))

# Plotting feature importance scores as a bar chart 
plt.figure(figsize = (11,5))
bars = plt.bar(list(feature_importance.keys())[:10],list(feature_importance.values())[:10])
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('Top 10 Important Features')
for bar, importance in zip(bars, list(feature_importance.values())[:10]):
    plt.text(bar.get_x() + bar.get_width() / 2 - 0.15, bar.get_height() + 0.01, f'{importance:.2f}', ha='center', color='black', fontsize=8)
plt.tight_layout()
plt.show()


# K-Means Clustering
df_k = df_x.select('duration','nr_employed','month_index','euribor3m','job_index')
assemble = VectorAssembler(inputCols=df_k.columns, outputCol='features') 

# Calculating and printing Silhouette scores
sil_score = []
inertial = []
for x in range(3,10):
    k = x
    kmeans = KMeans(k=k, seed=12345)
    # Building the pipeline
    kpipeline = Pipeline(stages=[assemble, kmeans])
    # Fitting the k-means pipeline on the dataframe
    model = kpipeline.fit(df_k)
    centers = model.stages[-1].clusterCenters()
    predictions = model.transform(df_k)
    evaluator = ClusteringEvaluator()
    silhouette = evaluator.evaluate(predictions)
    print("Silhouette Score for {0} = {1}".format(k,silhouette))
    sil_score.append(silhouette)




