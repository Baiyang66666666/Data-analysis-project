# coding: utf-8
import pyspark
from pyspark.sql import SparkSession
import matplotlib 
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab! 
import matplotlib.pyplot as plt
import pyspark.sql.functions as F
import pandas as pd
from pyspark.sql.functions import col, dayofmonth, hour, max, log, when,lit
import numpy as np
from pyspark.ml.feature import VectorAssembler, OneHotEncoder, StandardScaler, StringIndexer
from pyspark.ml.regression import GeneralizedLinearRegression, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import LogisticRegression


print("==================== Task A ====================")
spark = SparkSession.builder \
    .master("local[4]") \
    .appName("Q2") \
    .config("spark.local.dir","/fastdata/acp22bq") \
    .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN")  # This can only affect the log level after it is executed.

#load the data    
data = spark.read.csv('./Data/freMTPL2freq_full.csv', header=True)
data.cache()
data.show(35,False)

#process data
schemaNames = data.schema.names
ncolumns = len(data.columns)

data = data.withColumn(schemaNames[3],data[schemaNames[3]])
data = data.withColumn(schemaNames[8],data[schemaNames[8]])
data = data.withColumn(schemaNames[9],data[schemaNames[9]])
data = data.withColumn(schemaNames[11],data[schemaNames[11]])
data = data.withColumn(schemaNames[2],data[schemaNames[2]].cast(DoubleType()))
data = data.withColumn(schemaNames[4],data[schemaNames[4]].cast(DoubleType()))
data = data.withColumn(schemaNames[5],data[schemaNames[5]].cast(DoubleType()))
data = data.withColumn(schemaNames[6],data[schemaNames[6]].cast(DoubleType()))
data = data.withColumn(schemaNames[7],data[schemaNames[7]].cast(DoubleType()))
data = data.withColumn(schemaNames[10],data[schemaNames[10]].cast(DoubleType()))
data = data.withColumn(schemaNames[0],data[schemaNames[0]].cast(DoubleType()))
data = data.withColumn(schemaNames[1],data[schemaNames[1]].cast(DoubleType()))


#create and add new columns, and preprocess ClaimNb for LogClaimNb
data = data.withColumn("LogClaimNb", log(col("ClaimNb")+1))
data = data.withColumn("NZClaim", when(col("ClaimNb") > 0, 1).otherwise(0))





print("==================== Task B ====================")

#Standardize numeric features
numericCols = ['Exposure','VehPower', 'VehAge', 'DrivAge', 'BonusMalus', 'Density']
assembler = VectorAssembler(inputCols=numericCols, outputCol="numeric_features")
scaler = StandardScaler(inputCol="numeric_features", outputCol="scaled_numeric_features", withStd=True, withMean=False)

#One-hot encode the categorical columns
# Define the categorical columns
categoricalCols = ['Area', 'VehBrand', 'VehGas', 'Region']
# Convert the categorical columns to indexed columns
indexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c)) for c in categoricalCols]
# One-hot encode the indexed columns
encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol="{0}_ohe".format(indexer.getOutputCol())) for indexer in indexers]

#Create a pipeline with all the above stages
pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])

#Fit the pipeline on the input data and transform the input data using the pipeline
#drop the original non-encoded columns from our dataset as they are no longer required
model = pipeline.fit(data)
processedData = model.transform(data).drop(*['Area', 'VehBrand', 'VehGas', 'Region','Area_indexed','VehBrand_indexed','VehGas_indexed','Region_indexed','Exposure','VehPower','VehAge','DrivAge',
'BonusMalus','density','numeric_features'])  
processedData.show(35)
processedData.select([col('scaled_numeric_features'), col('Area_indexed_ohe'), col('VehBrand_indexed_ohe'), col('VehGas_indexed_ohe'), col('Region_indexed_ohe')]).show(10, False)
processedData.printSchema()

# Now, we select the feature we wish to use(alongside the OHE versions )
feature_cols = processedData.schema.names[4:9]
assembler = VectorAssembler(inputCols = feature_cols, outputCol = 'features') 

#Split the dataset into training (70%) and test (30%) sets
#calculate the fractions for stratified sampling
fractions = processedData.select("ClaimNb").distinct().withColumn("fraction", lit(0.7)).rdd.collectAsMap()

# perform the stratified split
trainingData = processedData.sampleBy("ClaimNb", fractions, seed=12)
testData = processedData.subtract(trainingData)
print('this is training data')
trainingData.show(10)





# Define the Poisson model
glm_poisson = GeneralizedLinearRegression(featuresCol='features', labelCol='ClaimNb', maxIter=50, regParam=0.01,\
                                          family='poisson', link='log')
# Construct and fit pipeline to data
stages = [assembler, glm_poisson]
pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(trainingData)
# Evaluate the model RMSE on the test set
predictions = pipelineModel.transform(testData)
evaluator = RegressionEvaluator(labelCol="ClaimNb", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("\nPoisson RMSE = %g \n" % rmse)
print('\n Poisson Model Coefficients')
print(pipelineModel.stages[-1].coefficients)


#input: training data,pipeline(model(lr), assembler, scaler) ,evaluator
def valcur_plot(trainingData, pipeline, lr, evaluator,title):
    
    # Define a subset of the training set for cross-validation
    subsetTrainingData = trainingData.sample(fraction=0.1, seed=12)
    
    # Define the grid of hyperparameters to search over
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.001, 0.01, 0.1, 1, 10]) \
        .build()

    # Define the cross-validation object
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

    # Fit the cross-validation model
    cvModel = cv.fit(subsetTrainingData)
    cvModel2 = cv.fit(trainingData)
    # Extract the results of cross-validation
    results = cvModel.avgMetrics
    results2 = cvModel2.avgMetrics
    print('size of trainscore')
    print(results)
    
    # Plot the validation curves
    plt.plot([0.001, 0.01, 0.1, 1, 10], results, label='val')
    plt.plot([0.001, 0.01, 0.1, 1, 10], results2, label='Train')
    plt.xscale('log')
    plt.xlabel('regParam')
    plt.ylabel('score')
    plt.legend(loc='best')
    plt.title('Validation Curve {}.png'.format(title))
    plt.savefig('/home/acp22bq/com6012/ScalableML/Output/Q2/Question2_B(c)_{}.png'.format(title))
    plt.show()
    plt.clf()

valcur_plot(trainingData, pipeline, glm_poisson,evaluator, 'Poisson')





# Define the linear regression models
a_lr_l1 = LinearRegression(featuresCol='features', labelCol='LogClaimNb', elasticNetParam=1.0, maxIter=50, regParam=0.01)
a_lr_l2 = LinearRegression(featuresCol='features', labelCol='LogClaimNb', elasticNetParam=0.0, maxIter=50, regParam=0.01)

# Construct pipelines for each model
a_stages_l1 = [assembler, a_lr_l1]
a_pipeline_l1 = Pipeline(stages=a_stages_l1)
a_stages_l2 = [assembler, a_lr_l2]
a_pipeline_l2 = Pipeline(stages=a_stages_l2)

# Fit the pipelines to training data
a_model_l1 = a_pipeline_l1.fit(trainingData)
a_model_l2 = a_pipeline_l2.fit(trainingData)

# Make predictions on test data
a_predictions_l1 = a_model_l1.transform(testData)
a_predictions_l2 = a_model_l2.transform(testData)

# Evaluate the models
a_evaluator = RegressionEvaluator(labelCol='LogClaimNb', predictionCol='prediction', metricName='rmse')
a_rmse_l1 = a_evaluator.evaluate(a_predictions_l1)
a_rmse_l2 = a_evaluator.evaluate(a_predictions_l2)

# Print the results
print('Linear Regression L1 Regularization: RMSE = %g' % a_rmse_l1)
print('Linear Regression L1 Regularization: Model Coefficients = %s' % str(a_model_l1.stages[-1].coefficients))
print('Linear Regression L2 Regularization: RMSE = %g' % a_rmse_l2)
print('Linear Regression L2 Regularization: Model Coefficients = %s' % str(a_model_l2.stages[-1].coefficients))

valcur_plot(trainingData, a_pipeline_l1, a_lr_l1, a_evaluator, 'LinearRegression_l1')
valcur_plot(trainingData, a_pipeline_l2, a_lr_l2, a_evaluator, 'LinearRegression_l2')






# Define the Logistic Regression model with L1 and L2 regularization
b_lr_l1 = LogisticRegression(featuresCol="features", labelCol="NZClaim", maxIter=50, regParam=0.1, elasticNetParam=1)
b_lr_l2 = LogisticRegression(featuresCol="features", labelCol="NZClaim", maxIter=50, regParam=0.1, elasticNetParam=0)

# Define the pipeline for the L1 and L2 model
b_pipeline_l1 = Pipeline(stages=[assembler, b_lr_l1])
b_pipeline_l2 = Pipeline(stages=[assembler, b_lr_l2])

# Fit the L1 model and make predictions on the test set
b_model_l1 = b_pipeline_l1.fit(trainingData)
b_predictions_l1 = b_model_l1.transform(testData)

# Fit the L2 model and make predictions on the test set
b_model_l2 = b_pipeline_l2.fit(trainingData)
b_predictions_l2 = b_model_l2.transform(testData)

# Evaluate the models
b_evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="NZClaim", metricName="areaUnderROC")

# Compute the AUC for the L1 model
auc_l1 = b_evaluator.evaluate(b_predictions_l1)
print("Logisti cRegression AUC (L1 Regularization) = {:.2f}%".format(auc_l1 * 100))

# Compute the AUC for the L2 model
auc_l2 = b_evaluator.evaluate(b_predictions_l2)
print("Logisti cRegression AUC (L2 Regularization) = {:.2f}%".format(auc_l2 * 100))

# Print the model coefficients for the L1 model
print("Logisti cRegression Model Coefficients (L1 Regularization):")
print(b_model_l1.stages[-1].coefficients)

# Print the model coefficients for the L2 model
print("Logisti cRegression Model Coefficients (L2 Regularization):")
print(b_model_l2.stages[-1].coefficients)

valcur_plot(trainingData, b_pipeline_l1, b_lr_l1, b_evaluator,'LogisticRegression_l1')
valcur_plot(trainingData, b_pipeline_l2, b_lr_l2,b_evaluator, 'LogisticRegression_l2')

