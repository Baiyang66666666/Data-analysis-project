import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import numpy as np
from pyspark.sql.types import StringType
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
import re



spark = SparkSession.builder \
    .master("local[10]") \
    .appName("Q5") \
    .config("spark.driver.memory", "20g") \
    .config("spark.local.dir", "/fastdata/acp22bq") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")

import warnings
warnings.filterwarnings("ignore")


print('===================Task A========================')
#Load data
data = spark.read.csv('/data/acp22bq/ScalableML/Data/HIGGS.csv')
# Rename columns
features = ['label','lepton_pT','lepton_eta','lepton_phi', 'missing_energy_magnitude','missing_energy_phi', 'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b-tag', 'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_btag', 'jet_3_pt', 'jet_3_eta','jet_3_phi', 'jet_3_btag', 'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_btag', 'mjj', 'mjjj', 'mlv', 'mjlv', 'mbb', 'mwbb', 'mwwbb']
ncolumns = len(data.columns)
schemaNames = data.schema.names
#Reassign names to each column
for i in range(ncolumns):
    data = data.withColumnRenamed(schemaNames[i], features[i])
    

# get the names of string columns   
StrColumns = [x.name for x in data.schema.fields if x.dataType == StringType()]
# Convert string columns to double 
for c in StrColumns:
    data = data.withColumn(c, col(c).cast("double"))
    
# Balance classes   
#count the number of positive labels
count_pos = data.filter(data.label==1).count()
# count the number of negative labels
count_neg = data.filter(data.label==0).count()
# calculate the fraction of the minority class
minority_fraction = min(count_pos, count_neg) / float(data.count())
# define class weights based on the minority class
class_weights = {0: minority_fraction, 1: minority_fraction}
 # resample the data to balance the classes
data = data.sampleBy("label", fractions=class_weights, seed=12)

# Sample and split data
sampled_data = data.sample(False, 0.01, seed=12).cache()
(sam_train_subset, sam_test_subset) = sampled_data.randomSplit([0.7, 0.3], seed=12)

#Write the training and test sets to disk
sam_train_subset.write.mode("overwrite").parquet('./Data/Q1subset_training.parquet')
sam_test_subset.write.mode("overwrite").parquet('./Data/Q1subset_test.parquet')
#Load the training and test sets from disk
subset_train = spark.read.parquet('./Data/Q1subset_training.parquet')
subset_test = spark.read.parquet('./Data/Q1subset_test.parquet')


print('===================Random Forest=========================')
#merge all features to one col
assembler = VectorAssembler(inputCols = features[1:], outputCol = 'features') 
#Creating a Random Forest classifier 
RF = RandomForestClassifier(labelCol="label", featuresCol="features", maxDepth=10, impurity='entropy')
#create pipeline
RF_stages = [assembler, RF]
RF_pipeline = Pipeline(stages=RF_stages)

#create a parameter grid    
RF_paramGrid = ParamGridBuilder() \
    .addGrid(RF.maxDepth, [1, 5, 10]) \
    .addGrid(RF.maxBins, [10, 20, 50]) \
    .addGrid(RF.numTrees, [1, 5, 10]) \
    .build()
#cross validation    
RF_crossvalidation = CrossValidator(estimator=RF_pipeline,
                          estimatorParamMaps=RF_paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=5)
#Fitting the cross-validator to get best model
RF_cvModel = RF_crossvalidation.fit(subset_train)
#use best model to predict
RF_predictions = RF_cvModel.transform(subset_test)

#Creating an acc evaluator
Acc_evaluator = MulticlassClassificationEvaluator\
      (labelCol="label", predictionCol="prediction", metricName="accuracy")
#Creating an AUC evaluator              
Area_evaluator = BinaryClassificationEvaluator\
      (labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
      
# calculate accuracy of the predictions      
RF_accuracy = Acc_evaluator.evaluate(RF_predictions)
print("RF accuracy = %g " % RF_accuracy)
# calculate AUC of the predictions
RF_area = Area_evaluator.evaluate(RF_predictions)
print("RF area under the curve = %g " % RF_area)




print('===================Gradient Boosting=========================')
# Create a GBT classifier
GBT = GBTClassifier(maxIter=5, maxDepth=2, labelCol="label", seed=12,
    featuresCol="features", lossType='logistic')

# Define the stages in the GBT pipeline
GBT_stages = [assembler, GBT]
GBT_pipeline = Pipeline(stages=GBT_stages)

#create parameter grid    
GBT_paramGrid = ParamGridBuilder() \
    .addGrid(GBT.maxDepth, [1, 5, 10]) \
    .addGrid(GBT.maxIter, [10, 20, 30]) \
    .addGrid(GBT.stepSize, [0.1, 0.2, 0.05]) \
    .build()
#cross validation    
GBT_crossvalidation = CrossValidator(estimator=GBT_pipeline,
                          estimatorParamMaps=GBT_paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=5)
#train model with cross validation
GBT_cvModel = GBT_crossvalidation.fit(subset_train)
GBT_predictions = GBT_cvModel.transform(subset_test)

#calculate GBT acc
GBT_accuracy = Acc_evaluator.evaluate(GBT_predictions)
print("GBT accuracy = %g " % GBT_accuracy)
#calculate GBT AUC
GBT_area = Area_evaluator.evaluate(GBT_predictions)
print("GBT area under the curve = %g " % GBT_area)



print('===========================Task B============================')

#RF_hyper and GBT_hyper stores the best hyperparameters found by cross-validation for two kinds of models
RF_hyper = RF_cvModel.getEstimatorParamMaps()[np.argmax(RF_cvModel.avgMetrics)]
GBT_hyper = GBT_cvModel.getEstimatorParamMaps()[np.argmax(GBT_cvModel.avgMetrics)]

#get the dictionary contains one hyperparameter and its value
def get_param(hyperparams):
  hyper_list = []
  for i in range(len(hyperparams.items())):
      hyper_name = re.search("name='(.+?)'", str([x for x in hyperparams.items()][i])).group(1)
      hyper_value = [x for x in hyperparams.items()][i][1]
      hyper_list.append({hyper_name: hyper_value})
      
  return hyper_list
  
# Extract the best hyperparameters for the Random Forest model      
RF_params = get_param(RF_hyper)   
print('===================best param for RF========================')  
print(RF_params) 
  
# Extract the best hyperparameters for the Gradient Boosted Tree model
GBT_params = get_param(GBT_hyper)     
print('=====================best param for GBT======================') 
print(GBT_params)
     



#split train and test dataset
(sam_train, sam_test) = sampled_data.randomSplit([0.7, 0.3], 12)
#Writing data to disk
sam_train.write.mode("overwrite").parquet('./Data/Q1training.parquet')
sam_test.write.mode("overwrite").parquet('./Data/Q1test.parquet')

#read data in disk
train = spark.read.parquet('./Data/Q1training.parquet')
test = spark.read.parquet('./Data/Q1test.parquet')


# Training the models on the training data 
# Using the best params
RF_best = RF_cvModel.bestModel
RF_stage = [assembler, RF_best]
RF_pipe = Pipeline(stages=RF_stage)

RF_Model = RF_pipe.fit(train)
RF_predictions = RF_Model.transform(test)
#calculate acc and AUC
RF_accuracy = evaluator_acc.evaluate(RF_predictions)
RF_auc = evaluator_auc.evaluate(RF_predictions)

print("Random Forests accuracy = %g " % RF_accuracy)
print("Random Forests area under the curve = %g " % RF_auc)


# Training the models on the training data 
# Using the best params                  
GBT_best = GBT_cvModel.bestModel
GBT_stage = [assembler, GBT_best]
GBT_pipe = Pipeline(stages=GBT_stage)

GBT_Model = GBT_pipe.fit(train)
GBT_predictions = GBT_Model.transform(test)
#calculate acc and AUC
GBT_accuracy = evaluator_acc.evaluate(GBT_predictions)
GBT_auc = evaluator_auc.evaluate(GBT_predictions)

print("Gradient Boosted Trees accuracy = %g " % GBT_accuracy)
print("Gradient Boosted Trees area under the curve = %g " % GBT_auc)
print('===========================================') 







