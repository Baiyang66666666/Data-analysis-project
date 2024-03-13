from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
import matplotlib.pyplot as plt
import numpy as np
import matplotlib 
matplotlib.use('Agg')
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.functions import monotonically_increasing_id




spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Q4") \
        .config("spark.local.dir","/fastdata/acp22bq") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR") 
spark.conf.set("spark.sql.debug.maxToStringFields", 1000)



print('=======================Task A================')
# Load the data from the file into a PySpark DataFrame
#compared to original file, NIPS_1987.csv add "Name" as the first col name
data = spark.read.load('./Data/NIPS_1987.csv', format = 'csv', inferSchema = "true", header = "true").cache()
data = data.drop("Name")

data_id = data.select("*").withColumn("id", F.monotonically_increasing_id())

#Convert data to vector
def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1]), r[-1]]).toDF(['features', 'id'])
    
dfFeatureVec = transData(data_id).cache()

#standard data
scaler_data = StandardScaler(withMean = True, withStd = True, inputCol = 'features', outputCol = 'scaled_features')
scaler_model = scaler_data.fit(dfFeatureVec)
dfFeatureVec = scaler_model.transform(dfFeatureVec).drop('features').withColumnRenamed('scaled_features','features')

# Compute the top 2 principal components using PCA
pca = PCA(k=2, inputCol="features", outputCol="pca_features")
model = pca.fit(dfFeatureVec)
pca_feature = model.transform(dfFeatureVec)

pca_data = np.array([row.__getitem__('pca_features').toArray() for row in pca_feature.collect()])

# Extract the percentage of variance captured
covariance  = model.explainedVariance.toArray()
print('the percentage of variance', covariance)


print('=======================Task B================')

# Plot the papers in a scatter plot using the top 2 principal components
plt.scatter(pca_data[:,0], pca_data[:,1], c='b',label='data')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA on NIPS Papers')
plt.legend()
plt.show()
plt.savefig('/home/acp22bq/com6012/ScalableML/Output/Q4/Question4_B.png')
plt.clf()



from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.linalg import Vectors

# RDD PCA API
iris_rm = RowMatrix(dfFeatureVec.rdd.map(lambda x: Vectors.dense(x[1].tolist())))
rdd_pc = iris_rm.computePrincipalComponents(2)
print("RDD PC:")
print(rdd_pc)
projected = iris_rm.multiply(rdd_pc)
print("RDD PCA projected features")
#print(projected.rows.collect())

# RDD SVD API
svd = iris_rm.computeSVD(2,True)
# The singular values are stored in a local dense vector.
s = svd.s
# The V factor is a local dense matrix.
V = svd.V
print("RDD SVD for PCA")


#eigenvalues
evs=s*s
print('eigenvalues ', evs)
#the percentage of variance
evs/sum(evs)
print(' the percentage of variance', evs)


from pyspark.ml.linalg import Vectors

# create a vector with 20 elements
v = Vectors.dense(range(20))

# convert the vector to an array and slice the first 10 elements
v_array = v.toArray()
print(v_array[:10])













