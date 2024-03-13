import matplotlib 
matplotlib.use('Agg')
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.window import Window
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import avg



spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Q3") \
        .config("spark.local.dir","/fastdata/acp22bq") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN") 



print('=======================Task A================')
#load data
origin_datas = spark.read.load('./Data/Q3/ratings.csv', format = 'csv', inferSchema = "true", header = "true").cache()
#origin_datas.show(20,False)

#sort all data by the timestamp
data_sorted = origin_datas.orderBy('timestamp', ascending=True).cache()

#  Sorting the dataframe in ascending order of timestamp
windowSpec  = Window.orderBy(origin_datas['timestamp'].asc())
# Then getting the percentage ranks to split the data
data_ranks = origin_datas.withColumn("percent_rank", F.percent_rank().over(windowSpec))
data_ranks.show(20, False)

# Splitting the data into the three splits, training data sizes: 40%, 60%, and 80%.
train1 = data_ranks.filter(data_ranks["percent_rank"]<0.4).cache()
train1.count()
test1 = data_ranks.filter(data_ranks["percent_rank"]>=0.4).cache()
test1.count()

train2 = data_ranks.filter(data_ranks["percent_rank"]<0.6).cache()
train2.count()
test2 = data_ranks.filter(data_ranks["percent_rank"]>=0.6).cache()
test2.count()

train3 = data_ranks.filter(data_ranks["percent_rank"]<0.8).cache()
train3.count()
test3 = data_ranks.filter(data_ranks["percent_rank"]>=0.8).cache()
test3.count()

#My seeds is220186812, and the setting used in lab5 is as follow
#als = ALS(userCol="userId", itemCol="movieId", seed=6012, coldStartStrategy="drop")
als1 = ALS(userCol="userId", itemCol="movieId", seed=220186812, coldStartStrategy="drop")
#calculate model rmse, mse, mae
rmse_evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
mse_evaluator = RegressionEvaluator(metricName="mse", labelCol="rating",predictionCol="prediction")
mae_evaluator = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction")


metrics = []

def run_model(_train, _test, _als, rmse_evaluator, mse_evaluator, mae_evaluator):
    #The function fits the ALS algorithm to the training data, makes predictions on the test data
    metrics = []
    model = _als.fit(_train)
    predictions = model.transform(_test)
    #calculates three metrics: RMSE, MSE, and MAE.
    rmse = rmse_evaluator.evaluate(predictions)
    mse = mse_evaluator.evaluate(predictions)
    mae = mae_evaluator.evaluate(predictions)
    
    metrics.append(rmse)
    metrics.append(mse)
    metrics.append(mae)

    return metrics

# Carrying out the als and evaluating the performance metrics using three different als settings
output1 = run_model(train1, test1, als1, rmse_evaluator, mse_evaluator, mae_evaluator)
output2 = run_model(train2, test2, als1, rmse_evaluator, mse_evaluator, mae_evaluator)
output3 = run_model(train3, test3, als1, rmse_evaluator, mse_evaluator, mae_evaluator)

#The rank of the ALS model determines the number of latent factors used to represent each user and item in the smaller matrices. By increasing the rank, we allow the model to capture more complex patterns and interactions between users and items, which can lead to better quality recommendations.
als2 = als1.setRank(15)
#The function fits the ALS algorithm to the training data, makes predictions on the test data
output4 = run_model(train1, test1, als2, rmse_evaluator, mse_evaluator, mae_evaluator)
output5 = run_model(train2, test2, als2, rmse_evaluator, mse_evaluator, mae_evaluator)
output6 = run_model(train3, test3, als2, rmse_evaluator, mse_evaluator, mae_evaluator)

print('als1 metrics')
print(output1, output2, output3)
print('als2 metrics')
print(output4, output5, output6)


#model based on Als setting2
model1 = als2.fit(train1)
predictions1 = model1.transform(test1)
model2 = als2.fit(train2)
predictions2 = model2.transform(test2)
model3 = als2.fit(train3)
predictions3 = model3.transform(test3)


#Task B     User Analysis
print('=======================Task B================')
# Getting the user factors vector from the two als models
def featuresvec(model):
  #Collecting the user factors vector from the ALS model
  User_factors = model.userFactors.collect()
  #Creating a dataframe from the collected user factors vector
  df = spark.createDataFrame(User_factors, ["userid", "features"])
  #Separating the userid and features columns into two separate RDDs
  id_rdd = df.rdd.map(lambda row:row[0])
  features_rdd = df.rdd.map(lambda row:row[1])
  #Combining the id and feature RDDs
  new_df = id_rdd.zip(features_rdd.map(lambda x:Vectors.dense(x))).toDF(schema=['userid','features'])
  return new_df

#Calling the function to get the user features vectors for each ALS model
dfFeatureVec1 = featuresvec(model1)
dfFeatureVec2 = featuresvec(model2)
dfFeatureVec3 = featuresvec(model3)

#Instantiating the KMeans clustering model with k=25 and seed=220186812(my reg num)
kmeans = KMeans(k=25, seed=220186812)

#Using kmeans on the two als settings 
kmeans1 = kmeans.fit(dfFeatureVec1.select('features'))
#Transforming the user feature vectors into k-means clusters
kpreds1_tr = kmeans1.transform(dfFeatureVec1.select('features'))
#Joining the original user feature vectors with the k-means cluster predictions
preds_with_ids1 = dfFeatureVec1.join(kpreds1_tr, ["features"], "leftouter").cache()

#Extracting cluster size information and printing the top 5 largest clusters
summary1 = kmeans1.summary
Cluster_sizes1 = summary1.clusterSizes
Cluster_sizes1sort = sorted(summary1.clusterSizes, reverse=True)
print('=========this is top 5 size clusters in split1====================')
print(Cluster_sizes1sort[0:5])
#find the largest size cluster
largest_cluster1 = Cluster_sizes1.index(max(Cluster_sizes1))

#Filtering out users belonging to the largest clusters
largest_cluster_ids1 =  preds_with_ids1.filter(preds_with_ids1["prediction"]==largest_cluster1).cache()
print('=========this is largest_cluster_ids1====================')
largest_cluster_ids1.show(5)

# Get user IDs in largest cluster 1
user_ids_largest_cluster1 = [row['userId'] for row in largest_cluster_ids1.select('userId').distinct().collect()]
print('=========this is user_ids_largest_cluster1====================')
print(user_ids_largest_cluster1[:10])

#Using kmeans on the two als settings
kmeans2 = kmeans.fit(dfFeatureVec2.select('features'))
kpreds2_tr = kmeans2.transform(dfFeatureVec2.select('features'))
#Joining the original user feature vectors with the k-means cluster predictions
preds_with_ids2 = dfFeatureVec2.join(kpreds2_tr, ["features"], "leftouter").cache()
#Extracting cluster size information and printing the top 5 largest clusters
summary2 = kmeans2.summary
Cluster_sizes2 = summary2.clusterSizes
Cluster_sizes2sort = sorted(summary2.clusterSizes, reverse=True)
print('=========this is top 5 size clusters in split2====================')
print(Cluster_sizes2sort[0:5])
#find the largest size cluster
largest_cluster2 = Cluster_sizes2.index(max(Cluster_sizes2))
#Filtering out users belonging to the largest clusters
largest_cluster_ids2 =  preds_with_ids2.filter(preds_with_ids2["prediction"]==largest_cluster2).cache()



#same process with last two kmean approaches
kmeans3 = kmeans.fit(dfFeatureVec3.select('features'))
kpreds3 = kmeans3.transform(dfFeatureVec3.select('features'))
preds_with_ids3 = dfFeatureVec3.join(kpreds3, ["features"], "leftouter").cache()
summary3 = kmeans3.summary
Cluster_sizes3 = summary3.clusterSizes
Cluster_sizes3sort = sorted(summary3.clusterSizes, reverse=True)
print('=========this is top 5 size clusters in split3====================')
print(Cluster_sizes3sort[0:5])
largest_cluster3 = Cluster_sizes3.index(max(Cluster_sizes3))
largest_cluster_ids3 =  preds_with_ids3.filter(preds_with_ids3["prediction"]==largest_cluster3).cache()



print('===========Task B 2)===============')
movie_data = spark.read.load('./Data/Q3/movies.csv', format = 'csv', inferSchema = "true", header = "true").cache()

# Filtering the ratings for the train set and test set
# Then selecting only the movieids in the clusters

#Extracting user ids belonging to the largest cluster
users_ids1 = largest_cluster_ids1.select("userid").rdd.flatMap(lambda x: x).collect()
#Filtering out the movies watched by users in the largest cluster with rating 4 or above
tr_ratings1 = train1.filter(train1.rating>=4).cache()
movies_largest_cluster1 = tr_ratings1.filter(F.col('userId').isin(users_ids1)).cache()
#Finding the top movies that were watched by users in the largest cluster
top_movies1 = movie_data.join(movies_largest_cluster1, movie_data.movieId == movies_largest_cluster1.movieId, "leftanti")
#Splitting and counting the genres of the top movies
tr_genre_split1 = top_movies1.select("title", F.split("genres", "\|").alias("genres")) \
    .withColumn("genre", F.explode("genres")) \
    .select("title", "genre")
#group movies by genres and counting them to get the top 10
tr_genre_counts1 = tr_genre_split1.groupBy("genre").count().sort(F.col("count").desc())
tr_top10_genre1 = tr_genre_counts1.take(10)
print("The top 10 movie genres for train split 1",tr_top10_genre1)


#same process as the first split
users_ids2 = largest_cluster_ids2.select("userid").rdd.flatMap(lambda x: x).collect()
tr_ratings2 = train2.filter(train2.rating>=4).cache()
movies_largest_cluster2 = tr_ratings2.filter(F.col('userId').isin(users_ids2)).cache()
top_movies2 = movie_data.join(movies_largest_cluster2, movie_data.movieId == movies_largest_cluster2.movieId, "leftanti")
tr_genre_split2 = top_movies2.select("title", F.split("genres", "\|").alias("genres")) \
    .withColumn("genre", F.explode("genres")) \
    .select("title", "genre")
tr_genre_counts2 = tr_genre_split2.groupBy("genre").count().sort(F.col("count").desc())
tr_top10_genre2 = tr_genre_counts2.take(10)
print("The top 10 movie genres for train split 2",tr_top10_genre2)



#same process as the first split
users_ids3 = largest_cluster_ids3.select("userid").rdd.flatMap(lambda x: x).collect()
tr_ratings3 = train3.filter(train3.rating>=4).cache()
movies_largest_cluster3 = tr_ratings3.filter(F.col('userId').isin(users_ids3)).cache()
top_movies3 = movie_data.join(movies_largest_cluster3, movie_data.movieId == movies_largest_cluster3.movieId, "leftanti")
tr_genre_split3 = top_movies3.select("title", F.split("genres", "\|").alias("genres")) \
    .withColumn("genre", F.explode("genres")) \
    .select("title", "genre")
tr_genre_counts3 = tr_genre_split3.groupBy("genre").count().sort(F.col("count").desc())
tr_top10_genre3 = tr_genre_counts3.take(10)
print("The top 10 movie genres for train split 3",tr_top10_genre3)



spark.stop()
























