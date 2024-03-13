# coding: utf-8
import pyspark
from pyspark.sql import SparkSession
import matplotlib 
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab! 
import matplotlib.pyplot as plt
from pyspark.sql.functions import desc
import pyspark.sql.functions as F
import pandas as pd
from pyspark.sql.functions import col, dayofmonth, hour, max
import numpy as np
from pyspark.sql.functions import to_timestamp



spark = SparkSession.builder \
    .master("local[2]") \
    .appName("Q1") \
    .config("spark.local.dir","/fastdata/acp22bq") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")  # This can only affect the log level after it is executed.

logFile=spark.read.text("/home/acp22bq/com6012/ScalableML/Data/NASA_access_log_Jul95.gz").cache()

# split into 5 columns using regex and split
data = logFile.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()
#data = data.withColumn("timestamp", to_timestamp("timestamp", "dd/MMM/yyyy:HH:mm:ss Z"))
data = data.na.drop().cache()
data.show(20,False)

hostsGermany = logFile.filter(logFile.value.contains(".de")).count()
hostsCanada = logFile.filter(logFile.value.contains(".ca")).count()
hostsSingapore = logFile.filter(logFile.value.contains(".sg")).count()

#Print the total number of requests
print(f"The total number of requests for Germany is {hostsGermany}")
print(f"The total number of requests for Canada is {hostsCanada}")
print(f"The total number of requests for Singapor is {hostsSingapore}")

#plot the Bar chart
countries = ['Germany', 'Canada', 'Singapore']
counts = [hostsGermany, hostsCanada, hostsSingapore]
plt.bar(countries, counts, color=['red', 'blue', 'green'])

#add title and label
plt.title('Number of Hosts by Country')
plt.xlabel('Country')
plt.ylabel('Number of Hosts')

plt.show()
plt.savefig('/home/acp22bq/com6012/ScalableML/Output/Question1_taskA.png', dpi=200, bbox_inches="tight")
plt.close()






# number of unique hosts of three countries
uniqueHostsGermany = data.filter(data.host.like('%.de')).agg(F.countDistinct('host').alias('unique_hosts')).collect()[0]['unique_hosts']
uniqueHostsCanada = data.filter(data.host.like('%.ca')).agg(F.countDistinct('host').alias('unique_hosts')).collect()[0]['unique_hosts']
uniqueHostsSingapore = data.filter(data.host.like('%.sg')).agg(F.countDistinct('host').alias('unique_hosts')).collect()[0]['unique_hosts']

print("==================== Task B ====================")
print(f"There are {uniqueHostsGermany} Germany unique hosts")
print(f"There are {uniqueHostsCanada} Canada unique hosts")
print(f"There are {uniqueHostsSingapore} Singapore unique hosts")

# Calculate top 9 hosts for each country
topHostsGermany = data.filter(data.host.like('%.de')).groupBy('host').count().orderBy(desc('count')).limit(9).collect()
topHostsCanada = data.filter(data.host.like('%.ca')).groupBy('host').count().orderBy(desc('count')).limit(9).collect()
topHostsSingapore = data.filter(data.host.like('%.sg')).groupBy('host').count().orderBy(desc('count')).limit(9).collect()

# Print results
print("====================================================")
print("Top 9 hosts in Germany:")
for row in topHostsGermany:
    print(row['host'], row['count'])
print("====================================================")
print("Top 9 hosts in Canada:")
for row in topHostsCanada:
    print(row['host'], row['count'])
print("====================================================")
print("Top 9 hosts in Singapore:")
for row in topHostsSingapore:
    print(row['host'], row['count'])
print("====================================================")


print("==================== Task C ====================")
# create a list of top 9 hosts and their counts
topHostsGermanyList = [(row['host'], row['count']) for row in topHostsGermany]
topHostsCanadaList = [(row['host'], row['count']) for row in topHostsCanada]
topHostsSingaporeList = [(row['host'], row['count']) for row in topHostsSingapore]
# calculate the count for the rest of the hosts
restCountGermany = data.filter(data.host.like('%.de')).count() - sum([count for _, count in topHostsGermanyList])
restCountCanada = data.filter(data.host.like('%.de')).count() - sum([count for _, count in topHostsCanadaList])
restCountSingapore = data.filter(data.host.like('%.de')).count() - sum([count for _, count in topHostsSingaporeList])
# append the rest count to the list
topHostsGermanyList.append(('rest', restCountGermany))
topHostsCanadaList.append(('rest', restCountCanada))
topHostsSingaporeList.append(('rest', restCountSingapore))
# calculate the percentage for each host
percentagesGermany = [count / sum([count for _, count in topHostsGermanyList]) for _, count in topHostsGermanyList]
percentagesCanada = [count / sum([count for _, count in topHostsCanadaList]) for _, count in topHostsCanadaList]
percentagesSingapore = [count / sum([count for _, count in topHostsSingaporeList]) for _, count in topHostsSingaporeList]
# create a list of labels for the chart
labelsGermany = [host for host, _ in topHostsGermanyList]
labelsCanada = [host for host, _ in topHostsCanadaList]
labelsSingapore = [host for host, _ in topHostsSingaporeList]
# create a list of colors for the chart
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

# create the pie chart for Germany
plt.figure(figsize=(15,15))
plt.pie(percentagesGermany, labels=labelsGermany, colors=colors, autopct='%1.1f%%',startangle=90,labeldistance=1.1, textprops={'fontsize': 20})
# add a title
plt.title('Percentage of Requests by Host for Germany', fontsize=30)
plt.legend(title="Legend", loc="best", fontsize=15)
# show the chart
plt.show()
# save the chart to a file
plt.savefig('/home/acp22bq/com6012/ScalableML/Output/Question1_taskC_Germany.png', dpi=200, bbox_inches="tight")
# clear the chart
plt.close()

# create the pie chart for Canada
plt.pie(percentagesCanada, labels=labelsCanada, colors=colors, autopct='%1.1f%%',startangle=90)
# add a title
plt.title('Percentage of Requests by Host for Canada')
#plt.legend(title="Legend", loc="best")
# show the chart
plt.show()
# save the chart to a file
plt.savefig('/home/acp22bq/com6012/ScalableML/Output/Question1_taskC_Canada.png', dpi=200, bbox_inches="tight")
# clear the chart
plt.close()

# create the pie chart for Singapore
plt.figure(figsize=(45,45))
#plt.pie(percentagesSingapore, labels=labelsSingapore, colors=colors, autopct='%1.1f%%',labeldistance=2.1, textprops={'fontsize': 30})
plt.pie(percentagesSingapore, colors=colors)
# create the legend with labels and percentages
legend_labels = [f"{label}: {percent*100:.1f}%" for label, percent in zip(labelsSingapore, percentagesSingapore)]
plt.legend(legend_labels, title="Legend", loc="best", fontsize=30)

# add a title
plt.title('Percentage of Requests by Host for Singapore',fontsize=45)
#plt.legend(title="Legend", loc="best", fontsize=30)
# show the chart
plt.show()
# save the chart to a file
plt.savefig('/home/acp22bq/com6012/ScalableML/Output/Question1_taskC_Singapore.png', dpi=200, bbox_inches="tight")
# clear the chart
plt.close()



print("==================== Task D ====================")
# Define the countries we're interested in
countries = ['Germany', 'Canada', 'Singapore']

# Define a function to extract the day and hour from the timestamp
def extract_day_hour(timestamp):
    day = timestamp.split(':')[0][0:2]
    hour = timestamp.split(':')[1]
    return (int(hour), int(day))

# Define a function to plot the heatmap
def plot_heatmap(data, title):
    # Create a 2D array with the number of visits for each hour of each day
    heatmap_data = np.zeros((24, 31))
    for row in data:
        day, hour, count = row
        heatmap_data[hour, day-1] = count

    # Create the plot
    fig, ax = plt.subplots(figsize=(10,5))
    ax.set_title(title)
    ax.set_ylabel('Hour')
    ax.set_xlabel('Day')
    ax.set_yticks(np.arange(0, 24))
    ax.set_xticks(np.arange(1, 32))
    ax.set_yticklabels(np.arange(0, 24))
    ax.set_xticklabels(np.arange(1, 32))
    
    heatmap = ax.pcolor(heatmap_data, cmap=plt.cm.Blues, edgecolors='k', linewidths=0.5)
    colorbar = plt.colorbar(heatmap)
    colorbar.set_label('Number of Visits')
    plt.show()
    plt.savefig('/home/acp22bq/com6012/ScalableML/Output/Question1_D_{}.png'.format(title), dpi=200, bbox_inches="tight")
    plt.close()
    
# Filter the data for the top host from each country
topHostsGermany = data.filter(data.host.like('%.de')).groupBy('host').count().orderBy(desc('count')).limit(1).collect()[0].host
topHostsCanada = data.filter(data.host.like('%.ca')).groupBy('host').count().orderBy(desc('count')).limit(1).collect()[0].host
topHostsSingapore = data.filter(data.host.like('%.sg')).groupBy('host').count().orderBy(desc('count')).limit(1).collect()[0].host


# Extract the day and hour of the visit from the timestamp
data_G = data.select('timestamp').filter((data.host == topHostsGermany) )
data_C = data.select('timestamp').filter((data.host == topHostsCanada))
data_S = data.select('timestamp').filter((data.host == topHostsSingapore))
data_G = data_G.rdd.map(lambda x: extract_day_hour(x.timestamp)).toDF(['hour', 'day'])
data_C = data_C.rdd.map(lambda x: extract_day_hour(x.timestamp)).toDF(['hour', 'day'])
data_S = data_S.rdd.map(lambda x: extract_day_hour(x.timestamp)).toDF(['hour', 'day'])

# Group the data by day and hour and count the number of visits
data_G = data_G.groupBy(['day', 'hour']).count().orderBy(['day', 'hour'])
data_C = data_C.groupBy(['day', 'hour']).count().orderBy(['day', 'hour'])
data_S = data_S.groupBy(['day', 'hour']).count().orderBy(['day', 'hour'])
data_C.orderBy(desc('day'), desc('hour')).show(5)
data_C.orderBy(('day'), ('hour')).show(5)
print('Germany')
data_G.orderBy(desc('day'), desc('hour')).show(5)
data_G.orderBy(('day'), ('hour')).show(5)
print('sin')
data_S.orderBy(('day'), ('hour')).show(5)
data_S.orderBy(desc('day'), desc('hour')).show(5)
# Plot the heatmap for each country
data_germany = data_G.filter(data_G.day <= 31)
plot_heatmap(data_germany.collect(), 'Germany')
data_canada = data_C.filter(data_C.day <= 31)
plot_heatmap(data_canada.collect(), 'Canada')
data_singapore = data_S.filter(data_S.day <= 31)
plot_heatmap(data_singapore.collect(), 'Singapore')




spark.stop()