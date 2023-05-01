
from pyspark.sql import SparkSession
spark=SparkSession.builder.appName('my_app').getOrCreate()
uber=spark.read.csv('C:\\Users\\Asus\\Downloads\\uber-raw-data-aug14.csv',header=True)
uber.groupBy("Base").count().show()
from pyspark.sql.functions import countDistinct

# بررسی تمام ستون‌های دیتافریم
for col_name in uber.columns:
    # پیدا کردن تعداد مقادیر یکتا در هر ستون
    unique_count = uber.select(countDistinct(col_name)).collect()[0][0]
    # چاپ نام ستون و تعداد مقادیر یکتا در آن
    print(f"{col_name}: {unique_count}")
import pandas as pd

# Convert the PySpark DataFrame to a Pandas DataFrame
uber_data = uber.toPandas()

# Convert the date/time column to a datetime object and extract additional columns
uber_data['Date/Time'] = pd.to_datetime(uber_data['Date/Time'], format="%m/%d/%Y %H:%M:%S")
uber_data['DayOfWeek'] = uber_data['Date/Time'].dt.dayofweek
uber_data['DayName'] = uber_data['Date/Time'].dt.day_name()
uber_data['Day'] = uber_data['Date/Time'].dt.day
uber_data['HourOfDay'] = uber_data['Date/Time'].dt.hour

# Create a pivot table to aggregate the total number of journeys by day of week
uber_weekdays = uber_data.pivot_table(index=['DayOfWeek', 'DayName'],
                                      values='Base',
                                      aggfunc='count')

# Plot the pivot table as a bar chart
uber_weekdays.plot(kind='bar', figsize=(8, 6))
uber_monthdays = uber_data.pivot_table(index=['Day'],
                                  values='Base',
                                  aggfunc='count')
uber_hour = uber_data.pivot_table(index=['HourOfDay'],
                                  values='Base',
                                  aggfunc='count')
from pyspark.sql.functions import isnan, when, count, col

# ساخت یک دیکشنری خالی برای نگه داری تعداد مقادیر نا معتبر هر ستون
null_counts = {}

# بررسی تمام ستون‌های دیتافریم
for i in uber.columns:
    # شمارش تعداد مقادیر نا معتبر در هر ستون و ذخیره آن در دیکشنری
    null_counts[i] = uber.filter(col(i).isNull() | isnan(col(i))).count()

col_count = len(uber.columns)

from pyspark.ml.feature import StringIndexer

# تعریف لیستی از نام ستون‌های مورد نظر
columns_to_index = ["Date/Time",'Base','Lat','Lon']
# ایجاد یک شی از کلاس StringIndexer برای هر ستون و اعمال آن بر روی داده
for column in columns_to_index:
    indexer = StringIndexer(inputCol=column, outputCol=column+"_N")
    uber = indexer.fit(uber).transform(uber)

from pyspark.ml.feature import VectorAssembler, StandardScaler


z=uber.select('Lat','Lon')

train_data, test_data = z.randomSplit([0.8, 0.2], seed=123)



# create vector feature column
#R=['Lat_N', 'Lon_N']
assembler = VectorAssembler(inputCols=train_data, outputCol='features')
data = assembler.transform(uber).select('features')
scaler = StandardScaler(inputCol='features', outputCol='scaledFeatures', withStd=True, withMean=False)

# استفاده از شی ایجاد شده برای استاندارد کردن داده‌های ویژگی
scaledData = scaler.fit(data).transform(data)
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

# Define a range of k values to try
k_values = range(2, 11)

# Empty list to store WCSS values for different k values
wcss_values = []

# Loop through different k values and calculate WCSS for each k
for k in k_values:
    kmeans = KMeans().setK(k).setSeed(1)
    model = kmeans.fit(scaledData)
    predictions = model.transform(scaledData)
    evaluator = ClusteringEvaluator()
    wcss = evaluator.evaluate(predictions)
    wcss_values.append(wcss)
    print("For k = {}, WCSS = {}".format(k, wcss))
test_data.toPandas().to_csv('f.csv',index=False)
# ایجاد شی KMeans با k=7
kmeans = KMeans().setK(7).setSeed(123)

# آموزش مدل با داده‌های آموزش
model = kmeans.fit(train_data)

# پیش‌بینی برچسب خوشه‌ها برای داده‌های تست
predictions = model.transform(test_data)
import numpy as np
import matplotlib.pyplot as plt
cnt=predictions.groupBy("prediction").count()
fig = plt.figure(figsize = (10, 5))
wcss = model.summary.trainingCost
# Import KMeansModel
from pyspark.ml.clustering import KMeansModel

# Train KMeans model
kmeans = KMeans(k=k, seed=1)
model = kmeans.fit(train_data)

# Save model
model_path = r"C:/Users/Asus/Desktop/sparkcode"
model.save(model_path)
