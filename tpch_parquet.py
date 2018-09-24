#./spark/bin/spark-submit tpch_parquet.py

from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import *
from datetime import datetime

def parse_dt(s):
    return datetime.strptime(s, '%Y-%m-%d')

def parse(line):
    items = line.split("|")
    return (int(items[0]), int(items[1]), int(items[2]), int(items[3]), float(items[4]), float(items[5]), float(items[6]), float(items[7]), items[8], items[9], parse_dt(items[10]), parse_dt(items[11]), parse_dt(items[12]), items[13], items[14], items[15])

if __name__ == "__main__":
  sc = SparkContext(appName="CSV2Parquet")
  sqlContext = SQLContext(sc)

  schema = StructType([
    StructField("l_orderkey", IntegerType(), True),
    StructField("l_partkey", IntegerType(), True),
    StructField("l_suppkey", IntegerType(), True),
    StructField("l_linenumber", IntegerType(), True),
    StructField("l_quantity", DoubleType(), True),
    StructField("l_extendedprice", DoubleType(), True),
    StructField("l_discount", DoubleType(), True),
    StructField("l_tax", DoubleType(), True),
    StructField("l_returnflag", StringType(), True),
    StructField("l_linestatus", StringType(), True),
    StructField("l_shipdate", DateType(), True),
    StructField("l_commitdate", DateType(), True),
    StructField("l_receiptdate", DateType(), True),
    StructField("l_shipinstruct", StringType(), True),
    StructField("l_shipmode", StringType(), True),
    StructField("l_comment", StringType(), True)
 ])

sc.setCheckpointDir('home/ewei/checkpoints')
rdd = sc.textFile("/home/ewei/lineitem.csv", use_unicode=False)
rdd.checkpoint()
rdd2 = rdd.map(parse)
df = sqlContext.createDataFrame(rdd2, schema)
df.write.parquet('/home/ewei/lineitem-parquet')
