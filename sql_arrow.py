from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SQLContext
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import time

if __name__ == "__main__":
    sc_conf = SparkConf()
    sc_conf.set("spark.sql.execution.arrow.enabled", "true")

    sc = SparkContext(appName="BenchmarkParquet", conf=sc_conf)
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

    df = sqlContext.read.option("header", "false").schema(schema).csv('/home/ewei/lineitem.csv')
    df.printSchema()
    df.registerTempTable("lineitem")

    start = time.time()
    df = sqlContext.sql("SELECT l_returnflag, l_linestatus, sum(l_quantity) as sum_qty, sum(l_extendedprice) as sum_base_price, sum(l_extendedprice * (1 - l_discount)) as sum_disc_price, sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge, avg(l_quantity) as avg_qty, avg(l_extendedprice) as avg_price, avg(l_discount) as avg_disc, count(*) as count_order FROM lineitem WHERE l_shipdate <= date '1998-12-01' - interval '90' day GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus")
    df.show(n = 5)
    end = time.time()
    print(end - start)
