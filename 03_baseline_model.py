#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Template script to connect to Active Spark Session
Usage:
    $ spark-submit --deploy-mode client baseline_model.py <any arguments you wish to add>
'''

import os
import sys
import getpass
import hashlib
from pathlib import Path

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import col, sum, count, lit, array, desc, row_number, collect_list, udf, countDistinct
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.evaluation import RankingEvaluator
from pyspark.ml.feature import StringIndexer
import numpy as np
import os

def main(spark, userID):
    DATA_DIR = Path(__file__).resolve().parent / "data"
    sample_frac = float(os.getenv("BASELINE_SAMPLE_FRACTION", "0.05"))  # 5% by default to fit memory locally
    print(f"Sampling fraction for baseline: {sample_frac}")
    
    # Read files from local data directory (with sampling to reduce memory footprint)
    full_interaction = spark.read.parquet(
        str(DATA_DIR / 'full_interaction_all_FINAL.parquet'),
        header=True,
        schema='interactions INT, recording_mbid STRING, user_id INT'
    ).sample(False, sample_frac, seed=42).repartition(50)
    train_interaction = spark.read.parquet(
        str(DATA_DIR / 'train_interaction_noncoldstart_ALL.parquet'),
        header=True,
        schema='user_id INT, interactions INT, recording_mbid STRING'
    ).sample(False, sample_frac, seed=42).repartition(50)
    test_interaction = spark.read.parquet(
        str(DATA_DIR / 'val_interaction_noncoldstart_ALL.parquet'),
        header=True,
        schema='user_id INT, interactions INT, recording_mbid STRING'
    ).sample(False, sample_frac, seed=42).repartition(50)
    train_interaction_cs = spark.read.parquet(
        str(DATA_DIR / 'train_interaction_coldstart_ALL.parquet'),
        header=True,
        schema='user_id INT, interactions INT, recording_mbid STRING'
    ).sample(False, sample_frac, seed=42).repartition(50)
    test_interaction_cs = spark.read.parquet(
        str(DATA_DIR / 'val_interaction_coldstart_ALL.parquet'),
        header=True,
        schema='user_id INT, interactions INT, recording_mbid STRING'
    ).sample(False, sample_frac, seed=42).repartition(50)

    # Using Marcia's StringIndexer Implementation
    indexer = StringIndexer(inputCol="recording_mbid", outputCol="recording_mbid_double", handleInvalid="keep").fit(full_interaction)
    full_indexed_double = indexer.transform(full_interaction)
    train_indexed_double = indexer.transform(train_interaction)
    test_indexed_double = indexer.transform(test_interaction)
    train_cs_indexed_double = indexer.transform(train_interaction_cs)
    test_cs_indexed_double = indexer.transform(test_interaction_cs)

    # Filter out interactions with 0 listens -- No longer necessary with our current pre-processing. But left it in as an extra check.
    filtered_df_train = train_indexed_double.filter(col("interactions") > 0)
    filtered_df_test = test_indexed_double.filter(col("interactions") > 0)

    top_k = 100
    
    # Group by song, calculate total listens and number of users who listened
    grouped_df_train = filtered_df_train.groupBy("recording_mbid_double") \
    .agg(
         sum("interactions").alias("total_listens"), 
         count("user_id").alias("num_users")
    )
    
    # Set beta (to be hypertuned)
    beta = 5000

    # Compute the average number of listens per user, conditioned on listening
    avg_listens_df_train = grouped_df_train.withColumn("avg_listens_per_user", col("total_listens") / (col("num_users") + beta))\
    .orderBy(col("avg_listens_per_user").desc())

    print("Converting data into format for RankingEvaluator; predictionCol and labelCol require list of type double....")
    
    # Compute top-k recommendations based on average listens per user
    top_k_recommendations = avg_listens_df_train.select("recording_mbid_double").limit(top_k)

    # Modify the code to use the UDF to convert recording_mbids to double type
    top_k_array = top_k_recommendations.withColumn("predicted_labels", col("recording_mbid_double")).agg(collect_list("predicted_labels").alias("predicted_labels"))

    # Assign the same top_k_recommendations to each user
    distinct_user_ids = filtered_df_train.select("user_id").distinct()
    top_k_array = top_k_array.repartition(100)
    distinct_user_ids = distinct_user_ids.repartition(100)
    predicted_labels = distinct_user_ids.crossJoin(top_k_array)

    # Define the window partitioned by user_id and ordered by interactions in descending order
    window_spec = Window.partitionBy("user_id").orderBy(desc("interactions"))

    # Rank recording_mbids by the number of interactions for each user and filter out those with rank > k
    filtered_df_test = filtered_df_test.withColumn("rank", row_number().over(window_spec))
    top_k_true_labels = filtered_df_test.filter(col("rank") <= top_k).drop("rank")
    true_labels = top_k_true_labels.withColumn("true_labels",col("recording_mbid_double")).groupBy("user_id").agg(collect_list("true_labels").alias("true_labels"))

    predicted_labels = predicted_labels.repartition(100)
    true_labels = true_labels.repartition(100)
    joined_labels = true_labels.join(predicted_labels, on="user_id")
    
    # Create a RankingEvaluator for evaluating the MAP metric
    evaluator_map = RankingEvaluator(k=top_k, metricName="meanAveragePrecisionAtK", labelCol="true_labels",
                             predictionCol="predicted_labels")
      
    # Compute the MAP
    map_score = evaluator_map.evaluate(joined_labels)

    # Print the MAP/NDCG score
    print("Mean Average Precision of non cold start:", map_score)

    print("Processing cold-start problem...")
    
    # Assign the same top_k_recommendations to each user
    distinct_user_ids_cs = train_cs_indexed_double.select("user_id").distinct()
    top_k_array = top_k_array.repartition(100)
    distinct_user_ids_cs = distinct_user_ids_cs.repartition(100)
    predicted_labels_cs = distinct_user_ids_cs.crossJoin(top_k_array)

    # Rank recording_mbids by the number of interactions for each user and filter out those with rank > k
    test_cs_indexed_double = test_cs_indexed_double.withColumn("rank", row_number().over(window_spec))
    top_k_true_labels_cs = test_cs_indexed_double.filter(col("rank") <= top_k).drop("rank")
    true_labels_cs = top_k_true_labels_cs.withColumn("true_labels",col("recording_mbid_double")).groupBy("user_id").agg(collect_list("true_labels").alias("true_labels"))
   
    predicted_labels_cs = predicted_labels_cs.repartition(100)
    true_labels_cs = true_labels_cs.repartition(100)
    joined_labels_cs = true_labels_cs.join(predicted_labels_cs, on="user_id")

    # Compute the MAP
    map_score_cs = evaluator_map.evaluate(joined_labels_cs)
    print("Mean Average Precision of cold start:", map_score_cs)

    # Skipping final test-set evaluation because test files are not available locally.



if __name__ == "__main__":

    # Create the spark session object
    spark = (
        SparkSession.builder
        .config("spark.driver.memory", "6g")
        .config("spark.executor.memory", "3g")
        .config("spark.executor.cores", "4")
        .config("spark.sql.shuffle.partitions", "50")
        .config("spark.default.parallelism", "50")
        .getOrCreate()
    )

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = getpass.getuser()
    # Call our main routine
    main(spark, userID)
