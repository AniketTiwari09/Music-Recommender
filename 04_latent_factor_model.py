#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Template script to connect to Active Spark Session
Usage:
    $ spark-submit --deploy-mode client latent_factor_model.py <any arguments you wish to add>
'''

import os
import sys
import getpass
import hashlib
import time
from pathlib import Path

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col,
    sum,
    count,
    lit,
    array,
    desc,
    row_number,
    collect_list,
    udf,
    countDistinct,
    from_json,
    explode,
)
from pyspark.sql.types import IntegerType, DoubleType, ArrayType
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RankingEvaluator
from sklearn.metrics import ndcg_score

from ast import literal_eval

def main(spark, userID):
    DATA_DIR = Path(__file__).resolve().parent / "data"
    sample_frac = float(os.getenv("ALS_SAMPLE_FRACTION", "0.05"))  # 5% by default to fit locally
    print(f"Sampling fraction for ALS: {sample_frac}")
    filtered_df_full = spark.read.parquet(
        str(DATA_DIR / 'full_interaction_all_FINAL.parquet'),
        header=True,
        schema='interactions INT, recording_mbid STRING, user_id INT'
    ).sample(False, sample_frac, seed=42).repartition(50)
    filtered_df_train = spark.read.parquet(
        str(DATA_DIR / 'train_interaction_noncoldstart_ALL.parquet'),
        header=True,
        schema='user_id INT, interactions INT, recording_mbid STRING'
    ).sample(False, sample_frac, seed=42).repartition(50)
    filtered_df_val = spark.read.parquet(
        str(DATA_DIR / 'val_interaction_noncoldstart_ALL.parquet'),
        header=True,
        schema='user_id INT, interactions INT, recording_mbid STRING'
    ).sample(False, sample_frac, seed=42).repartition(50)
    # Local run: skip test set (files not available). Comment out if you have it.
    # df_test = spark.read.parquet(str(DATA_DIR / 'test.parquet'))
    
    #cold = spark.read.parquet('hdfs:/user/xm618_nyu_edu/cold_start_users.parquet')
    #cold_start_user_test = df_test.join(cold, df_test.user_id==cold.user_id, "inner").drop(df_test.user_id)
    #cold_start_user_test = cold_start_user_test.repartition(100)
    #print("SAVING")
    #cold_start_user_test.write.mode("overwrite").parquet("cold_start_test.parquet")
    #print("SAVED")
    
    # Skipping test-set filtering; no test set locally.

    #filtered_df_train.show(3)
    # 1. Hash recording.mbid to intergers
    print("Converting 'recording.mbid' to integers... '")

    indexer = StringIndexer(inputCol="recording_mbid", outputCol="recording_mbid_double", handleInvalid="keep").fit(filtered_df_full)
    full_filtered_indexed_double = indexer.transform(filtered_df_full)
    # Mapping from internal integer IDs back to original recording_mbid
    item_mapping = (
        full_filtered_indexed_double.select("recording_mbid_double", "recording_mbid")
        .distinct()
        .withColumn("recording_mbid_int", col("recording_mbid_double").cast("int"))
        .select("recording_mbid_int", "recording_mbid")
    )
    train_filtered_indexed_double = indexer.transform(filtered_df_train)
    val_filtered_indexed_double = indexer.transform(filtered_df_val)
    # test_indexed_double = indexer.transform(df_test)

    full_filtered_indexed_int = full_filtered_indexed_double.withColumn("recording_mbid_int", col("recording_mbid_double").cast("int")).drop(full_filtered_indexed_double.recording_mbid_double)
    train_filtered_indexed_int = train_filtered_indexed_double.withColumn("recording_mbid_int", col("recording_mbid_double").cast("int")).drop(train_filtered_indexed_double.recording_mbid_double).repartition(200)
    val_filtered_indexed_int = val_filtered_indexed_double.withColumn("recording_mbid_int", col("recording_mbid_double").cast("int")).drop(val_filtered_indexed_double.recording_mbid_double)
    # test_indexed_int = test_indexed_double.withColumn("recording_mbid_int", col("recording_mbid_double").cast("int")).drop(test_indexed_double.recording_mbid_double)

    # 2. Hyperparameter tuning for ALS
    print("Fitting ALS... ")
    print("\n")
    print("Starting timer")
    
    start_time = time.time()
    # Use lighter settings locally to avoid OOM on a laptop.
    als = ALS(
        userCol="user_id",
        itemCol="recording_mbid_int",
        ratingCol="interactions",
        maxIter=10,
        numUserBlocks=10,
        numItemBlocks=10,
        implicitPrefs=True,
        rank=50,
        regParam=0.5,
        alpha=1,
    )
    alsModel = als.fit(train_filtered_indexed_int)
    
    end_time = time.time()
    print("Total ALS model training time: {} seconds".format(end_time - start_time))


    # Making the top_k recommendations for users in val
    filtered_df_val.createOrReplaceTempView('filtered_df_val')
    users_in_val = spark.sql("SELECT DISTINCT user_id from filtered_df_val")
    top_k = 100
    print("Making top_k recommendations... ")
    predictions = alsModel.recommendForUserSubset(users_in_val, 100)
    predictions.createOrReplaceTempView("predictions")
    predictions = predictions.withColumn("song_recs",col("recommendations.recording_mbid_int")).drop("recommendation.recording_mbid_int")
    predictions.createOrReplaceTempView("predictions")

    # Transforming recommendtaion format for evaluation
    print("Transformning recommendation format for evaluation... ")
    df_predicted_labels = predictions.withColumn("predicted_labels", col("song_recs").cast(ArrayType(DoubleType()))).drop(predictions.song_recs)

    # Transforming true label format for evaluation
    window_spec = Window.partitionBy("user_id").orderBy(desc("interactions"))
    df_val_label = val_filtered_indexed_double.withColumn("rank", row_number().over(window_spec))
    top_k_true_labels = df_val_label.filter(col("rank") <= top_k).drop("rank")
    df_true_labels = top_k_true_labels.groupBy("user_id").agg(collect_list("recording_mbid_double").alias("true_labels"))
    
    joined_labels = df_true_labels.join(df_predicted_labels, on="user_id")

    # Create a RankingEvaluator for evaluating the MAP metric
    evaluator_map = RankingEvaluator(k=top_k, metricName="meanAveragePrecisionAtK", labelCol="true_labels",
                             predictionCol="predicted_labels")
      
    # Compute the MAP
    map_score = evaluator_map.evaluate(joined_labels)

    # Print the MAP/NDCG score
    print("Mean Average Precision(val):", map_score)

    # Making the top_k recommendations for users in test
    print("For test (local run: generating recommendations only, no test metrics)")
    top_k = 100
    print("Making top_k recommendations... ")
    predictions_test = alsModel.recommendForAllUsers(top_k)
    predictions_test.createOrReplaceTempView("predictions_test")
    predictions_test = predictions_test.withColumn("song_recs",col("recommendations.recording_mbid_int")).drop("recommendation.recording_mbid_int")
    predictions_test.createOrReplaceTempView("predictions_test")

    # Save top-N ALS recommendations per user for later use
    print("Saving ALS top-100 recommendations per user...")
    recs_exploded = predictions_test.select(
        col("user_id"),
        explode("recommendations").alias("rec"),
    ).select(
        col("user_id"),
        col("rec.recording_mbid_int").alias("recording_mbid_int"),
        col("rec.rating").alias("score"),
    )

    recs_with_mbid = recs_exploded.join(item_mapping, on="recording_mbid_int", how="left")
    rec_window = Window.partitionBy("user_id").orderBy(desc("score"))
    recs_ranked = recs_with_mbid.withColumn("rank", row_number().over(rec_window))

    recs_ranked.write.mode("overwrite").parquet(str(DATA_DIR / "als_recommendations.parquet"))
    print("ALS recommendations saved to data/als_recommendations.parquet")

if __name__ == "__main__":

    # Create the spark session object
    spark = (
        SparkSession.builder
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "4g")
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
