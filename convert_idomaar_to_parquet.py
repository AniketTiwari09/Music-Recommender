#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Convert raw Idomaar files to parquet inputs expected by the pipeline.

Run from repo root:
    spark-submit --master local[*] convert_idomaar_to_parquet.py

Outputs into data/:
    users_train.parquet
    tracks_train.parquet
    interactions_train.parquet
"""

from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, lit
from pyspark.sql.types import (
    ArrayType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    entities_dir = data_dir / "entities"
    relations_dir = data_dir / "relations"

    spark = SparkSession.builder.master("local[*]").getOrCreate()

    # Schemas for JSON columns
    track_meta_schema = StructType(
        [
            StructField("duration", LongType(), True),
            StructField("playcount", LongType(), True),
            StructField("MBID", StringType(), True),
            StructField("name", StringType(), True),
        ]
    )
    user_meta_schema = StructType(
        [
            StructField("lastfm_username", StringType(), True),
            StructField("gender", StringType(), True),
            StructField("age", StringType(), True),
            StructField("country", StringType(), True),
            StructField("playcount", LongType(), True),
            StructField("playlists", LongType(), True),
            StructField("subscribertype", StringType(), True),
        ]
    )
    entity_ref_schema = ArrayType(
        StructType(
            [
                StructField("type", StringType(), True),
                StructField("id", IntegerType(), True),
            ]
        )
    )
    event_entities_schema = StructType(
        [
            StructField("subjects", entity_ref_schema, True),
            StructField("objects", entity_ref_schema, True),
        ]
    )

    # Tracks -> tracks_train.parquet
    tracks_raw = spark.read.csv(
        str(entities_dir / "tracks.idomaar"),
        sep="\t",
        header=False,
        schema=StructType(
            [
                StructField("type", StringType(), True),
                StructField("track_id", IntegerType(), True),
                StructField("ts", LongType(), True),
                StructField("meta_json", StringType(), True),
                StructField("rel_json", StringType(), True),
            ]
        ),
    )
    tracks = tracks_raw.withColumn("meta", from_json(col("meta_json"), track_meta_schema))
    tracks_df = tracks.select(
        col("track_id").cast("string").alias("recording_msid"),
        col("meta.MBID").alias("recording_mbid"),
    )
    tracks_df.write.mode("overwrite").parquet(str(data_dir / "tracks_train.parquet"))

    # Users -> users_train.parquet
    users_raw = spark.read.csv(
        str(entities_dir / "users.idomaar"),
        sep="\t",
        header=False,
        schema=StructType(
            [
                StructField("type", StringType(), True),
                StructField("user_id", IntegerType(), True),
                StructField("ts", LongType(), True),
                StructField("meta_json", StringType(), True),
            ]
        ),
    )
    users = users_raw.withColumn("meta", from_json(col("meta_json"), user_meta_schema))
    users_df = users.select("user_id", "meta.*")
    users_df.write.mode("overwrite").parquet(str(data_dir / "users_train.parquet"))

    # Events -> interactions_train.parquet
    events_raw = spark.read.csv(
        str(relations_dir / "events.idomaar"),
        sep="\t",
        header=False,
        schema=StructType(
            [
                StructField("event_type", StringType(), True),
                StructField("event_id", LongType(), True),
                StructField("ts", LongType(), True),
                StructField("props_json", StringType(), True),
                StructField("entities_json", StringType(), True),
            ]
        ),
    )
    events = events_raw.withColumn(
        "entities", from_json(col("entities_json"), event_entities_schema)
    )
    interactions_df = (
        events.select(
            col("entities.subjects")[0]["id"].alias("user_id"),
            col("entities.objects")[0]["id"].cast("string").alias("recording_msid"),
        )
        .withColumn("interactions", lit(1))
    )
    interactions_df.write.mode("overwrite").parquet(
        str(data_dir / "interactions_train.parquet")
    )

    spark.stop()


if __name__ == "__main__":
    main()
