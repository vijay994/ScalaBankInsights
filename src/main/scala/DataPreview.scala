package com.example.scalabankinsights

import org.apache.spark.sql.SparkSession

object DataPreview {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("DataPreview")
      .master("local[*]")
      .getOrCreate()

    val filePath = "data/Original/bank/bank.csv"

    // Load the dataset
    val rawData = spark.read
      .option("header", value = true)
      .option("delimiter", ";")
      .csv(filePath)

    // Show the first few rows of the raw data
    rawData.show()

    // Stop the Spark session
    spark.stop()
  }
}
