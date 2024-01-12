package com.example.scalabankinsights

import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * Object containing functions for loading datasets.
 */
object DataLoader {

  /**
   * Load datasets from Raw CSV files.
   *
   * @param trainFile1 The path to the CSV file for the first training set.
   * @param trainFile2 The path to the CSV file for the second training set.
   * @param spark      The Spark session.
   * @return A tuple of DataFrames containing the loaded data.
   */
  def loadRawData(
                   trainFile1: String,
                   trainFile2: String,
                   spark: SparkSession
                 ): (DataFrame, DataFrame) = {
    // Read the CSV files for the first training dataset
    val train1 = readCSV(trainFile1, spark)

    // Read the CSV files for the second training dataset
    val train2 = readCSV(trainFile2, spark)

    // Return a tuple of DataFrames
    (train1, train2)
  }

  /**
   * Read a CSV file into a DataFrame.
   *
   * @param filePath The path to the CSV file.
   * @param spark    The Spark session.
   * @return The DataFrame containing the data from the CSV file.
   */
  private def readCSV(filePath: String, spark: SparkSession): DataFrame = {
    spark.read
      .option("header", value = true)
      .option("delimiter", ";")
      .option("inferSchema", "true")
      .csv(filePath)
  }
}
