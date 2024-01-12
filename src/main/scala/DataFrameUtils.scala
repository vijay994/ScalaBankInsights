package com.example.scalabankinsights

import org.apache.spark.sql.DataFrame

/**
 * Object containing utility functions for working with DataFrames and machine learning models.
 */
object DataFrameUtils {

  /**
   * Calculate the shape of a DataFrame (number of rows and columns).
   *
   * @param df DataFrame for which to calculate the shape.
   * @return A tuple representing the shape of the DataFrame (number of rows, number of columns).
   */
  def shape(df: DataFrame): (Long, Int) = {
    // Get the number of rows and columns
    val numRows = df.count()
    val numCols = df.columns.length

    // Return the tuple representing the shape
    (numRows, numCols)
  }

  /**
   * Save a DataFrame to a specified file path in a specific format (e.g., Parquet, CSV).
   *
   * @param df     DataFrame to be saved.
   * @param path   File path where the DataFrame will be saved.
   * @param format File format for saving the DataFrame (e.g., "parquet", "csv").
   */
  def saveDataFrame(df: DataFrame, path: String, format: String): Unit = {
    // Save the DataFrame to the specified path in the given format
    df.write.format(format).mode("overwrite").save(path)
  }
}
