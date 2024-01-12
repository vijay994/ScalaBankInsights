package com.example.scalabankinsights

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * Object containing functions to calculate and print classification metrics.
 */
object ClassificationMetrics {

  /**
   * Calculates and prints classification metrics for a given set of predictions.
   *
   * @param predictions DataFrame with label and prediction columns.
   * @param modelName   Name of the model for which metrics are calculated.
   * @param outputPath  Output path for saving the metrics DataFrame.
   * @param spark       SparkSession.
   */
  def calculateAndSaveMetrics(
                               predictions: DataFrame,
                               modelName: String,
                               outputPath: String,
                               spark: SparkSession
                             ): Unit = {
    import spark.implicits._

    // Create a multiclass classification evaluator
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")

    // Calculate accuracy, precision, recall, and f1 score
    val accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    val precision = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
    val recall = evaluator.setMetricName("weightedRecall").evaluate(predictions)
    val f1Score = evaluator.setMetricName("f1").evaluate(predictions)

    // Create a DataFrame with metrics results
    val metricsDF = Seq(
      ("Accuracy", accuracy),
      ("Precision", precision),
      ("Recall", recall),
      ("F1 Score", f1Score)
    ).toDF("Metric", "Value").withColumn("Model", lit(modelName))

    // Save the DataFrame to the specified path in a specified format (e.g., "parquet")
    DataFrameUtils.saveDataFrame(metricsDF, outputPath, "parquet")
  }
}
