package com.example.scalabankinsights

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.classification.{LogisticRegressionModel, LinearSVCModel, DecisionTreeClassificationModel, RandomForestClassificationModel, GBTClassificationModel}

/**
 * Object containing functions for loading and previewing a classification model's predictions and metrics.
 */
object ModelEvaluation {

  /**
   * Loads and previews predictions for a specified classification model.
   *
   * @param modelName Name of the classification model to load.
   * @param spark     SparkSession.
   */
  def loadModelPredictions(modelName: String, spark: SparkSession): Unit = {
    // Hardcoded path for predictions
    val predictionsPath = s"C:\\Users\\chand\\Desktop\\ScalaBankInsights\\Models\\$modelName\\predict"

    // Load predictions data
    val predictionsDF = spark.read.parquet(predictionsPath)

    // Print predictions
    predictionsDF.show()
  }

  /**
   * Loads and previews metrics for a specified classification model.
   *
   * @param modelName Name of the classification model to load.
   * @param spark     SparkSession.
   */
  def loadModelMetrics(modelName: String, spark: SparkSession): Unit = {
    // Hardcoded path for metrics
    val metricsPath = s"C:\\Users\\chand\\Desktop\\ScalaBankInsights\\Models\\$modelName\\metrics"

    // Load metrics data
    val metricsDF = spark.read.parquet(metricsPath)

    // Print Metrics
    metricsDF.show()
  }

  /**
   * Loads a specific classification model based on the model name and prints its details.
   *
   * @param modelName Name of the classification model to load.
   */
  def loadModelFile(modelName: String): Unit = {
    val modelPath = s"C:\\Users\\chand\\Desktop\\ScalaBankInsights\\Models\\$modelName\\model"

    // Load specific classification model based on modelName
    val loadedModel = modelName match {
      case "LRModel"   => LogisticRegressionModel.load(modelPath)
      case "SVCModel"  => LinearSVCModel.load(modelPath)
      case "DTModel"   => DecisionTreeClassificationModel.load(modelPath)
      case "RFModel"   => RandomForestClassificationModel.load(modelPath)
      case "GBTModel"  => GBTClassificationModel.load(modelPath)
      case _           => throw new IllegalArgumentException(s"Unsupported model name: $modelName")
    }

    // Print loaded model details
    println(loadedModel)
  }
}
