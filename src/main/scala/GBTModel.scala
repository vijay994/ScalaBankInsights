package com.example.scalabankinsights

import org.apache.spark.ml.classification.{GBTClassifier, GBTClassificationModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * Class representing a Gradient-Boosted Trees Classification model.
 */
class GBTModel {

  /**
   * Trains a Gradient-Boosted Trees Classification model on the provided training data and evaluates on testing data.
   *
   * @param trainingData DataFrame containing the training data.
   * @param testingData  DataFrame containing the testing data.
   * @param spark       SparkSession.
   * @param labelCol    Name of the label column in the data.
   * @param featuresCol Name of the features column in the data.
   * @param maxDepth    Maximum depth of the tree (default: 5).
   * @param maxBins     Maximum number of bins for discretizing continuous features (default: 32).
   * @param maxIter     Maximum number of iterations (default: 20).
   * @param stepSize    Step size (default: 0.1).
   * @return Tuple containing the trained Gradient-Boosted Trees Classification model and predictions DataFrame.
   */
  def trainModel(
                  trainingData: DataFrame,
                  testingData: DataFrame,
                  spark: SparkSession,
                  labelCol: String = "label",
                  featuresCol: String = "features",
                  maxDepth: Int = 5,
                  maxBins: Int = 32,
                  maxIter: Int = 20,
                  stepSize: Double = 0.1
                ): (GBTClassificationModel, DataFrame) = {

    // Create a Gradient-Boosted Trees Classification instance with additional parameters
    val gbt = new GBTClassifier()
      .setLabelCol(labelCol)
      .setFeaturesCol(featuresCol)
      .setMaxDepth(maxDepth)
      .setMaxBins(maxBins)
      .setMaxIter(maxIter)
      .setStepSize(stepSize)

    // Train the model on the training data
    val model = gbt.fit(trainingData)

    // Make predictions on the testing data
    val predictions = model.transform(testingData)

    // Calculate and save regression metrics
    ClassificationMetrics.calculateAndSaveMetrics(predictions, "Gradient-Boosted Trees Classification", "C:\\Users\\chand\\Desktop\\ScalaBankInsights\\Models\\GBTModel\\metrics", spark)

    // Save the model and predictions
    model.write.overwrite().save("C:\\Users\\chand\\Desktop\\ScalaBankInsights\\Models\\GBTModel\\model")
    DataFrameUtils.saveDataFrame(predictions, "C:\\Users\\chand\\Desktop\\ScalaBankInsights\\Models\\GBTModel\\predict", "parquet")

    // Return the trained model and predictions
    (model, predictions)
  }
}
