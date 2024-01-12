package com.example.scalabankinsights

import org.apache.spark.ml.classification.{DecisionTreeClassifier, DecisionTreeClassificationModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * Class representing a Decision Tree Classification model.
 */
class DTModel {

  /**
   * Trains a Decision Tree Classification model on the provided training data and evaluates on testing data.
   *
   * @param trainingData          DataFrame containing the training data.
   * @param testingData           DataFrame containing the testing data.
   * @param spark                 SparkSession.
   * @param labelCol              Name of the label column in the data.
   * @param featuresCol           Name of the features column in the data.
   * @param maxDepth              Maximum depth of the tree (default: 5).
   * @param maxBins               Maximum number of bins for discretizing continuous features (default: 32).
   * @param minInstancesPerNode   Minimum number of instances each child must have after split (default: 1).
   * @return Tuple containing the trained Decision Tree Classification model and predictions DataFrame.
   */
  def trainModel(
                  trainingData: DataFrame,
                  testingData: DataFrame,
                  spark: SparkSession,
                  labelCol: String = "label",
                  featuresCol: String = "features",
                  maxDepth: Int = 5,
                  maxBins: Int = 32,
                  minInstancesPerNode: Int = 1
                ): (DecisionTreeClassificationModel, DataFrame) = {

    // Create a Decision Tree Classification instance with additional parameters
    val dt = new DecisionTreeClassifier()
      .setLabelCol(labelCol)
      .setFeaturesCol(featuresCol)
      .setMaxDepth(maxDepth)
      .setMaxBins(maxBins)
      .setMinInstancesPerNode(minInstancesPerNode)

    // Train the model on the training data
    val model = dt.fit(trainingData)

    // Make predictions on the testing data
    val predictions = model.transform(testingData)

    // Calculate and save regression metrics
    ClassificationMetrics.calculateAndSaveMetrics(predictions, "Decision Tree Classification", "C:\\Users\\chand\\Desktop\\ScalaBankInsights\\Models\\DTModel\\metrics", spark)

    // Save the model and predictions
    model.write.overwrite().save("C:\\Users\\chand\\Desktop\\ScalaBankInsights\\Models\\DTModel\\model")
    DataFrameUtils.saveDataFrame(predictions, "C:\\Users\\chand\\Desktop\\ScalaBankInsights\\Models\\DTModel\\predict", "parquet")

    // Return the trained model and predictions
    (model, predictions)
  }
}
