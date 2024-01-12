package com.example.scalabankinsights

import org.apache.spark.ml.classification.{RandomForestClassifier, RandomForestClassificationModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * Class representing a Random Forest Classification model.
 */
class RFModel {

  /**
   * Trains a Random Forest Classification model on the provided training data and evaluates on testing data.
   *
   * @param trainingData      DataFrame containing the training data.
   * @param testingData       DataFrame containing the testing data.
   * @param spark             SparkSession.
   * @param labelCol          Name of the label column in the data.
   * @param featuresCol       Name of the features column in the data.
   * @param numTrees          Number of trees in the forest (default: 20).
   * @param maxDepth          Maximum depth of the tree (default: 5).
   * @param maxBins           Maximum number of bins for discretizing continuous features (default: 32).
   * @param minInstancesPerNode   Minimum number of instances each child must have after split (default: 1).
   * @return Tuple containing the trained Random Forest Classification model and predictions DataFrame.
   */
  def trainModel(
                  trainingData: DataFrame,
                  testingData: DataFrame,
                  spark: SparkSession,
                  labelCol: String = "label",
                  featuresCol: String = "features",
                  numTrees: Int = 20,
                  maxDepth: Int = 5,
                  maxBins: Int = 32,
                  minInstancesPerNode: Int = 1
                ): (RandomForestClassificationModel, DataFrame) = {

    // Create a Random Forest Classification instance with additional parameters
    val rf = new RandomForestClassifier()
      .setLabelCol(labelCol)
      .setFeaturesCol(featuresCol)
      .setNumTrees(numTrees)
      .setMaxDepth(maxDepth)
      .setMaxBins(maxBins)
      .setMinInstancesPerNode(minInstancesPerNode)

    // Train the model on the training data
    val model = rf.fit(trainingData)

    // Make predictions on the testing data
    val predictions = model.transform(testingData)

    // Calculate and save regression metrics
    ClassificationMetrics.calculateAndSaveMetrics(predictions, "Random Forest Classification", "C:\\Users\\chand\\Desktop\\ScalaBankInsights\\Models\\RFModel\\metrics", spark)

    // Save the model and predictions
    model.write.overwrite().save("C:\\Users\\chand\\Desktop\\ScalaBankInsights\\Models\\RFModel\\model")
    DataFrameUtils.saveDataFrame(predictions, "C:\\Users\\chand\\Desktop\\ScalaBankInsights\\Models\\RFModel\\predict", "parquet")

    // Return the trained model and predictions
    (model, predictions)
  }
}
