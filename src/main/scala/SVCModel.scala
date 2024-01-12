package com.example.scalabankinsights

import org.apache.spark.ml.classification.{LinearSVC, LinearSVCModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * Class representing a Linear Support Vector Classification model.
 */
class SVCModel {

  /**
   * Trains a Linear Support Vector Classification model on the provided training data and evaluates on testing data.
   *
   * @param trainingData DataFrame containing the training data.
   * @param testingData  DataFrame containing the testing data.
   * @param spark        SparkSession.
   * @param labelCol     Name of the label column in the data.
   * @param featuresCol  Name of the features column in the data.
   * @param maxIter      Maximum number of iterations for the optimization algorithm.
   * @param regParam     Regularization parameter (λ).
   * @param tol          Convergence tolerance of the iterations.
   * @return Tuple containing the trained Linear Support Vector Classification model and predictions DataFrame.
   */
  def trainModel(
                  trainingData: DataFrame,
                  testingData: DataFrame,
                  spark: SparkSession,
                  labelCol: String = "label",
                  featuresCol: String = "features",
                  maxIter: Int = 100,
                  regParam: Double = 0.0,
                  tol: Double = 1e-6
                ): (LinearSVCModel, DataFrame) = {

    // Create a Linear Support Vector Classification instance with additional parameters
    val svc = new LinearSVC()
      .setLabelCol(labelCol)
      .setFeaturesCol(featuresCol)
      .setMaxIter(maxIter)
      .setRegParam(regParam)
      .setTol(tol)

    // Train the model on the training data
    val model = svc.fit(trainingData)

    // Make predictions on the testing data
    val predictions = model.transform(testingData)

    // Calculate and save regression metrics
    ClassificationMetrics.calculateAndSaveMetrics(predictions, "Linear Support Vector Classification", "C:\\Users\\chand\\Desktop\\ScalaBankInsights\\Models\\SVCModel\\metrics", spark)

    // Save the model and predictions
    model.write.overwrite().save("C:\\Users\\chand\\Desktop\\ScalaBankInsights\\Models\\SVCModel\\model")
    DataFrameUtils.saveDataFrame(predictions, "C:\\Users\\chand\\Desktop\\ScalaBankInsights\\Models\\SVCModel\\predict", "parquet")

    // Return the trained model and predictions
    (model, predictions)
  }
}
