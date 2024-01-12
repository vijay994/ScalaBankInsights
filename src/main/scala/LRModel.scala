package com.example.scalabankinsights

import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * Class representing a Logistic Regression model.
 */
class LRModel {

  /**
   * Trains a Logistic Regression model on the provided training data and evaluates on testing data.
   *
   * @param trainingData DataFrame containing the training data.
   * @param testingData  DataFrame containing the testing data.
   * @param spark        SparkSession.
   * @param labelCol     Name of the label column in the data.
   * @param featuresCol  Name of the features column in the data.
   * @param maxIter      Maximum number of iterations for the optimization algorithm.
   * @param regParam     Regularization parameter (λ).
   * @param elasticNetParam Elastic Net mixing parameter, in range [0, 1].
   * @param tol          Convergence tolerance of the iterations.
   * @param family       The name of the family which is a description of the label distribution to be used in the model.
   * @return Tuple containing the trained Logistic Regression model and predictions DataFrame.
   */
  def trainModel(
                  trainingData: DataFrame,
                  testingData: DataFrame,
                  spark: SparkSession,
                  labelCol: String = "label",
                  featuresCol: String = "features",
                  maxIter: Int = 100,
                  regParam: Double = 0.0,
                  elasticNetParam: Double = 0.0,
                  tol: Double = 1e-6,
                  family: String = "auto"
                ): (LogisticRegressionModel, DataFrame) = {

    // Create a Logistic Regression instance with additional parameters
    val lr = new LogisticRegression()
      .setLabelCol(labelCol)
      .setFeaturesCol(featuresCol)
      .setMaxIter(maxIter)
      .setRegParam(regParam)
      .setElasticNetParam(elasticNetParam)
      .setTol(tol)
      .setFamily(family)

    // Train the model on the training data
    val model = lr.fit(trainingData)

    // Make predictions on the testing data
    val predictions = model.transform(testingData)

    // Calculate and save regression metrics
    ClassificationMetrics.calculateAndSaveMetrics(predictions, "Logistic Regression", "C:\\Users\\chand\\Desktop\\ScalaBankInsights\\Models\\LRModel\\metrics", spark)

    // Save the model and predictions
    model.write.overwrite().save("C:\\Users\\chand\\Desktop\\ScalaBankInsights\\Models\\LRModel\\model")
    DataFrameUtils.saveDataFrame(predictions, "C:\\Users\\chand\\Desktop\\ScalaBankInsights\\Models\\LRModel\\predict", "parquet")

    // Return the trained model and predictions
    (model, predictions)
  }
}
