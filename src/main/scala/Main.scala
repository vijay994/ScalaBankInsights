package com.example.scalabankinsights

import org.apache.spark.sql.SparkSession

/**
 * Main entry point for the ScalaBankInsights application.
 */
object Main {
  /**
   * Main method to run the ScalaBankInsights application.
   *
   * @param args Command-line arguments (not used in this application).
   */
  def main(args: Array[String]): Unit = {
    // Set Hadoop home directory
    System.setProperty("hadoop.home.dir", "C:/Users/chand/Desktop/ScalaBankInsights/Hadoop")

    // Create a Spark session
    val spark = SparkSession.builder()
      .appName("ScalaBankInsights")
      .master("local[*]") // Run Spark locally using all available cores
      .getOrCreate()

    try {
      // Paths to the Raw Data files
      val trainPath1 = "data_org/bank/bank-full.csv"
      val trainPath2 = "data_org/bank-additional/bank-additional-full.csv"

      // Load Raw datasets
      val (trainData1, trainData2) =
        DataLoader.loadRawData(trainPath1, trainPath2, spark)

      // Preprocess the datasets
      val (train, test) = DataPreprocessing.preprocessData(trainData1, trainData2, spark)

      // Training and testing the Machine learning models
      val (_, _) = new LRModel().trainModel(train, test, spark)
      val (_, _) = new SVCModel().trainModel(train, test, spark)
      val (_, _) = new DTModel().trainModel(train, test, spark)
      val (_, _) = new RFModel().trainModel(train, test, spark)
      val (_, _) = new GBTModel().trainModel(train, test, spark)

    } finally {
      // Checking the Evaluation metrics of the models
      ModelEvaluation.loadModelMetrics("LRModel", spark)
      ModelEvaluation.loadModelMetrics("DTModel", spark)
      ModelEvaluation.loadModelMetrics("RFModel", spark)
      ModelEvaluation.loadModelMetrics("SVCModel", spark)
      ModelEvaluation.loadModelMetrics("GBTModel", spark)

      // Checking the Predictions of the models
      ModelEvaluation.loadModelPredictions("LRModel", spark)
      ModelEvaluation.loadModelPredictions("DTModel", spark)
      ModelEvaluation.loadModelPredictions("RFModel", spark)
      ModelEvaluation.loadModelPredictions("SVCModel", spark)
      ModelEvaluation.loadModelPredictions("GBTModel", spark)

      // Loading the models
      ModelEvaluation.loadModelFile("LRModel")
      ModelEvaluation.loadModelFile("DTModel")
      ModelEvaluation.loadModelFile("RFModel")
      ModelEvaluation.loadModelFile("SVCModel")
      ModelEvaluation.loadModelFile("GBTModel")

      // Stop the Spark session in a finally block to ensure it gets stopped even if an exception occurs
      spark.stop()
    }
  }
}
