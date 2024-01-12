package com.example.scalabankinsights

import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
 * Object containing functions for preprocessing data.
 */
object DataPreprocessing {

  /**
   * Preprocess the input DataFrames.
   *
   * @param trainDf1 The data for the first training set.
   * @param trainDf2 The data for the second training set.
   * @param spark    SparkSession
   * @return A tuple of DataFrames containing the loaded data.
   */
  def preprocessData(
                      trainDf1: DataFrame,
                      trainDf2: DataFrame,
                      spark: SparkSession
                    ): (DataFrame, DataFrame) = {

    // === Step 1: Rename 'y' column to 'subscribed' ===
    val trainDf1Renamed = trainDf1.withColumnRenamed("y", "subscribed")
    val trainDf2Renamed = trainDf2.withColumnRenamed("y", "subscribed")

    // === Step 2: Define and order columns based on their types ===
    val categoricalColumns: Array[String] = Array("job", "marital", "education", "contact", "month")
    val binaryColumns: Array[String] = Array("default", "housing")
    val orderedFeatures: Array[String] = Array("job", "marital", "education", "default", "housing", "contact", "month")
    val predictColumn: String = "subscribed"
    val orderedColumns = orderedFeatures :+ predictColumn

    // === Step 3: Select and order columns before using unionByName ===
    val orderedTrainDf1 = trainDf1Renamed.select(orderedColumns.head, orderedColumns.tail: _*)
    val orderedTrainDf2 = trainDf2Renamed.select(orderedColumns.head, orderedColumns.tail: _*)

    // === Step 4: Use unionByName after ordering columns ===
    val orderedTrainData = orderedTrainDf1.unionByName(orderedTrainDf2)

    // === Step 5: Handle Missing Values (remove rows with any missing values) ===
    val cleanedTrainData = orderedTrainData.na.drop()

    // Set the seed for reproducibility
    val seed = 1234L

    // === Step 6: Shuffle and split the DataFrame into training and testing sets ===
    val shuffledData = cleanedTrainData.orderBy(org.apache.spark.sql.functions.rand(seed))
    val Array(trainData, testData) = shuffledData.randomSplit(Array(0.8, 0.2), seed = seed)

    // === Step 7: String Indexing and Vector Assembling ===
    val stringIndexers = categoricalColumns.map { col =>
      new StringIndexer()
        .setInputCol(col)
        .setOutputCol(s"${col}_index")
    }

    val binaryIndexers = binaryColumns.map { col =>
      new StringIndexer()
        .setInputCol(col)
        .setOutputCol(s"${col}_index")
    }

    val predictIndexer = new StringIndexer()
      .setInputCol(predictColumn)
      .setOutputCol("label")
      .setHandleInvalid("skip")  // Skip rows with invalid labels

    val featureCols = categoricalColumns.map(col => s"${col}_index") ++ binaryColumns.map(col => s"${col}_index")
    val vectorAssembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val pipeline = new Pipeline()
      .setStages(stringIndexers ++ binaryIndexers :+ predictIndexer :+ vectorAssembler)

    // === Step 8: Fit and Transform data using the Pipeline ===
    val fittedPipeline = pipeline.fit(trainData)
    val preproTrainDf = fittedPipeline.transform(trainData)
    val preproTestDf = fittedPipeline.transform(testData)

    // === Step 9: Drop unnecessary columns ===
    val finalTrainData = preproTrainDf.drop(orderedColumns.map(colName => s"${colName}_index"): _*)
    val finalTestData = preproTestDf.drop(orderedColumns.map(colName => s"${colName}_index"): _*)

    // === Step 10: Select only the necessary columns ===
    val selectedColumns = Seq("features", "label")
    val finalTrain = finalTrainData.select(selectedColumns.head, selectedColumns.tail: _*)
    val finalTest = finalTestData.select(selectedColumns.head, selectedColumns.tail: _*)

    // === Step 11: Save the preprocessed DataFrames ===
    DataFrameUtils.saveDataFrame(finalTrainData, "C:\\Users\\chand\\Desktop\\ScalaBankInsights\\data_pro\\train\\Processed", "parquet")
    DataFrameUtils.saveDataFrame(finalTestData, "C:\\Users\\chand\\Desktop\\ScalaBankInsights\\data_pro\\test\\Processed", "parquet")
    DataFrameUtils.saveDataFrame(finalTrain, "C:\\Users\\chand\\Desktop\\ScalaBankInsights\\data_pro\\train\\Required", "parquet")
    DataFrameUtils.saveDataFrame(finalTest, "C:\\Users\\chand\\Desktop\\ScalaBankInsights\\data_pro\\test\\Required", "parquet")

    // Set the configuration for eager evaluation
    spark.conf.set("spark.sql.repl.eagerEval.enabled", value = true)

    // Display sample Processed datasets
    println(finalTrainData.show(10, truncate = false))
    println(finalTrain.show(10, truncate = false))
    println(finalTestData.show(10, truncate = false))
    println(finalTest.show(10, truncate = false))

    // === Step 11: Return a tuple of preprocessed DataFrames ===
    (finalTrain, finalTest)
  }
}
