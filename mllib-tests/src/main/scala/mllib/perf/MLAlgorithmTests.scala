package mllib.perf


import mllib.perf.util.DataGenerator

import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Gini
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext


/** Parent class for tests which run on a large dataset. */
abstract class RegressionAndClassificationTests(sc: SparkContext) extends PerfTest {

  def runTest(rdd: RDD[LabeledPoint], numIterations: Int): GeneralizedLinearModel

  def validate(model: GeneralizedLinearModel, rdd: RDD[LabeledPoint]): Double

  val NUM_EXAMPLES =  ("num-examples",   "number of examples for regression tests")
  val NUM_FEATURES =  ("num-features",   "number of features of each example for regression tests")
  val STEP_SIZE =     ("step-size",   "step size for SGD")

  intOptions = intOptions ++ Seq(NUM_FEATURES)
  longOptions = Seq(NUM_EXAMPLES)
  doubleOptions = doubleOptions ++ Seq(STEP_SIZE)

  var rdd: RDD[LabeledPoint] = _
  var testRdd: RDD[LabeledPoint] = _

  override def run(): Seq[(Double, Double, Double)] = {
    val numTrials = intOptionValue(NUM_TRIALS)
    val numIterations = intOptionValue(NUM_ITERATIONS)
    val interTrialWait = intOptionValue(INTER_TRIAL_WAIT)

    val result = (1 to numTrials).map { t =>
      val start = System.currentTimeMillis()
      val model = runTest(rdd, numIterations)
      val end = System.currentTimeMillis()
      val time = (end - start).toDouble / 1000.0
      val metricOnTrain = validate(model, rdd)
      val metric = validate(model, testRdd)
      System.gc()
      Thread.sleep(interTrialWait * 1000)
      (time, metricOnTrain, metric)
    }

    result
  }
}

abstract class RegressionTest(sc: SparkContext) extends RegressionAndClassificationTests(sc) {

  val INTERCEPT =  ("intercept",   "intercept for random data generation")
  val EPS =  ("epsilon",   "scale factor for the noise during data generation")

  doubleOptions = doubleOptions ++ Seq(INTERCEPT, EPS)

  val options = intOptions ++ stringOptions  ++ booleanOptions ++ doubleOptions ++ longOptions
  addOptionsToParser()
  override def createInputData() = {
    val numExamples: Long = longOptionValue(NUM_EXAMPLES)
    val numFeatures: Int = intOptionValue(NUM_FEATURES)
    val numPartitions: Int = intOptionValue(NUM_PARTITIONS)
    val randomSeed: Int = intOptionValue(RANDOM_SEED)
    val intercept: Double = doubleOptionValue(INTERCEPT)
    val eps: Double = doubleOptionValue(EPS)

    val data = DataGenerator.generateLabeledPoints(sc, math.ceil(numExamples*1.25).toLong,
      numFeatures, intercept, eps, numPartitions,randomSeed)

    val split = data.randomSplit(Array(0.8, 0.2), randomSeed)

    rdd = split(0).cache()
    testRdd = split(1)

    // Materialize rdd
    println("Num Examples: " + rdd.count())
  }

  override def validate(model: GeneralizedLinearModel, rdd: RDD[LabeledPoint]): Double = {
    val numExamples = rdd.count()

    val predictions: RDD[(Double, Double)] = rdd.map { example =>
      (model.predict(example.features), example.label)
    }
    predictions.map{case (pred, label) =>
      (pred-label) * (pred-label)
    }.reduce(_ + _) / numExamples
  }
}

abstract class ClassificationTest(sc: SparkContext) extends RegressionAndClassificationTests(sc) {

  val THRESHOLD =  ("per-negative",   "probability for a negative label during data generation")
  val SCALE =  ("scale-factor",   "scale factor for the noise during data generation")
  val SMOOTHING =     ("nb-lambda",   "the smoothing parameter lambda for Naive Bayes")

  doubleOptions = doubleOptions ++ Seq(THRESHOLD, SCALE, SMOOTHING)

  val options = intOptions ++ stringOptions  ++ booleanOptions ++ doubleOptions ++ longOptions
  addOptionsToParser()

  override def createInputData() = {
    val numExamples: Long = longOptionValue(NUM_EXAMPLES)
    val numFeatures: Int = intOptionValue(NUM_FEATURES)
    val numPartitions: Int = intOptionValue(NUM_PARTITIONS)
    val randomSeed: Int = intOptionValue(RANDOM_SEED)
    val threshold: Double = doubleOptionValue(THRESHOLD)
    val sf: Double = doubleOptionValue(SCALE)

    val data = DataGenerator.generateClassificationLabeledPoints(sc, math.ceil(numExamples*1.25).toLong,
      numFeatures, threshold, sf, numPartitions,randomSeed)

    val split = data.randomSplit(Array(0.8, 0.2), randomSeed)

    rdd = split(0).cache()
    testRdd = split(1)

    // Materialize rdd
    println("Num Examples: " + rdd.count())
  }

  override def validate(model: GeneralizedLinearModel, rdd: RDD[LabeledPoint]): Double = {
    val numExamples = rdd.count()

    val predictions: RDD[(Double, Double)] = rdd.map { example =>
      (model.predict(example.features), example.label)
    }
    predictions.map{case (pred, label) =>
      pred.toByte ^ label.toByte ^ 1
    }.reduce(_ + _) * 100.0 / numExamples
  }

}

abstract class RecommendationTests(sc: SparkContext) extends PerfTest {

  def runTest(rdd: RDD[Rating], numIterations: Int, rank: Int)

  val NUM_USERS =     ("num-users",   "number of users for recommendation tests")
  val NUM_PRODUCTS =  ("num-products", "number of features of each example for recommendation tests")
  val NUM_RATINGS =   ("num-ratings",   "number of ratings for recommendation tests")
  val RANK =          ("rank", "rank of factorized matrices for recommendation tests")

  intOptions = intOptions ++ Seq(NUM_USERS, NUM_PRODUCTS, RANK)
  longOptions = longOptions ++ Seq(NUM_RATINGS)
  val options = intOptions ++ stringOptions  ++ booleanOptions ++ longOptions ++ doubleOptions
  addOptionsToParser()

  var rdd: RDD[Rating] = _

  override def createInputData() = {
    val numPartitions: Int = intOptionValue(NUM_PARTITIONS)
    val randomSeed: Int = intOptionValue(RANDOM_SEED)
    val numUsers: Int = intOptionValue(NUM_USERS)
    val numProducts: Int = intOptionValue(NUM_PRODUCTS)
    val numRatings: Long = intOptionValue(NUM_RATINGS).toLong

    rdd = DataGenerator.generateRatings(sc, numUsers, numProducts, numRatings, numPartitions,randomSeed).cache()

    // Materialize rdd
    println("Num Examples: " + rdd.count())

  }

  override def run(): Seq[(Double, Double, Double)] = {
    val numTrials = intOptionValue(NUM_TRIALS)
    val interTrialWait: Int = intOptionValue(INTER_TRIAL_WAIT)
    val numIterations: Int = intOptionValue(NUM_ITERATIONS)
    val rank: Int = intOptionValue(RANK)

    val result = (1 to numTrials).map { t =>
      val start = System.currentTimeMillis()
      runTest(rdd, numIterations, rank)
      val end = System.currentTimeMillis()
      val time = (end - start).toDouble / 1000.0
      System.gc()
      Thread.sleep(interTrialWait * 1000)
      // TODO: Dummy 1.0 for now
      (time, 1.0, 1.0)
    }

    result
  }
}

abstract class ClusteringTests(sc: SparkContext) extends PerfTest {

  def runTest(rdd: RDD[Vector], numIterations: Int, numCenters: Int)

  val NUM_POINTS =    ("num-points",   "number of points for clustering tests")
  val NUM_COLUMNS =   ("num-columns",   "number of columns for each point for clustering tests")
  val NUM_CENTERS =   ("num-centers",   "number of centers for clustering tests")

  intOptions = intOptions ++ Seq(NUM_CENTERS, NUM_COLUMNS)
  longOptions = longOptions ++ Seq(NUM_POINTS)
  val options = intOptions ++ stringOptions  ++ booleanOptions ++ longOptions ++ doubleOptions
  addOptionsToParser()

  var rdd: RDD[Vector] = _

  override def createInputData() = {
    val numPartitions: Int = intOptionValue(NUM_PARTITIONS)
    val randomSeed: Int = intOptionValue(RANDOM_SEED)
    val numPoints: Long = longOptionValue(NUM_POINTS)
    val numColumns: Int = intOptionValue(NUM_COLUMNS)

    rdd = DataGenerator.generateVectors(sc, numPoints, numColumns, numPartitions, randomSeed).cache()

    // Materialize rdd
    println("Num Examples: " + rdd.count())
  }

  override def run(): Seq[(Double, Double, Double)] = {
    val numTrials = intOptionValue(NUM_TRIALS)
    val interTrialWait: Int = intOptionValue(INTER_TRIAL_WAIT)
    val numIterations: Int = intOptionValue(NUM_ITERATIONS)
    val k: Int = intOptionValue(NUM_CENTERS)

    val result = (1 to numTrials).map { t =>
      val start = System.currentTimeMillis()
      runTest(rdd, numIterations, k)
      val end = System.currentTimeMillis()
      val time = (end - start).toDouble / 1000.0
      System.gc()
      Thread.sleep(interTrialWait * 1000)
      // TODO: Dummy 1.0 for now
      (time, 1.0, 1.0)
    }

    result
  }
}

// Regression Algorithms
class LinearRegressionTest(sc: SparkContext) extends RegressionTest(sc) {
  override def runTest(rdd: RDD[LabeledPoint], numIterations: Int): LinearRegressionModel = {
    val stepSize = doubleOptionValue(STEP_SIZE)
    LinearRegressionWithSGD.train(rdd, numIterations, stepSize)
  }
}

class RidgeRegressionTest(sc: SparkContext) extends RegressionTest(sc) {
  override def runTest(rdd: RDD[LabeledPoint], numIterations: Int): RidgeRegressionModel = {
    val stepSize = doubleOptionValue(STEP_SIZE)
    val regParam = doubleOptionValue(REGULARIZATION)

    RidgeRegressionWithSGD.train(rdd, numIterations, stepSize, regParam)
  }
}

class LassoTest(sc: SparkContext) extends RegressionTest(sc) {
  override def runTest(rdd: RDD[LabeledPoint], numIterations: Int): LassoModel = {
    val stepSize = doubleOptionValue(STEP_SIZE)
    val regParam = doubleOptionValue(REGULARIZATION)

    LassoWithSGD.train(rdd, numIterations, stepSize, regParam)
  }
}

// Classification Algorithms
class LogisticRegressionTest(sc: SparkContext) extends ClassificationTest(sc) {
  override def runTest(rdd: RDD[LabeledPoint], numIterations: Int): LogisticRegressionModel = {
    val stepSize = doubleOptionValue(STEP_SIZE)

    LogisticRegressionWithSGD.train(rdd, numIterations, stepSize)
  }
}
/*
class NaiveBayesTest(sc: SparkContext) extends ClassificationTest(sc) {
  override def runTest(rdd: RDD[LabeledPoint], numIterations: Int): NaiveBayesModel = {
    val lambda = doubleOptionValue(SMOOTHING)

    NaiveBayes.train(rdd, lambda)
  }
}
*/
class SVMTest(sc: SparkContext) extends ClassificationTest(sc) {
  override def runTest(rdd: RDD[LabeledPoint], numIterations: Int): SVMModel = {
    val stepSize = doubleOptionValue(STEP_SIZE)
    val regParam = doubleOptionValue(REGULARIZATION)

    SVMWithSGD.train(rdd, numIterations, stepSize, regParam)
  }
}

// TODO: Joseph, it might be best to create a separate class than ClassificationTest to get more parameters in for Strategy
// Decision Trees
/*
class DecisionTreeTest(sc: SparkContext) extends ClassificationTest(sc) {
  override def runTest(rdd: RDD[LabeledPoint], numIterations: Int) {
    DecisionTree.train(rdd,new Strategy(Classification, Gini, 4))
  }
}
*/
// Recommendation
class ALSTest(sc: SparkContext) extends RecommendationTests(sc) {
  override def runTest(rdd: RDD[Rating], numIterations: Int, rank: Int) {
    ALS.train(rdd, rank, numIterations)
  }
}

// Clustering
class KMeansTest(sc: SparkContext) extends ClusteringTests(sc) {
  override def runTest(rdd: RDD[Vector], numIterations: Int, numCenters: Int) {
    KMeans.train(rdd, numCenters, numIterations)
  }
}





