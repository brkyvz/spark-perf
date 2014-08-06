package mllib.perf


import mllib.perf.util.DataGenerator

import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.recommendation.{MatrixFactorizationModel, ALS, Rating}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Gini
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext

import scala.reflect.ClassTag


/** Parent class for tests which run on a large dataset. */
abstract class RegressionAndClassificationTests[M](sc: SparkContext) extends PerfTest {

  def runTest(rdd: RDD[LabeledPoint], numIterations: Int): M

  def validate(model: M, rdd: RDD[LabeledPoint]): Double

  val NUM_EXAMPLES =  ("num-examples",   "number of examples for regression tests")
  val NUM_FEATURES =  ("num-features",   "number of features of each example for regression tests")
  val STEP_SIZE =     ("step-size",   "step size for SGD")

  intOptions = intOptions ++ Seq(NUM_FEATURES)
  longOptions = Seq(NUM_EXAMPLES)
  doubleOptions = doubleOptions ++ Seq(STEP_SIZE)

  var rdd: RDD[LabeledPoint] = _
  var testRdd: RDD[LabeledPoint] = _

}

abstract class RegressionTest(sc: SparkContext) extends RegressionAndClassificationTests[GeneralizedLinearModel](sc) {

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
    val error = predictions.map{case (pred, label) =>
      (pred-label) * (pred-label)
    }.reduce(_ + _)

    math.sqrt(error / numExamples)
  }

  override def run(): Seq[(Double, Double, Double)] = {
    val numTrials = intOptionValue(NUM_TRIALS)
    val numIterations = intOptionValue(NUM_ITERATIONS)
    val interTrialWait = intOptionValue(INTER_TRIAL_WAIT)
    val intercept = doubleOptionValue(INTERCEPT)

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

abstract class ClassificationTest[M](sc: SparkContext) extends RegressionAndClassificationTests[M](sc) {

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

  def calculateAccuracy(predictions: RDD[(Double, Double)], numExamples: Long): Double = {
    predictions.map{case (pred, label) =>
      pred.toByte ^ label.toByte ^ 1
    }.reduce(_ + _) * 100.0 / numExamples
  }

}

abstract class RecommendationTests(sc: SparkContext) extends PerfTest {

  def runTest(rdd: RDD[Rating], numIterations: Int, rank: Int): MatrixFactorizationModel



  val NUM_USERS =     ("num-users",   "number of users for recommendation tests")
  val NUM_PRODUCTS =  ("num-products", "number of features of each example for recommendation tests")
  val NUM_RATINGS =   ("num-ratings",   "number of ratings for recommendation tests")
  val RANK =          ("rank", "rank of factorized matrices for recommendation tests")

  intOptions = intOptions ++ Seq(NUM_USERS, NUM_PRODUCTS, RANK)
  longOptions = longOptions ++ Seq(NUM_RATINGS)
  val options = intOptions ++ stringOptions  ++ booleanOptions ++ longOptions ++ doubleOptions
  addOptionsToParser()

  var rdd: RDD[Rating] = _
  var testRdd: RDD[Rating] = _

  override def createInputData() = {
    val numPartitions: Int = intOptionValue(NUM_PARTITIONS)
    val randomSeed: Int = intOptionValue(RANDOM_SEED)
    val numUsers: Int = intOptionValue(NUM_USERS)
    val numProducts: Int = intOptionValue(NUM_PRODUCTS)
    val numRatings: Long = longOptionValue(NUM_RATINGS)

    val data = DataGenerator.generateRatings(sc, numUsers, numProducts, math.ceil(numRatings*1.25).toLong,
      numPartitions,randomSeed)

    rdd = data._1.cache()
    testRdd = data._2

    // Materialize rdd
    println("Num Examples: " + rdd.count())

  }

  def validate(model: MatrixFactorizationModel, rdd: RDD[Rating]): Double = {
    val numRatings = rdd.cache().count()

    val error = rdd.map{ rating =>
      val prediction = model.predict(rating.user, rating.product)
      (prediction - rating.rating) * (prediction - rating.rating)
    }.reduce(_ + _)

    math.sqrt(error/numRatings)
  }

  override def run(): Seq[(Double, Double, Double)] = {
    val numTrials = intOptionValue(NUM_TRIALS)
    val interTrialWait: Int = intOptionValue(INTER_TRIAL_WAIT)
    val numIterations: Int = intOptionValue(NUM_ITERATIONS)
    val rank: Int = intOptionValue(RANK)

    val result = (1 to numTrials).map { t =>
      val start = System.currentTimeMillis()
      val model = runTest(rdd, numIterations, rank)
      val end = System.currentTimeMillis()
      val time = (end - start).toDouble / 1000.0

      val trainError = validate(model, rdd)
      val testError = validate(model, testRdd)
      System.gc()
      Thread.sleep(interTrialWait * 1000)

      (time, trainError, testError)
    }

    result
  }
}

abstract class ClusteringTests(sc: SparkContext) extends PerfTest {

  def runTest(rdd: RDD[Vector], numIterations: Int, numCenters: Int): KMeansModel

  val NUM_POINTS =    ("num-points",   "number of points for clustering tests")
  val NUM_COLUMNS =   ("num-columns",   "number of columns for each point for clustering tests")
  val NUM_CENTERS =   ("num-centers",   "number of centers for clustering tests")

  intOptions = intOptions ++ Seq(NUM_CENTERS, NUM_COLUMNS)
  longOptions = longOptions ++ Seq(NUM_POINTS)
  val options = intOptions ++ stringOptions  ++ booleanOptions ++ longOptions ++ doubleOptions
  addOptionsToParser()

  var rdd: RDD[Vector] = _
  var testRdd: RDD[Vector] = _

  def validate(model: KMeansModel, rdd: RDD[Vector]): Double = {
    val numPoints = rdd.cache().count()

    val error = model.computeCost(rdd)

    math.sqrt(error/numPoints)
  }

  override def createInputData() = {
    val numPartitions: Int = intOptionValue(NUM_PARTITIONS)
    val randomSeed: Int = intOptionValue(RANDOM_SEED)
    val numPoints: Long = longOptionValue(NUM_POINTS)
    val numColumns: Int = intOptionValue(NUM_COLUMNS)
    val numCenters: Int = intOptionValue(NUM_CENTERS)

    val data = DataGenerator.generateKMeansVectors(sc, math.ceil(numPoints*1.25).toLong, numColumns,
      numCenters, numPartitions, randomSeed)

    val split = data.randomSplit(Array(0.8, 0.2), randomSeed)

    rdd = split(0).cache()
    testRdd = split(1)

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
      val model = runTest(rdd, numIterations, k)
      val end = System.currentTimeMillis()
      val time = (end - start).toDouble / 1000.0
      val trainError = validate(model, rdd)
      val testError = validate(model, testRdd)
      System.gc()
      Thread.sleep(interTrialWait * 1000)

      (time, trainError, testError)
    }

    result
  }
}

// Regression Algorithms
class LinearRegressionTest(sc: SparkContext) extends RegressionTest(sc) {
  override def runTest(rdd: RDD[LabeledPoint], numIterations: Int): LinearRegressionModel = {
    val stepSize = doubleOptionValue(STEP_SIZE)
    val lr = new LinearRegressionWithSGD().setIntercept(true)
    lr.optimizer.setNumIterations(numIterations).setStepSize(stepSize)

    lr.run(rdd)
  }
}

class RidgeRegressionTest(sc: SparkContext) extends RegressionTest(sc) {
  override def runTest(rdd: RDD[LabeledPoint], numIterations: Int): RidgeRegressionModel = {
    val stepSize = doubleOptionValue(STEP_SIZE)
    val regParam = doubleOptionValue(REGULARIZATION)
    val rr = new RidgeRegressionWithSGD().setIntercept(true)
    rr.optimizer.setNumIterations(numIterations).setStepSize(stepSize).setRegParam(regParam)

    rr.run(rdd)
  }
}

class LassoTest(sc: SparkContext) extends RegressionTest(sc) {
  override def runTest(rdd: RDD[LabeledPoint], numIterations: Int): LassoModel = {
    val stepSize = doubleOptionValue(STEP_SIZE)
    val regParam = doubleOptionValue(REGULARIZATION)
    val lasso = new LassoWithSGD().setIntercept(true)
    lasso.optimizer.setNumIterations(numIterations).setStepSize(stepSize).setRegParam(regParam)

    lasso.run(rdd)
  }
}

// Classification Algorithms
class LogisticRegressionTest(sc: SparkContext) extends ClassificationTest[LogisticRegressionModel](sc) {

  override def validate(model: LogisticRegressionModel, rdd: RDD[LabeledPoint]): Double = {
    val numExamples = rdd.count()

    val predictions: RDD[(Double, Double)] = rdd.map { example =>
      (model.predict(example.features), example.label)
    }
    calculateAccuracy(predictions, numExamples)
  }

  override def runTest(rdd: RDD[LabeledPoint], numIterations: Int): LogisticRegressionModel = {
    val stepSize = doubleOptionValue(STEP_SIZE)

    LogisticRegressionWithSGD.train(rdd, numIterations, stepSize)
  }
}

class NaiveBayesTest(sc: SparkContext) extends ClassificationTest[NaiveBayesModel](sc) {

  override def validate(model: NaiveBayesModel, rdd: RDD[LabeledPoint]): Double = {
    val numExamples = rdd.count()

    val predictions: RDD[(Double, Double)] = rdd.map { example =>
      (model.predict(example.features), example.label)
    }
    calculateAccuracy(predictions, numExamples)
  }

  override def runTest(rdd: RDD[LabeledPoint], numIterations: Int): NaiveBayesModel = {
    val lambda = doubleOptionValue(SMOOTHING)

    NaiveBayes.train(rdd, lambda)
  }
}

class SVMTest(sc: SparkContext) extends ClassificationTest[SVMModel](sc) {

  override def validate(model: SVMModel, rdd: RDD[LabeledPoint]): Double = {
    val numExamples = rdd.count()

    val predictions: RDD[(Double, Double)] = rdd.map { example =>
      (model.predict(example.features), example.label)
    }
    calculateAccuracy(predictions, numExamples)
  }

  override def createInputData() = {
    val numExamples: Long = longOptionValue(NUM_EXAMPLES)
    val numFeatures: Int = intOptionValue(NUM_FEATURES)
    val numPartitions: Int = intOptionValue(NUM_PARTITIONS)
    val randomSeed: Int = intOptionValue(RANDOM_SEED)
    val sf: Double = doubleOptionValue(SCALE)

    val data = DataGenerator.generateLabeledPoints(sc, math.ceil(numExamples*1.25).toLong,
      numFeatures, 0.0, sf, numPartitions,randomSeed, "SVM")

    val split = data.randomSplit(Array(0.8, 0.2), randomSeed)

    rdd = split(0).cache()
    testRdd = split(1)

    // Materialize rdd
    println("Num Examples: " + rdd.count())
  }

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
  override def runTest(rdd: RDD[Rating], numIterations: Int, rank: Int): MatrixFactorizationModel = {
    val regParam = doubleOptionValue(REGULARIZATION)

    ALS.train(rdd, rank, numIterations, regParam)
  }
}

// Clustering
class KMeansTest(sc: SparkContext) extends ClusteringTests(sc) {
  override def runTest(rdd: RDD[Vector], numIterations: Int, numCenters: Int): KMeansModel = {
    KMeans.train(rdd, numCenters, numIterations)
  }
}





