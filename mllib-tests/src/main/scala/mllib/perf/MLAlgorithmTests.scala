package mllib.perf


import mllib.perf.util.DataGenerator

import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.clustering._
import org.apache.spark.mllib.linalg._
import org.apache.spark.mllib.recommendation.{ALS, Rating}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.configuration.QuantileStrategy
import org.apache.spark.mllib.tree.impurity._
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext


/** Parent class for tests which run on a large dataset. */
abstract class RegressionAndClassificationTests(sc: SparkContext) extends PerfTest {

  def runTest(rdd: RDD[LabeledPoint], numIterations: Int)

  val NUM_EXAMPLES = ("num-examples",   "number of examples for regression tests")
  val NUM_FEATURES = ("num-features",   "number of features of each example for regression tests")

  val intOptions: Seq[(String, String)] = Seq(NUM_TRIALS, INTER_TRIAL_WAIT, NUM_PARTITIONS, RANDOM_SEED,
    NUM_ITERATIONS, NUM_EXAMPLES, NUM_FEATURES)
  val options = intOptions ++ stringOptions  ++ booleanOptions

  intOptions.map{case (opt, desc) =>
    parser.accepts(opt, desc).withRequiredArg().ofType(classOf[Int]).required()
  }

  var rdd: RDD[LabeledPoint] = _

  override def run(): Seq[Double] = {
    val numTrials = intOptionValue(NUM_TRIALS)
    val numIterations = intOptionValue(NUM_ITERATIONS)
    val interTrialWait = intOptionValue(INTER_TRIAL_WAIT)

    val result = (1 to numTrials).map { t =>
      val start = System.currentTimeMillis()
      runTest(rdd, numIterations)
      val end = System.currentTimeMillis()
      val time = (end - start).toDouble / 1000.0
      System.gc()
      Thread.sleep(interTrialWait * 1000)
      time
    }

    result
  }
}

abstract class RegressionTest(sc: SparkContext) extends RegressionAndClassificationTests(sc) {
  override def createInputData() = {
    val numExamples: Long = intOptionValue(NUM_EXAMPLES).toLong
    val numFeatures: Int = intOptionValue(NUM_FEATURES)
    val numPartitions: Int = intOptionValue(NUM_PARTITIONS)
    val randomSeed: Int = intOptionValue(RANDOM_SEED)

    rdd = DataGenerator.generateLabeledPoints(sc, numExamples, numFeatures, numPartitions,randomSeed).cache()

    // Materialize rdd
    println("Num Examples: " + rdd.count())
  }
}

abstract class ClassificationTest(sc: SparkContext) extends RegressionAndClassificationTests(sc) {
  override def createInputData() = {
    val numExamples: Long = intOptionValue(NUM_EXAMPLES).toLong
    val numFeatures: Int = intOptionValue(NUM_FEATURES)
    val numPartitions: Int = intOptionValue(NUM_PARTITIONS)
    val randomSeed: Int = intOptionValue(RANDOM_SEED)

    rdd = DataGenerator.generateClassificationLabeledPoints(sc, numExamples, numFeatures, numPartitions,randomSeed).cache()

    // Materialize rdd
    println("Num Examples: " + rdd.count())
  }
}

/**
 * Parent class for tests which run on a large dataset.
 *
 * This class is specific to [[org.apache.spark.mllib.tree.DecisionTree]].
 * It should eventually be generalized and merged with [[mllib.perf.RegressionAndClassificationTests]]
 */
abstract class DecisionTreeTests(sc: SparkContext) extends PerfTest {

  def runTest(rdd: RDD[LabeledPoint])

  val NUM_EXAMPLES = ("num-examples",   "number of examples for regression tests")
  val NUM_FEATURES = ("num-features",   "number of features of each example for regression tests")
  val LABEL_TYPE =
    ("label-type", "Type of label: 0 indicates regression, 2+ indicates classification with this many classes")
  val FRAC_CATEGORICAL_FEATURES = ("frac-categorical-features", "Fraction of features which are categorical")
  val FRAC_BINARY_FEATURES =
    ("frac-binary-features", "Fraction of categorical features which are binary. Others have 20 categories.")
  val TREE_DEPTH = ("tree-depth", "Depth of true decision tree model used to label examples.")
  val MAX_BINS = ("max-bins", "Maximum number of bins for the decision tree learning algorithm.")

  val intOptions: Seq[(String, String)] = Seq(NUM_TRIALS, INTER_TRIAL_WAIT, NUM_PARTITIONS, RANDOM_SEED,
    NUM_EXAMPLES, NUM_FEATURES, LABEL_TYPE, TREE_DEPTH, MAX_BINS)
  val doubleOptions: Seq[(String, String)] = Seq(FRAC_CATEGORICAL_FEATURES, FRAC_BINARY_FEATURES)
  val options = intOptions ++ stringOptions ++ booleanOptions ++ doubleOptions

  intOptions.map{case (opt, desc) =>
    parser.accepts(opt, desc).withRequiredArg().ofType(classOf[Int]).required()
  }
  doubleOptions.map{case (opt, desc) =>
    parser.accepts(opt, desc).withRequiredArg().ofType(classOf[Double]).required()
  }

  var rdd: RDD[LabeledPoint] = _
  var categoricalFeaturesInfo: Map[Int, Int] = Map.empty

  override def run(): Seq[Double] = {
    val numTrials = intOptionValue(NUM_TRIALS)
    val interTrialWait = intOptionValue(INTER_TRIAL_WAIT)

    val result = (1 to numTrials).map { t =>
      val start = System.currentTimeMillis()
      runTest(rdd)
      val end = System.currentTimeMillis()
      val time = (end - start).toDouble / 1000.0
      System.gc()
      Thread.sleep(interTrialWait * 1000)
      time
    }

    result
  }
}

abstract class RecommendationTests(sc: SparkContext) extends PerfTest {

  def runTest(rdd: RDD[Rating], numIterations: Int, rank: Int)

  val NUM_USERS =     ("num-users",   "number of users for recommendation tests")
  val NUM_PRODUCTS =  ("num-products", "number of features of each example for recommendation tests")
  val NUM_RATINGS =   ("num-ratings",   "number of ratings for recommendation tests")
  val RANK = ("rank",   "rank of factorized matrices for recommendation tests")

  val intOptions = Seq(NUM_TRIALS, INTER_TRIAL_WAIT, NUM_PARTITIONS, RANDOM_SEED,
    NUM_ITERATIONS, NUM_USERS, NUM_PRODUCTS, NUM_RATINGS, RANK)
  val options = intOptions ++ stringOptions  ++ booleanOptions

  intOptions.map{case (opt, desc) =>
    parser.accepts(opt, desc).withRequiredArg().ofType(classOf[Int]).required()
  }

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

  override def run(): Seq[Double] = {
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
      time
    }

    result
  }
}

abstract class ClusteringTests(sc: SparkContext) extends PerfTest {

  def runTest(rdd: RDD[Vector], numIterations: Int, numCenters: Int)

  val NUM_POINTS =    ("num-points",   "number of points for clustering tests")
  val NUM_COLUMNS =   ("num-columns",   "number of columns for each point for clustering tests")
  val NUM_CENTERS =   ("num-centers",   "number of centers for clustering tests")

  val intOptions = Seq(NUM_TRIALS, INTER_TRIAL_WAIT, NUM_PARTITIONS, RANDOM_SEED,
    NUM_ITERATIONS, NUM_POINTS, NUM_CENTERS, NUM_COLUMNS)
  val options = intOptions ++ stringOptions  ++ booleanOptions

  intOptions.map{case (opt, desc) =>
    parser.accepts(opt, desc).withRequiredArg().ofType(classOf[Int]).required()
  }

  var rdd: RDD[Vector] = _

  override def createInputData() = {
    val numPartitions: Int = intOptionValue(NUM_PARTITIONS)
    val randomSeed: Int = intOptionValue(RANDOM_SEED)
    val numPoints: Long = intOptionValue(NUM_POINTS).toLong
    val numColumns: Int = intOptionValue(NUM_COLUMNS)

    rdd = DataGenerator.generateVectors(sc, numPoints, numColumns, numPartitions, randomSeed).cache()

    // Materialize rdd
    println("Num Examples: " + rdd.count())
  }

  override def run(): Seq[Double] = {
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
      time
    }

    result
  }
}

// Regression Algorithms
class LinearRegressionTest(sc: SparkContext) extends RegressionTest(sc) {
  override def runTest(rdd: RDD[LabeledPoint], numIterations: Int) {
    LinearRegressionWithSGD.train(rdd, numIterations)
  }
}

class RidgeRegressionTest(sc: SparkContext) extends RegressionTest(sc) {
  override def runTest(rdd: RDD[LabeledPoint], numIterations: Int) {
    RidgeRegressionWithSGD.train(rdd, numIterations)
  }
}

class LassoTest(sc: SparkContext) extends RegressionTest(sc) {
  override def runTest(rdd: RDD[LabeledPoint], numIterations: Int) {
    LassoWithSGD.train(rdd, numIterations)
  }
}

// Classification Algorithms
class LogisticRegressionTest(sc: SparkContext) extends ClassificationTest(sc) {
  override def runTest(rdd: RDD[LabeledPoint], numIterations: Int) {
    LogisticRegressionWithSGD.train(rdd, numIterations)
  }
}

class NaiveBayesTest(sc: SparkContext) extends ClassificationTest(sc) {
  override def runTest(rdd: RDD[LabeledPoint], numIterations: Int) {
    NaiveBayes.train(rdd)
  }
}

class SVMTest(sc: SparkContext) extends ClassificationTest(sc) {
  override def runTest(rdd: RDD[LabeledPoint], numIterations: Int) {
    SVMWithSGD.train(rdd, numIterations)
  }
}


class DecisionTreeTest(sc: SparkContext) extends DecisionTreeTests(sc) {

  override def createInputData() = {
    // Generic test options
    val randomSeed: Int = intOptionValue(RANDOM_SEED)
    val numPartitions: Int = intOptionValue(NUM_PARTITIONS)
    // Data dimensions and type
    val numExamples: Long = intOptionValue(NUM_EXAMPLES).toLong
    val numFeatures: Int = intOptionValue(NUM_FEATURES)
    val labelType: Int = intOptionValue(LABEL_TYPE)
    val fracCategoricalFeatures: Double = doubleOptionValue(FRAC_CATEGORICAL_FEATURES)
    val fracBinaryFeatures: Double = doubleOptionValue(FRAC_BINARY_FEATURES)
    // Model specification
    val treeDepth: Int = intOptionValue(TREE_DEPTH)

    val (rdd_, categoricalFeaturesInfo_) =
      DataGenerator.generateDecisionTreeLabeledPoints(sc, numExamples, numFeatures, numPartitions,
        labelType, fracCategoricalFeatures, fracBinaryFeatures, treeDepth, randomSeed)
    rdd = rdd_.cache()
    categoricalFeaturesInfo = categoricalFeaturesInfo_

    // Materialize rdd
    println("Num Examples: " + rdd.count())
  }

  override def runTest(rdd: RDD[LabeledPoint]) {
    val labelType: Int = intOptionValue(LABEL_TYPE)
    val treeDepth: Int = intOptionValue(TREE_DEPTH)
    val maxBins: Int = intOptionValue(MAX_BINS)
    if (labelType == 0) {
      // Regression
      DecisionTree.train(rdd, Regression, Variance, treeDepth, 0, maxBins, QuantileStrategy.Sort,
        categoricalFeaturesInfo)
    } else if (labelType >= 2) {
      // Classification
      DecisionTree.train(rdd, Classification, Gini, treeDepth, labelType, maxBins, QuantileStrategy.Sort,
        categoricalFeaturesInfo)
    } else {
      throw new IllegalArgumentException(s"Bad label-type parameter given to DecisionTreeTest: $labelType")
    }
  }
}


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





