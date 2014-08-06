package mllib.perf.util

import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.random.{RandomDataGenerator, RandomRDDGenerators}
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext


import scala.collection.mutable

object DataGenerator {

  def generateLabeledPoints( sc: SparkContext,
                             numRows: Long,
                             numCols: Int,
                             intercept: Double,
                             eps: Double,
                             numPartitions: Int,
                             seed: Long = System.currentTimeMillis(),
                             problem: String = ""): RDD[LabeledPoint] = {

    RandomRDDGenerators.randomRDD(sc, new LinearDataGenerator(numCols,intercept, seed, eps, problem),
      numRows, numPartitions, seed)

  }

  def generateClassificationLabeledPoints( sc: SparkContext,
                             numRows: Long,
                             numCols: Int,
                             threshold: Double,
                             scaleFactor: Double,
                             numPartitions: Int,
                             seed: Long = System.currentTimeMillis()): RDD[LabeledPoint] = {

    RandomRDDGenerators.randomRDD(sc, new ClassLabelGenerator(numCols,threshold, scaleFactor),
      numRows, numPartitions, seed)
  }

  def generateKMeansVectors( sc: SparkContext,
                             numRows: Long,
                             numCols: Int,
                             numCenters: Int,
                             numPartitions: Int,
                             seed: Long = System.currentTimeMillis()): RDD[Vector] = {

    RandomRDDGenerators.randomRDD(sc, new KMeansDataGenerator(numCenters, numCols, seed), numRows, numPartitions, seed)
  }

  // TODO: Smart generation of synthetic data
  def generateRatings( sc: SparkContext,
                       numUsers: Int,
                       numProducts: Int,
                       numRatings: Long,
                       numPartitions: Int,
                       seed: Long = System.currentTimeMillis()): RDD[Rating] = {

    RandomRDDGenerators.randomRDD(sc, new RatingGenerator(numUsers, numProducts), numRatings, numPartitions, seed)
  }


}

class RatingGenerator(val numUsers: Int,
                      val numProducts: Int) extends RandomDataGenerator[Rating] {

  private val rng = new java.util.Random()
  private val observed = new mutable.HashMap[(Int, Int), Boolean]()

  override def nextValue(): Rating = {
    var tuple: (Int, Int) = (0,0)
    do {
      tuple = (rng.nextInt(numUsers),rng.nextInt(numProducts))
    }while(observed.getOrElse(tuple, false))
    observed += (tuple -> true)

    new Rating(tuple._1, tuple._2, (rng.nextInt(5)+1)*1.0)
  }

  override def setSeed(seed: Long) {
    rng.setSeed(seed)
  }

  override def copy(): RatingGenerator = new RatingGenerator(numUsers, numProducts)
}

// For general classification
class ClassLabelGenerator(private val numFeatures: Int,
                          private val threshold: Double,
                          private val scaleFactor: Double) extends RandomDataGenerator[LabeledPoint] {

  private val rng = new java.util.Random()

  override def nextValue(): LabeledPoint = {
    val y = if (rng.nextDouble() < threshold) 0.0 else 1.0
    val x = Array.fill[Double](numFeatures) {
      rng.nextGaussian() + (y * scaleFactor)
    }

    LabeledPoint(y, Vectors.dense(x))
  }

  override def setSeed(seed: Long) {
    rng.setSeed(seed)
  }

  override def copy(): ClassLabelGenerator = new ClassLabelGenerator(numFeatures, threshold, scaleFactor)
}

class LinearDataGenerator(val numFeatures: Int,
                          val intercept: Double,
                          val seed: Long,
                          val eps: Double,
                          val problem: String = "") extends RandomDataGenerator[LabeledPoint] {

  private val rng = new java.util.Random(seed)

  private val weights = Array.fill(numFeatures)(rng.nextDouble())

  override def nextValue(): LabeledPoint = {
    val x = Array.fill[Double](numFeatures)(2*rng.nextDouble()-1)
    val y = weights.zip(x).map(p => p._1 * p._2).sum + intercept + eps*rng.nextGaussian()
    val yD =
      if (problem == "SVM") {
        if (y < 0.0) 0.0 else 1.0
      }else {
        y
      }

    LabeledPoint(yD, Vectors.dense(x))
  }

  override def setSeed(seed: Long) {
    rng.setSeed(seed)
  }

  override def copy(): LinearDataGenerator = new LinearDataGenerator(numFeatures, intercept, seed, eps, problem)
}

class KMeansDataGenerator(val numCenters: Int,
                          val numColumns: Int,
                          val seed: Long) extends RandomDataGenerator[Vector] {

  private val rng = new java.util.Random(seed)
  private val rng2 = new java.util.Random(seed+24)
  private val scale_factors = Array.fill(numCenters)(rng.nextInt(20)-10)

  // Have a random number of points around a cluster
  private val concentrations: IndexedSeq[Double] = {
    val rand = Array.fill(numCenters)(rng.nextDouble())
    val randSum = rand.sum
    val scaled = rand.map(x => x / randSum)

    (0 until numCenters).map{i =>
      scaled.slice(0, i).sum
    }
  }

  private val centers = (0 until numCenters).map{i =>
    Array.fill(numColumns)((2*rng.nextDouble()-1)*scale_factors(i))
  }

  def binarySearch(target: Double, list: Seq[Double]): Int = {
    var left = 0
    var right = list.length-1
    var mid = 0
    while (left<= right){
      mid = left + (right-left)/2
      if (list(mid)==target || mid==left)
        return mid
      else if (list(mid)>target)
        right = mid-1
      else
        left = mid+1
    }
    mid
  }

  override def nextValue(): Vector = {
    val pick_center_rand = rng2.nextDouble()

    val centerToAddTo = centers(binarySearch(pick_center_rand, concentrations))

    Vectors.dense(Array.tabulate(numColumns)(i => centerToAddTo(i) + rng2.nextGaussian()))
  }

  override def setSeed(seed: Long) {
    rng.setSeed(seed)
  }

  override def copy(): KMeansDataGenerator = new KMeansDataGenerator(numCenters, numColumns, seed)
}

