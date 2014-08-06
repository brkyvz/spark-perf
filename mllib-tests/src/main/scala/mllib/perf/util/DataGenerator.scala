package mllib.perf.util

import com.github.fommil.netlib.BLAS.{getInstance => blas}

import org.apache.spark.mllib.linalg.{DenseMatrix, Vectors, Vector}
import org.apache.spark.mllib.random.{RandomDataGenerator, RandomRDDGenerators}
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.{PairRDDFunctions, RDD}
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


  // Problems with having a userID or productID in the test set but not training set leads to a lot of work...
  def generateRatings( sc: SparkContext,
                       numUsers: Int,
                       numProducts: Int,
                       numRatings: Long,
                       implicitPrefs: Boolean,
                       numPartitions: Int,
                       seed: Long = System.currentTimeMillis()): (RDD[Rating],RDD[Rating]) = {

    val train = RandomRDDGenerators.randomRDD(sc, new RatingGenerator(numUsers, numProducts,implicitPrefs),
       numRatings, numPartitions, seed).cache()

    val test = RandomRDDGenerators.randomRDD(sc, new RatingGenerator(numUsers, numProducts,implicitPrefs),
      math.ceil(numRatings*0.25).toLong, numPartitions, seed+24)

    // Now get rid of duplicate ratings and remove non-existant userID's and prodID's from the test set
    val commons: PairRDDFunctions[(Int,Int),Rating] = new PairRDDFunctions(train.keyBy(rating => (rating.user, rating.product)).cache())

    val exact = commons.join(test.keyBy(rating => (rating.user, rating.product)))

    val trainPruned = commons.subtractByKey(exact).map(_._2).cache()

    // Now get rid of users that don't exist in the train set
    val trainUsers: RDD[(Int,Rating)] = trainPruned.keyBy(rating => rating.user)
    val testUsers: PairRDDFunctions[Int,Rating] = new PairRDDFunctions(test.keyBy(rating => rating.user))
    val testWithAdditionalUsers = testUsers.subtractByKey(trainUsers)

    val userPrunedTestProds: RDD[(Int,Rating)] = testUsers.subtractByKey(testWithAdditionalUsers).map(_._2).keyBy(rating => rating.product)

    val trainProds: RDD[(Int,Rating)] = trainPruned.keyBy(rating => rating.product)

    val testWithAdditionalProds = new PairRDDFunctions[Int, Rating](userPrunedTestProds).subtractByKey(trainProds)
    val finalTest = new PairRDDFunctions[Int, Rating](userPrunedTestProds).subtractByKey(testWithAdditionalProds).map(_._2)

    (trainPruned, finalTest)
  }


}

// For general classification
class RatingGenerator( private val numUsers: Int,
                       private val numProducts: Int,
                       private val implicitPrefs: Boolean) extends RandomDataGenerator[Rating] {

  private val rng = new java.util.Random()

  private val observed = new mutable.HashMap[(Int, Int), Boolean]()

  override def nextValue(): Rating = {
    var tuple = (rng.nextInt(numUsers),rng.nextInt(numProducts))
    while (observed.getOrElse(tuple,false)){
      tuple = (rng.nextInt(numUsers),rng.nextInt(numProducts))
    }
    observed += (tuple -> true)

    val rating = if (implicitPrefs) rng.nextInt(2)*1.0 else rng.nextDouble()*5

    new Rating(tuple._1, tuple._2, rating)
  }

  override def setSeed(seed: Long) {
    rng.setSeed(seed)
  }

  override def copy(): RatingGenerator = new RatingGenerator(numUsers, numProducts, implicitPrefs)
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

