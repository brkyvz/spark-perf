package mllib.perf.util

import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.random.{RandomDataGenerator, RandomRDDGenerators}
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext

import scala.collection.mutable

object DataGenerator {

  // TODO: Smart generation of synthetic data
  def generateLabeledPoints( sc: SparkContext,
                             numRows: Long,
                             numCols: Int,
                             numPartitions: Int,
                             seed: Long = System.currentTimeMillis()): RDD[LabeledPoint] = {

    val featureMatrix = RandomRDDGenerators.uniformVectorRDD(sc, numRows, numCols, numPartitions, seed)
    val labelMatrix = RandomRDDGenerators.normalRDD(sc, numRows, numPartitions, seed)

    labelMatrix.zip(featureMatrix).map(pair => new LabeledPoint(pair._1, pair._2))
  }

  def generateClassificationLabeledPoints( sc: SparkContext,
                             numRows: Long,
                             numCols: Int,
                             numPartitions: Int,
                             seed: Long = System.currentTimeMillis()): RDD[LabeledPoint] = {

    val featureMatrix = RandomRDDGenerators.uniformVectorRDD(sc, numRows, numCols, numPartitions, seed)
    val labelMatrix = RandomRDDGenerators.randomRDD(sc, new ClassLabelGenerator, numRows, numPartitions, seed)

    labelMatrix.zip(featureMatrix).map(pair => new LabeledPoint(pair._1, pair._2))
  }

  def generateVectors( sc: SparkContext,
                             numRows: Long,
                             numCols: Int,
                             numPartitions: Int,
                             seed: Long = System.currentTimeMillis()): RDD[Vector] = {

    RandomRDDGenerators.uniformVectorRDD(sc, numRows, numCols, numPartitions, seed)
  }

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

  private val rng = new scala.util.Random()
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

class ClassLabelGenerator extends RandomDataGenerator[Double] {

  private val rng = new scala.util.Random()

  override def nextValue(): Double = if (rng.nextBoolean()) 1.0 else -1.0

  override def setSeed(seed: Long) {
    rng.setSeed(seed)
  }

  override def copy(): ClassLabelGenerator = new ClassLabelGenerator()
}
