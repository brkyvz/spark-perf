package mllib.perf

import mllib.perf.util.DataGenerator
import org.apache.spark.mllib.linalg.{Matrices, Vectors, Matrix, Vector}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.random.RandomRDDGenerators
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.rdd.RDD

import scala.util.Random

/** Parent class for the tests for the statistics toolbox
  */
abstract class StatTests(sc: SparkContext) extends PerfTest {

  def runTest(rdd: RDD[Vector])

  val NUM_ROWS =  ("num-rows",   "number of rows of the matrix")
  val NUM_COLS =  ("num-cols",   "number of columns of the matrix")

  longOptions = Seq(NUM_ROWS)
  intOptions = intOptions ++ Seq(NUM_COLS)

  var rdd: RDD[Vector] = _
  var trainingSet: RDD[LabeledPoint] = _
  var observed: Vector = _
  var counts: Matrix = _

  val options = intOptions ++ stringOptions  ++ booleanOptions ++ doubleOptions ++ longOptions
  addOptionsToParser()
  override def createInputData(seed: Long) = {
    val m: Long = longOptionValue(NUM_ROWS)
    val n: Int = intOptionValue(NUM_COLS)
    val numPartitions: Int = intOptionValue(NUM_PARTITIONS)
    val rng = new Random()

    rdd = RandomRDDGenerators.normalVectorRDD(sc, m, n, numPartitions, seed)
    trainingSet = DataGenerator.generateClassificationLabeledPoints(sc, m, n, 0.5, 1.0, numPartitions)
    observed = Vectors.dense(Array.fill(m.toInt)(rng.nextDouble()))
    counts = Matrices.dense(m.toInt, m.toInt, Array.fill(m.toInt * m.toInt)(rng.nextDouble()))
  }

  override def run(): (Double, Double, Double) = {

    val start = System.currentTimeMillis()
    runTest(rdd)
    val end = System.currentTimeMillis()
    val time = (end - start).toDouble / 1000.0

    (time, 0.0, 0.0)
  }

}


class PearsonCorrelationTest(sc: SparkContext) extends StatTests(sc) {
  // data's never used?
  override def runTest(data: RDD[Vector]) {
     Statistics.corr(rdd)
  }
}

class SpearmanCorrelationTest(sc: SparkContext) extends StatTests(sc) {
  override def runTest(data: RDD[Vector]) {
    Statistics.corr(rdd, "spearman")
  }
}

class ChiSquaredFeatureTest(sc: SparkContext) extends StatTests(sc) {
  override def runTest(data: RDD[Vector]) {
    Statistics.chiSqTest(trainingSet)
  }
}

class ChiSquaredGoFTest(sc: SparkContext) extends StatTests(sc) {
  override def runTest(data: RDD[Vector]) {
    Statistics.chiSqTest(observed)
  }
}

class ChiSquaredMatTest(sc: SparkContext) extends StatTests(sc) {
  override def runTest(data: RDD[Vector]) {
    Statistics.chiSqTest(counts)
  }
}
