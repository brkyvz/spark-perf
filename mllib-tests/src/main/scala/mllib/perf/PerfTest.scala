package mllib.perf

import org.apache.spark.Logging
import joptsimple.{OptionSet, OptionParser}
import org.apache.spark.rdd.RDD

abstract class PerfTest extends Logging {

  val NUM_TRIALS =          ("num-trials",    "number of trials to run")
  val INTER_TRIAL_WAIT =    ("inter-trial-wait",   "seconds to sleep between trials")
  val NUM_PARTITIONS =      ("num-partitions", "number of input partitions")
  val RANDOM_SEED =         ("random-seed", "seed for random number generator")
  val NUM_ITERATIONS =      ("num-iterations",   "number of iterations for the algorithm")

  /** Initialize internal state based on arguments */
  def initialize(testName_ : String, otherArgs: Array[String]) {
    testName = testName_
    optionSet = parser.parse(otherArgs:_*)
  }

  def createInputData()

  /** Runs the test and returns a series of results, along with values of any parameters */
  def run(): Seq[Double]

  val parser = new OptionParser()
  var optionSet: OptionSet = _
  var testName: String = _


  val stringOptions: Seq[(String, String)] = Seq()
  val booleanOptions: Seq[(String, String)] = Seq()

  // add all the options to parser
  stringOptions.map{case (opt, desc) =>
    parser.accepts(opt, desc).withRequiredArg().ofType(classOf[String]).required()
  }
  booleanOptions.map{case (opt, desc) =>
    parser.accepts(opt, desc).withRequiredArg().ofType(classOf[Boolean]).required()
  }

  def intOptionValue(option: (String, String)) =
    optionSet.valueOf(option._1).asInstanceOf[Int]

  def doubleOptionValue(option: (String, String)) =
    optionSet.valueOf(option._1).asInstanceOf[Double]

  def stringOptionValue(option: (String, String)) =
    optionSet.valueOf(option._1).asInstanceOf[String]

  def booleanOptionValue(option: (String, String)) =
    optionSet.valueOf(option._1).asInstanceOf[Boolean]
}
