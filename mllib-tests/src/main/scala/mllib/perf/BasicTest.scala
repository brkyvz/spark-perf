package mllib.perf

class BasicTest extends PerfTest {

  override def createInputData() {}

  def run(): Seq[Double] =  {
    println("Running BasicTest.run()")
    Array(0.0)
  }
}
