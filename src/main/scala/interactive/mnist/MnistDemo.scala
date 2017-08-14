/*
 * Copyright (c) 2017 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package interactive.mnist

import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicInteger

import _root_.util.{MindsEyeNotebook, _}
import com.simiacryptus.mindseye.layers.NNLayer
import com.simiacryptus.mindseye.layers.activation._
import com.simiacryptus.mindseye.layers.loss.EntropyLossLayer
import com.simiacryptus.mindseye.layers.synapse.{BiasLayer, DenseSynapseLayer}
import com.simiacryptus.mindseye.layers.util.MonitoringWrapper
import com.simiacryptus.mindseye.network.{PipelineNetwork, SimpleLossNetwork, SupervisedNetwork}
import com.simiacryptus.mindseye.opt.trainable.StochasticArrayTrainable
import com.simiacryptus.mindseye.opt.orient.LBFGS
import com.simiacryptus.mindseye.opt.orient.{GradientDescent, LBFGS}
import com.simiacryptus.util.StreamNanoHTTPD
import com.simiacryptus.util.io.{HtmlNotebookOutput, MarkdownNotebookOutput}
import com.simiacryptus.util.ml.Tensor
import com.simiacryptus.util.test.MNIST
import com.simiacryptus.util.text.TableOutput

import scala.collection.JavaConverters._
import scala.util.Random


object MnistDemo extends Report {

  def main(args: Array[String]): Unit = {
    HtmlNotebookOutput.DEFAULT_ROOT = "https://github.com/SimiaCryptus/mindseye-scala/tree/master/"
    report((s,log)⇒new MnistDemo(s,log).run)
    System.exit(0)
  }
}

class MnistDemo(server: StreamNanoHTTPD, log: HtmlNotebookOutput with ScalaNotebookOutput) extends MindsEyeNotebook(server, log) {

  def run {
    defineHeader()
    log.p("In this demo we train a simple neural network against the MNIST handwritten digit dataset")
    phase1()
    phase2()
    validateModel(log, model)
    waitForExit()
  }

  val inputSize = Array[Int](28, 28, 1)
  val outputSize = Array[Int](10)
  val iterationCounter = new AtomicInteger(0)

  lazy val trainingData = {
    log.p("Load the MNIST training dataset: ")
    val data: Seq[Array[Tensor]] = log.eval {
      MNIST.trainingDataStream().iterator().asScala.toStream.map(labeledObj ⇒ {
        Array(labeledObj.data, toOutNDArray(toOut(labeledObj.label), 10))
      })
    }
    log.p("Here is a sample of the data:")
    log.eval {
      TableOutput.create(data.take(10).map(testObj ⇒ Map[String, AnyRef](
        "Input1 (as Image)" → log.image(testObj(0).toGrayImage(), testObj(1).toString),
        "Input2 (as String)" → testObj(1).toString
      ).asJava): _*)
    }
    data
  }

  def phase1() = phase({
    log.p("We construct a new model:")
    log.eval {
      def wrap(n:NNLayer) = new MonitoringWrapper(n).addTo(monitoringRoot)
      var model: PipelineNetwork = new PipelineNetwork
      model.add(wrap(new BiasLayer(inputSize: _*).setName("inbias")))
      model.add(wrap(new DenseSynapseLayer(inputSize, outputSize)
        .setWeights(Java8Util.cvt(() ⇒ 0.001 * (Random.nextDouble() - 0.5))).setName("synapse")))
      model.add(wrap(new ReLuActivationLayer().setName("relu")))
      model.add(wrap(new BiasLayer(outputSize: _*).setName("outbias")))
      model.add(new SoftmaxActivationLayer)
      model.asInstanceOf[NNLayer]
    }
  }, (model: NNLayer) ⇒ {
    log.p("The model is pre-trained map some data before being saved:")
    log.eval {
      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new EntropyLossLayer)
      val trainable = new StochasticArrayTrainable(trainingData.toArray, trainingNetwork, 1000)
      val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
      trainer.setMonitor(monitor)
      trainer.setCurrentIteration(iterationCounter)
      trainer.setOrientation(new GradientDescent);
      trainer.setTimeout(1, TimeUnit.MINUTES)
      trainer.setTerminateThreshold(0.0)
      trainer.run()
    }
  }, "mnist_initialized")

  def phase2() = phase("mnist_initialized", (model: NNLayer) ⇒ {
    log.p("A second phase of training:")
    log.eval {
      val trainingNetwork: SupervisedNetwork = new SimpleLossNetwork(model, new EntropyLossLayer)
      val trainable = new StochasticArrayTrainable(trainingData.toArray, trainingNetwork, 10000)
      val trainer = new com.simiacryptus.mindseye.opt.IterativeTrainer(trainable)
      trainer.setMonitor(monitor)
      trainer.setCurrentIteration(iterationCounter)
      trainer.setOrientation(new LBFGS);
      trainer.setTimeout(5, TimeUnit.MINUTES)
      trainer.setTerminateThreshold(0.0)
      trainer.run()
    }
  }, "mnist_trained")

  override def defineReports(log: HtmlNotebookOutput with ScalaNotebookOutput) = {
    log.p("Interactive Reports: <a href='/history.html'>Convergence History</a> <a href='/test.html'>Model Validation</a>")
    server.addSyncHandler("test.html", "text/html", Java8Util.cvt(out ⇒ {
      Option(new HtmlNotebookOutput(log.workingDir, out) with ScalaNotebookOutput).foreach(log ⇒ {
        validateModel(log, getModelCheckpoint)
      })
    }), false)
  }

  def validateModel(log: HtmlNotebookOutput with ScalaNotebookOutput, model: NNLayer) = {
    log.h2("Validation")
    log.p("Here we examine a sample of validation rows, randomly selected: ")
    log.eval {
      TableOutput.create(MNIST.validationDataStream().iterator().asScala.toStream.take(10).map(testObj ⇒ {
        val result = model.eval(new NNLayer.NNExecutionContext() {}, testObj.data).getData.get(0)
        Map[String, AnyRef](
          "Input" → log.image(testObj.data.toGrayImage(), testObj.label),
          "Predicted Label" → (0 to 9).maxBy(i ⇒ result.get(i)).asInstanceOf[Integer],
          "Actual Label" → testObj.label,
          "Network Output" → result
        ).asJava
      }): _*)
    }
    log.p("Validation rows that are mispredicted are also sampled: ")
    log.eval {
      TableOutput.create(MNIST.validationDataStream().iterator().asScala.toStream.filterNot(testObj ⇒ {
        val result = model.eval(new NNLayer.NNExecutionContext() {}, testObj.data).getData.get(0)
        val prediction: Int = (0 to 9).maxBy(i ⇒ result.get(i))
        val actual = toOut(testObj.label)
        prediction == actual
      }).take(10).map(testObj ⇒ {
        val result = model.eval(new NNLayer.NNExecutionContext() {}, testObj.data).getData.get(0)
        Map[String, AnyRef](
          "Input" → log.image(testObj.data.toGrayImage(), testObj.label),
          "Predicted Label" → (0 to 9).maxBy(i ⇒ result.get(i)).asInstanceOf[Integer],
          "Actual Label" → testObj.label,
          "Network Output" → result
        ).asJava
      }): _*)
    }
    log.p("To summarize the accuracy of the model, we calculate several summaries: ")
    log.p("The (mis)categorization matrix displays a count matrix for every actual/predicted category: ")
    val categorizationMatrix: Map[Int, Map[Int, Int]] = log.eval {
      MNIST.validationDataStream().iterator().asScala.toStream.map(testObj ⇒ {
        val result = model.eval(new NNLayer.NNExecutionContext() {}, testObj.data).getData.get(0)
        val prediction: Int = (0 to 9).maxBy(i ⇒ result.get(i))
        val actual: Int = toOut(testObj.label)
        actual → prediction
      }).groupBy(_._1).mapValues(_.groupBy(_._2).mapValues(_.size))
    }
    writeMislassificationMatrix(log, categorizationMatrix)
    log.out("")
    log.p("The accuracy, summarized per category: ")
    log.eval {
      (0 to 9).map(actual ⇒ {
        actual → (categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(actual, 0) * 100.0 / categorizationMatrix.getOrElse(actual, Map.empty).values.sum)
      }).toMap
    }
    log.p("The accuracy, summarized over the entire validation set: ")
    log.eval {
      (0 to 9).map(actual ⇒ {
        categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(actual, 0)
      }).sum.toDouble * 100.0 / categorizationMatrix.values.flatMap(_.values).sum
    }
  }

  def writeMislassificationMatrix(log: HtmlNotebookOutput, categorizationMatrix: Map[Int, Map[Int, Int]]) = {
    log.out("<table>")
    log.out("<tr>")
    log.out((List("Actual \\ Predicted | ") ++ (0 to 9)).map("<td>"+_+"</td>").mkString(""))
    log.out("</tr>")
    (0 to 9).foreach(actual ⇒ {
      log.out("<tr>")
      log.out(s"<td>$actual</td>" + (0 to 9).map(prediction ⇒ {
        categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(prediction, 0)
      }).map("<td>"+_+"</td>").mkString(""))
      log.out("</tr>")
    })
    log.out("</table>")
  }

  def writeMislassificationMatrix(log: MarkdownNotebookOutput, categorizationMatrix: Map[Int, Map[Int, Int]]) = {
    log.out("Actual \\ Predicted | " + (0 to 9).mkString(" | "))
    log.out((0 to 10).map(_ ⇒ "---").mkString(" | "))
    (0 to 9).foreach(actual ⇒ {
      log.out(s" **$actual** | " + (0 to 9).map(prediction ⇒ {
        categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(prediction, 0)
      }).mkString(" | "))
    })
  }

  def toOut(label: String): Int = {
    (0 until 10).find(label == "[" + _ + "]").get
  }

  def toOutNDArray(out: Int, max: Int): Tensor = {
    val ndArray = new Tensor(max)
    ndArray.set(out, 1)
    ndArray
  }

}