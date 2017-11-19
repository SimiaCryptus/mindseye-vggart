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

package report

import java.awt.Color
import java.lang

import _root_.util.Java8Util._
import _root_.util.{ReportNotebook, ScalaNotebookOutput}
import com.simiacryptus.mindseye.data._
import com.simiacryptus.mindseye.lang._
import com.simiacryptus.mindseye.network._
import com.simiacryptus.mindseye.opt._
import com.simiacryptus.text.TableOutput
import org.scalatest.{MustMatchers, WordSpec}
import smile.plot.{PlotCanvas, ScatterPlot}

import scala.collection.JavaConverters._
import scala.collection.mutable

class ConvAutoencoderDemo extends WordSpec with MustMatchers with ReportNotebook {

  var data: TensorList = null
  val history = new mutable.ArrayBuffer[com.simiacryptus.mindseye.opt.Step]
  var monitor = new TrainingMonitor {
    override def log(msg: String): Unit = {
      System.err.println(msg)
    }

    override def onStepComplete(currentPoint: Step): Unit = {
      history += currentPoint
    }
  }
  val minutesPerStep = 5


  private def mnistClassificationReport(log: ScalaNotebookOutput, categorizationNetwork : PipelineNetwork) = {
    log.eval {
      log.p("The (mis)categorization matrix displays a count matrix for every actual/predicted category: ")
      val categorizationMatrix: Map[Int, Map[Int, Int]] = log.eval {
        MNIST.validationDataStream().iterator().asScala.toStream.map(testObj ⇒ {
          val result = categorizationNetwork.eval(new NNExecutionContext {}, testObj.data).getData.get(0)
          val prediction: Int = (0 to 9).maxBy(i ⇒ result.get(i))
          val actual: Int = toOut(testObj.label)
          actual → prediction
        }).groupBy(_._1).mapValues(_.groupBy(_._2).mapValues(_.size))
      }
      log.out("Actual \\ Predicted | " + (0 to 9).mkString(" | "))
      log.out((0 to 10).map(_ ⇒ "---").mkString(" | "))
      (0 to 9).foreach(actual ⇒ {
        log.out(s" **$actual** | " + (0 to 9).map(prediction ⇒ {
          categorizationMatrix.getOrElse(actual, Map.empty).getOrElse(prediction, 0)
        }).mkString(" | "))
      })
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
  }

  private def reportMatrix(log: ScalaNotebookOutput, encoder: NNLayer, decoder: NNLayer, band: Int = 0) = {
    val inputPrototype = data.get(0)
    val dims = inputPrototype.getDimensions()
    val encoded = encoder.eval(new NNExecutionContext {}, inputPrototype).getData.get(0)
    val width = encoded.getDimensions()(0)
    val height = encoded.getDimensions()(1)
    log.draw(gfx ⇒ {
      (0 until width).foreach(x ⇒ {
        (0 until height).foreach(y ⇒ {
          encoded.fill(cvt((i: Int) ⇒ 0.0))
          encoded.set(Array(x, y, band), 1.0)
          val tensor = decoder.eval(new NNExecutionContext {}, encoded).getData.get(0)
          val sum = tensor.getData.sum
          val min = tensor.getData.min
          val max = tensor.getData.max
          var getPixel: (Int, Int) ⇒ Color = null
          val dims = tensor.getDimensions
          if (3 == dims.length) {
            if (3 == dims(2)) {
              getPixel = (xx: Int, yy: Int) ⇒ {
                val red: Double = 255 * (tensor.get(xx, yy, 0) - min) / (max - min)
                val blue: Double = 255 * (tensor.get(xx, yy, 1) - min) / (max - min)
                val green: Double = 255 * (tensor.get(xx, yy, 2) - min) / (max - min)
                new Color(red.toInt, blue.toInt, green.toInt)
              }
            } else {
              assert(1 == dims(2))
              getPixel = (xx: Int, yy: Int) ⇒ {
                val value: Double = 255 * (tensor.get(xx, yy) - min) / (max - min)
                new Color(value.toInt, value.toInt, value.toInt)
              }
            }
          } else {
            assert(2 == dims.length)
            getPixel = (xx: Int, yy: Int) ⇒ {
              val value: Double = 255 * (tensor.get(xx, yy) - min) / (max - min)
              new Color(value.toInt, value.toInt, value.toInt)
            }
          }
          (0 until dims(0)).foreach(xx ⇒
            (0 until dims(1)).foreach(yy ⇒ {
              gfx.setColor(getPixel(xx, yy))
              gfx.drawRect((x * dims(0)) + xx, (y * dims(1)) + yy, 1, 1)
            }))
        })
      })
    }, width = dims(0) * width, height = dims(1) * height)
  }

  private def preview(log: ScalaNotebookOutput, width: Int, height: Int) = {
    val inputPrototype = data.get(0)
    val dims = inputPrototype.getDimensions
    log.draw(gfx ⇒ {
      (0 until width).foreach(x ⇒ {
        (0 until height).foreach(y ⇒ {
          val tensor = data.get((y * width + x) % data.length)
          val min = 0 // tensor.getData.min
          val max = 255 // tensor.getData.max
          var getPixel: (Int, Int) ⇒ Color = null
          if (3 == dims.length) {
            if (3 == dims(2)) {
              getPixel = (xx: Int, yy: Int) ⇒ {
                val red: Double = 255 * (tensor.get(xx, yy, 0) - min) / (max - min)
                val green: Double = 255 * (tensor.get(xx, yy, 1) - min) / (max - min)
                val blue: Double = 255 * (tensor.get(xx, yy, 2) - min) / (max - min)
                new Color(red.toInt, green.toInt, blue.toInt)
              }
            } else {
              assert(1 == dims(2))
              getPixel = (xx: Int, yy: Int) ⇒ {
                val value: Double = 255 * (tensor.get(xx, yy) - min) / (max - min)
                new Color(value.toInt, value.toInt, value.toInt)
              }
            }
          } else {
            assert(2 == dims.length)
            getPixel = (xx: Int, yy: Int) ⇒ {
              val value: Double = 255 * (tensor.get(xx, yy) - min) / (max - min)
              new Color(value.toInt, value.toInt, value.toInt)
            }
          }
          (0 until dims(0)).foreach(xx ⇒
            (0 until dims(1)).foreach(yy ⇒ {
              gfx.setColor(getPixel(xx, yy))
              gfx.drawRect((x * dims(0)) + xx, (y * dims(1)) + yy, 1, 1)
            }))
        })
      })
    }, width = dims(0) * width, height = dims(1) * height)
  }

  private def reportTable(log: ScalaNotebookOutput, encoder: NNLayer, decoder: NNLayer) = {
    log.eval {
      TableOutput.create(data.stream().iterator().asScala.toList.take(20).map(testObj ⇒ {
        var evalModel: PipelineNetwork = new PipelineNetwork
        evalModel.add(encoder)
        evalModel.add(decoder)
        val result = evalModel.eval(new NNExecutionContext {}, testObj).getData.get(0)
        Map[String, AnyRef](
          "Input" → log.image(testObj.toImage(), "Input"),
          "Output" → log.image(result.toImage(), "Autoencoder Output")
        ).asJava
      }): _*)
    }
  }

  private def summarizeHistory(log: ScalaNotebookOutput) = {
    log.eval {
      val step = Math.max(Math.pow(10, Math.ceil(Math.log(history.size) / Math.log(10)) - 2), 1).toInt
      TableOutput.create(history.filter(0 == _.iteration % step).map(state ⇒
        Map[String, AnyRef](
          "iteration" → state.iteration.toInt.asInstanceOf[Integer],
          "time" → state.time.toDouble.asInstanceOf[lang.Double],
          "fitness" → state.point.sum.toDouble.asInstanceOf[lang.Double]
        ).asJava
      ): _*)
    }
    if (!history.isEmpty) log.eval {
      val plot: PlotCanvas = ScatterPlot.plot(history.map(item ⇒ Array[Double](
        item.iteration, Math.log(item.point.sum)
      )).toArray: _*)
      plot.setTitle("Convergence Plot")
      plot.setAxisLabels("Iteration", "log(Fitness)")
      plot.setSize(600, 400)
      plot
    }
  }

  def toOut(label: String): Int = {
    var i = 0
    while ( {
      i < 10
    }) {
      if (label == "[" + i + "]") return i

      {
        i += 1;
        i - 1
      }
    }
    throw new RuntimeException
  }

  def toOutNDArray(out: Int, max: Int): Tensor = {
    val ndArray = new Tensor(max)
    ndArray.set(out, 1)
    ndArray
  }

}