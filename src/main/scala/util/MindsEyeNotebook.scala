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

package util

import java.io._
import java.util.concurrent.{Semaphore, TimeUnit}
import java.{lang, util}

import com.aparapi.internal.kernel.KernelManager
import com.fasterxml.jackson.databind.ObjectMapper
import com.google.gson.{GsonBuilder, JsonObject}
import com.simiacryptus.mindseye.layers.NNLayer
import com.simiacryptus.mindseye.network.graph.DAGNetwork
import com.simiacryptus.mindseye.opt.{IterativeTrainer, Step, TrainingMonitor}
import com.simiacryptus.util.ArrayUtil._
import com.simiacryptus.util.io.{HtmlNotebookOutput, IOUtil, KryoUtil, TeeOutputStream}
import com.simiacryptus.util.text.TableOutput
import com.simiacryptus.util.{MonitoredObject, StreamNanoHTTPD, TimerText}
import fi.iki.elonen.NanoHTTPD
import fi.iki.elonen.NanoHTTPD.IHTTPSession
import org.apache.commons.io.IOUtils
import smile.plot.{PlotCanvas, ScatterPlot}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration
import scala.concurrent.{Await, Future}

abstract class MindsEyeNotebook(server: StreamNanoHTTPD, out: HtmlNotebookOutput with ScalaNotebookOutput) {

  val history = new scala.collection.mutable.ArrayBuffer[Step]()
  val logOut = new TeeOutputStream(out.file("../log.txt"), true)
  val logPrintStream = new PrintStream(logOut)
  val monitoringRoot = new MonitoredObject()
  val dataTable = new TableOutput()
  val checkpointFrequency = 10
  var model: NNLayer = null
  var modelCheckpoint : NNLayer = null
  def getModelCheckpoint = Option(modelCheckpoint).getOrElse(model)
  val pauseSemaphore = new Semaphore(1)

  def onStepComplete(currentPoint: Step): Unit = {}
  val monitor = new TrainingMonitor {
    val timer = new TimerText
    override def log(msg: String): Unit = {
      println(timer + " " + msg)
      logPrintStream.println(timer + " " + msg)
    }

    override def onStepComplete(currentPoint: Step): Unit = {
      try {
        history += currentPoint
        if(0 == currentPoint.iteration % checkpointFrequency) {
          modelCheckpoint = KryoUtil.kryo().copy(model)
          IOUtil.writeString(model.getJsonString, out.file("../model.json"))
        }
        val iteration = currentPoint.iteration
        if(shouldReplotMetrics(iteration)) regenerateReports()
        def flatten(prefix:String,data:Map[String,AnyRef]) : Map[String,AnyRef] = {
          data.flatMap({
            case (key, value) ⇒ value match {
              case value : Number ⇒ Map((prefix + key) → value)
              case value : util.List[String] ⇒ Map.empty[String,AnyRef]
              case value : util.Map[String,AnyRef] ⇒ flatten(prefix+key+".", value.asScala.toMap)
              case value : Map[String,AnyRef] ⇒ flatten(prefix+key, value)
            }
          }).map(e⇒(if(e._1.startsWith(".")) e._1.substring(1) else e._1)→e._2)
        }
        dataTable.putRow((flatten(".",monitoringRoot.getMetrics.asScala.toMap)++Map(
          "epoch" → currentPoint.iteration.asInstanceOf[lang.Long],
          "time" → currentPoint.time.asInstanceOf[lang.Long],
          "value" → currentPoint.point.value.asInstanceOf[lang.Double]
        )).asJava)
        MindsEyeNotebook.this.onStepComplete(currentPoint)
        pauseSemaphore.acquire()
        pauseSemaphore.release()
      } catch {
        case e : Throwable ⇒ e.printStackTrace()
      }
    }
  }
  monitoringRoot.addField("openCL",Java8Util.cvt(()⇒{
    val sb = new java.lang.StringBuilder()
    KernelManager.instance().reportDeviceUsage(sb,true)
    util.Arrays.asList(sb.toString().split("\n"))
  }))


  protected def shouldReplotMetrics(iteration: Long) = iteration match {
    case _ if List(10,50).contains(iteration) ⇒ true
    case _ if 100 > iteration ⇒ false
    case _ if 0 == iteration % 100 ⇒ true
    case _ ⇒ false
  }


  def defineHeader(log: HtmlNotebookOutput with ScalaNotebookOutput = out): Unit = {

    log.p("Reports: <a href='model.json'>Model Json</a>, <a href='metricsHistory.html'>Metrics Plots</a>, <a href='mobility.html'>Mobility</a>, <a href='log.txt'>Optimization Log</a>, or <a href='metrics.csv'>Metrics Data</a>")
    server.addSyncHandler("model.json", "application/json", Java8Util.cvt(out ⇒ {
      out.write(new GsonBuilder().setPrettyPrinting().create().toJson(getModelCheckpoint.getJson).getBytes)
    }), false)
    server.addSessionHandler("log.txt", Java8Util.cvt((session : IHTTPSession)⇒{
      NanoHTTPD.newChunkedResponse(NanoHTTPD.Response.Status.OK, "text/plain", logOut.newInputStream())
    }))
    server.addSyncHandler("table.csv", "text/csv", Java8Util.cvt(out ⇒ {
      IOUtils.write(dataTable.toCSV(false), out, "UTF-8")
    }), false)
    server.addSyncHandler("netmon.json", "application/json", Java8Util.cvt(out ⇒ {
      val mapper = new ObjectMapper().enableDefaultTyping(ObjectMapper.DefaultTyping.NON_FINAL)
      val buffer = new ByteArrayOutputStream()
      mapper.writeValue(buffer, monitoringRoot.getMetrics)
      out.write(buffer.toByteArray)
    }), false)


    log.p("Process Control: <a href='/pause' target='Pause'>Pause</a> or <a href='/resume' target='Pause'>Resume</a> Training</p><iframe name='Pause' style='height:50pt;width:200;float:right;border:none;'></iframe>")
    server.addSyncHandler("pause", "text/html", Java8Util.cvt(out ⇒ {
      Option(new HtmlNotebookOutput(log.workingDir, out) with ScalaNotebookOutput).foreach(log ⇒ {
        pauseSemaphore.acquire(1)
        log.h1("Paused")
      })
    }), false)
    server.addSyncHandler("resume", "text/html", Java8Util.cvt(out ⇒ {
      Option(new HtmlNotebookOutput(log.workingDir, out) with ScalaNotebookOutput).foreach(log ⇒ {
        pauseSemaphore.drainPermits()
        pauseSemaphore.release(1)
        log.h1("Resumed")
      })
    }), false)
    server.addSyncHandler("history.html", "text/html", Java8Util.cvt(out ⇒ {
      Option(new HtmlNotebookOutput(log.workingDir, out) with ScalaNotebookOutput).foreach(log ⇒ {
        summarizeHistory(log)
      })
    }), false)
    defineReports(log)
    log.out("<hr/>")
  }


  def defineReports(log: HtmlNotebookOutput with ScalaNotebookOutput) = {
    log.p("Interactive Reports: <a href='/history.html'>Convergence History</a>")
  }

  def summarizeHistory(log: ScalaNotebookOutput = out) = {
    if (!history.isEmpty) {
      log.eval {
        val plot: PlotCanvas = ScatterPlot.plot(history.map(item ⇒ Array[Double](
          item.iteration, Math.log(item.point.value)
        )).toArray: _*)
        plot.setTitle("Convergence Plot")
        plot.setAxisLabels("Iteration", "log(Fitness)")
        plot.setSize(600, 400)
        plot
      }
    }
  }

  def regenerateReports() = Future.sequence(List(
    {
      val file = out.file("../metricsHistory.html")
      val report = new HtmlNotebookOutput(out.workingDir, file) with ScalaNotebookOutput
      generateMetricsHistoryReport(report).andThen({case _⇒report.close()})
    },{
      val file = out.file("../mobility.html")
      val report = new HtmlNotebookOutput(out.workingDir, file) with ScalaNotebookOutput
      generateMobilityReport(report).andThen({case _⇒report.close()})
    }
  ))

  def generateMobilityReport(log: ScalaNotebookOutput = out): Future[Unit] = Future {
    if (!history.isEmpty) {
      val layers: Array[NNLayer] = history.flatMap(_.point.weights.map.asScala.keySet).distinct.toArray
      val outputTable = new mutable.HashMap[Int, mutable.Map[String, AnyRef]]()
      log.out("<table>")
      layers.foreach(layer ⇒ {
        try {
          val transcript: List[Array[Double]] = history.map(_.point.weights.map.get(layer).delta).toList
          log.out("<tr><td>")
          log.p(s"${layer.getName}")
          List(1, 5, 20).foreach(lag ⇒ {
            log.out("</td><td>")
            val xy = (lag until transcript.size).map(i ⇒ {
              val v = Math.log10(magnitude(subtract(transcript(i), transcript(i - lag))) / lag)
              outputTable.getOrElseUpdate(i, new mutable.HashMap[String,AnyRef]())(s"${layer.getName}/$lag") = v.asInstanceOf[Object]
              i → v
            }).filter(d ⇒ java.lang.Double.isFinite(d._2))
            if (xy.size > 1) {
              val plot: PlotCanvas = ScatterPlot.plot(xy.map(xy ⇒ Array(xy._1.toDouble, xy._2)): _*)
              plot.setTitle(s"${layer.getName}")
              plot.setAxisLabels("Epoch", s"log(dist(n,n-$lag)/$lag)")
              plot.setSize(600, 400)
              log.eval {
                plot
              }
            } else log.out("No Data")
          })
          log.out("</td></tr>")
        } catch {
          case e: Throwable ⇒
        }
      })
      IOUtil.writeString(TableOutput.create(outputTable.toList.sortBy(_._1).map(_._2.toMap.asJava).toArray: _*).toCSV(true), log.file("../mobility.csv"))
      log.out("</table>")
    }
  }

  def generateMetricsHistoryReport(log: ScalaNotebookOutput = out): Future[Unit] = {
    if(!history.isEmpty) {
      val dataAsScala: Array[Map[String, AnyRef]] = dataTable.rows.asScala.map(_.asScala.toMap).toArray
      val keys: Array[String] = dataTable.schema.asScala.keySet.toArray
      val outputTable = new mutable.HashMap[Int, mutable.Map[String, AnyRef]]()
      Future {
        log.out("<table><tr><th>Vs Iteration</th><th>Vs Objective</th></tr>")
        keys
          .filterNot(_.contains("Performance"))
          .filterNot(_.contains("PerItem"))
          .filterNot(_.contains("count"))
          .sorted
          .foreach(key⇒{
            log.out("<tr><td>")
            try {
              val data = dataAsScala.map(row ⇒ {
                val v = row(key).toString.toDouble
                val i = row("epoch").toString.toInt
                outputTable.getOrElseUpdate(i, new mutable.HashMap[String,AnyRef]())(key) = v.asInstanceOf[Object]
                Array[Double](
                  i.toDouble, v
                )
              }).filter(d ⇒ d.forall(java.lang.Double.isFinite))
              log.p(s"$key vs Epoch")
              if(data.size > 1) {
                val plot: PlotCanvas = ScatterPlot.plot(data: _*)
                plot.setAxisLabels("Epoch", key)
                plot.setSize(600, 400)
                log.eval {
                  plot
                }
              } else log.out("No Data")
            } catch {
              case e : Throwable ⇒
            }
            log.out("</td><td>")
            try {
              val data = dataAsScala.map(row ⇒ Array[Double](
                Math.log(row("value").toString.toDouble), row(key).toString.toDouble
              )).filter(d ⇒ d.forall(java.lang.Double.isFinite))
              log.p(s"$key vs Log(Fitness)")
              if(data.size > 1) {
                val plot: PlotCanvas = ScatterPlot.plot(data: _*)
                plot.setAxisLabels("log(value)", key)
                plot.setSize(600, 400)
                log.eval {
                  plot
                }
              } else log.out("No Data")
            } catch {
              case e : Throwable ⇒
            }
            log.out("</td></tr>")
          })
        log.out("</table>")
        IOUtil.writeString(TableOutput.create(outputTable.toList.sortBy(_._1).map(_._2.toMap.asJava): _*).toCSV(true), log.file("../metrics.csv"))
      }
    } else Future.successful()
  }

  def waitForExit(log: HtmlNotebookOutput with ScalaNotebookOutput = out): Unit = {
    logOut.close()
    val onExit = new Semaphore(0)
    log.out("<hr/>")
    log.p("To exit the sever: <a href='/exit' target='Pause'>/exit</a>")
    server.addAsyncHandler("exit", "text/html", Java8Util.cvt(out ⇒ {
      Option(new HtmlNotebookOutput(log.workingDir, out) with ScalaNotebookOutput).foreach(log ⇒ {
        log.h1("EXIT")
        onExit.release(1)
      })
    }), false)
    onExit.acquire()
  }

  def phase[T](inputFile: String, fn: NNLayer ⇒ T): Unit = {
    phase(read(inputFile),
      layer ⇒ {
        fn(layer)
        layer
      }, model ⇒ {})
  }

  def phase[T](inputFile: String, fn: NNLayer ⇒ T, outputFile: String): Unit = {
    phase(read(inputFile),
      layer ⇒ {
        fn(layer)
        layer
      }, model ⇒ write(outputFile, model))
  }

  def write(name: String, model: NNLayer) = {
    val file = nextFile(name)
    out.p(s"Saving $file")
    IOUtil.writeString(model.getJsonString, new FileOutputStream(file))
  }

  def nextFile(name: String): String = Stream.from(1).map(name + "." + _ + ".json").find(!new File(_).exists).get
  def findFile(name: String): Option[String] = Stream.from(1).map(name + "." + _ + ".json").takeWhile(new File(_).exists).lastOption

  def read(name: String): NNLayer = {
    findFile(name).map(inputFile⇒{
      out.p(s"Loading $inputFile")
      NNLayer.fromJson(new GsonBuilder().create().fromJson(IOUtils.toString(new FileInputStream(inputFile), "UTF-8"), classOf[JsonObject]))
    }).getOrElse(throw new RuntimeException(s"Could not find any files named $name.*.json"))
  }

  def phase[T](input: ⇒ NNLayer, fn: NNLayer ⇒ T, outputFile: String): T = {
    phase(input,
      layer ⇒ {
        val result = fn(layer)
        layer
        result
      }, model ⇒ write(outputFile, model))
  }

  private def phase[T](initializer: ⇒ NNLayer, fn: NNLayer ⇒ T, onComplete: NNLayer ⇒ Unit): T = {
    out.p("Loading Model")
    model = initializer
    out.p("Model Loaded")
    try {
      model.asInstanceOf[DAGNetwork].attach(monitoringRoot)
      val result = fn(model)
      onComplete(model)
      out.p("Training Phase Complete")
      result
    } catch {
      case e : Throwable ⇒ throw e
    } finally {
      summarizeHistory(out)
      Await.result(regenerateReports, Duration(10,TimeUnit.MINUTES))
    }
  }

}
