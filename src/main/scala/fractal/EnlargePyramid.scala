/*
 * Copyright (c) 2019 by Andrew Charneski.
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

package fractal

import java.awt.image.BufferedImage
import java.io.File
import java.util.function.Function

import com.simiacryptus.lang.{SerializableFunction, SerializableSupplier}
import com.simiacryptus.mindseye.applications.{ArtistryUtil, ImageArtUtil}
import com.simiacryptus.mindseye.lang.cudnn.Precision
import com.simiacryptus.mindseye.pyramid.ImagePyramid.ImageTile
import com.simiacryptus.mindseye.pyramid.{ImagePyramid, PyramidUtil}
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.notebook.{MarkdownNotebookOutput, NotebookOutput}
import com.simiacryptus.sparkbook.WorkerRunner
import com.simiacryptus.sparkbook.repl.SparkSessionProvider
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.ScalaJson
import sun.awt.AWTAutoShutdown

import scala.collection.JavaConverters._
import scala.concurrent.duration._
import scala.reflect.ClassTag

abstract class EnlargePyramid
(
  styleSources: Array[CharSequence]
) extends SerializableFunction[NotebookOutput, Object] with StyleTransferParams with SparkSessionProvider {

  val tileSize: Int = 512
  val magLevels: Int = 1
  val padding: Int = 20
  val startLevel: Int
  val aspect: Double = 1.0
  val trainingMinutes: Int = 10
  val maxIterations: Int = 20
  val verbose: Boolean = false
  val style_resolution: Int = -1

  def inputHref: String

  def imagePrefix: String

  def inputHadoop: String

  override def apply(log: NotebookOutput): Object = {
    TestUtil.addGlobalHandlers(log.getHttpd)
    log.asInstanceOf[MarkdownNotebookOutput].setMaxImageSize(8 * 1024)
    log.eval(() => {
      ScalaJson.toJson(EnlargePyramid.this, ScalaJson.getExplicitMapper)
    })
    try {
      PyramidUtil.initJS(log)
      PyramidUtil.writeViewer(log, "source", new ImagePyramid(tileSize, startLevel, aspect, inputHref))
      enlarge(log)
      null
    } catch {
      case throwable: Throwable =>
        log.eval(() => {
          throwable
        })
    }
  }

  def enlarge(log: NotebookOutput) = {
    val destLevel = startLevel + magLevels
    val sourcePyramid = new ImagePyramid(tileSize, startLevel, aspect, inputHadoop)
    log.p(log.jpg(sourcePyramid.assemble((tileSize * Math.pow(2, startLevel)).toInt), "Full Source Image"))
    // Await cluster init
    await(5 minutes) {
      sc.getExecutorMemoryStatus.size < 4
    }
    Thread.sleep(20000) // Await full init

    var tileRdd = sc.parallelize(sourcePyramid.getImageTileFns(padding).asScala)
    WorkerRunner.mapPartitions(tileRdd, (workerLog, tiles: Iterator[_ <: SerializableSupplier[ImageTile]]) => {
      val imageFunction = getImageEnlargingFunction(
        workerLog,
        ((tileSize + 2 * padding) * Math.pow(2, magLevels)).toInt,
        ((tileSize + 2 * padding) * Math.pow(2, magLevels)).toInt,
        trainingMinutes,
        maxIterations,
        verbose,
        magLevels,
        padding,
        styleSources: _*)
      AWTAutoShutdown.getInstance().run()
      tiles.map(tileF => {
        val tile = tileF.get()
        val sourceImage = tile.getImage
        workerLog.p(workerLog.jpg(sourceImage, "Source"))
        val resultImage = imageFunction.apply(sourceImage)
        workerLog.p(workerLog.jpg(resultImage, "Result"))
        new ImageTile(tile.getRow(), tile.getCol, resultImage)
      }).toList.iterator
    })(ClassTag(classOf[ImageTile]), ClassTag(classOf[ImageTile]), log, spark).foreach(sourcePyramid.collect(magLevels, padding, inputHadoop, _))
    new ImagePyramid(tileSize, destLevel, aspect, inputHadoop).copyReducePyramid(log.getResourceDir.getAbsolutePath + File.separator + "tile_")
    new ImagePyramid(tileSize, destLevel, aspect, "tile_").writeViewer(log)
  }

  def getImageEnlargingFunction(log: NotebookOutput, width: Int, height: Int, trainingMinutes: Int, maxIterations: Int, verbose: Boolean, magLevels: Int, padding: Int, styleSources: CharSequence*): Function[BufferedImage, BufferedImage] = {
    val tileLayout = new ImageArtUtil.TileLayout(600, padding, 0, 0, Array[Int](width, height))
    styleSources.foreach(source => try {
      log.p(log.jpg(ArtistryUtil.load(source, style_resolution), source))
    } catch {
      case e: Throwable => e.printStackTrace()
    })
    val styleTransfer = PyramidUtil.getStyleTransfer()
    val styleSetup = getStyleSetup_SegmentedStyleTransfer(Precision.Float, styleSources, style_resolution)
    val opParams = new ImageArtUtil.ImageArtOpParams(log, trainingMinutes, maxIterations, verbose)
    val transformer = new ImageArtUtil.StyleTransformer(opParams, styleTransfer, tileLayout, padding, 0, 0, styleSetup)
    PyramidUtil.getImageEnlargingFunction(log, trainingMinutes, maxIterations, verbose, magLevels, padding, tileLayout, styleTransfer, styleSetup, transformer)
  }


}
