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
import java.util.function.Function

import com.simiacryptus.aws.S3Util
import com.simiacryptus.lang.SerializableFunction
import com.simiacryptus.mindseye.applications.{ArtistryUtil, ImageArtUtil}
import com.simiacryptus.mindseye.lang.cudnn.Precision
import com.simiacryptus.mindseye.pyramid.ImagePyramid.ImageTile
import com.simiacryptus.mindseye.pyramid.{ImagePyramid, PyramidUtil}
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.notebook.{MarkdownNotebookOutput, NotebookOutput}
import com.simiacryptus.sparkbook.WorkerRunner
import com.simiacryptus.sparkbook.repl.SparkSessionProvider
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.{LocalAppSettings, ScalaJson}

import scala.collection.JavaConverters._

abstract class InitialPyramid
(
  initialContent: String,
  styleSources: Array[CharSequence]
) extends SerializableFunction[NotebookOutput, Object] with StyleTransferParams with SparkSessionProvider {

  val initialSize = -1
  val tileSize: Int = 512
  val padding: Int = 20
  val trainingMinutes: Int = 10
  val maxIterations: Int = 10
  val verbose: Boolean = false
  val style_resolution = 800
  var aspect: Double = 1.0

  override def apply(log: NotebookOutput): Object = {
    implicit val _log = log
    implicit val _spark = spark
    TestUtil.addGlobalHandlers(log.getHttpd)
    log.asInstanceOf[MarkdownNotebookOutput].setMaxImageSize(8 * 1024)
    log.eval(() => {
      ScalaJson.toJson(InitialPyramid.this)
    })
    try {
      PyramidUtil.initJS(log)
      val image = ArtistryUtil.load(initialContent, initialSize)
      log.p(log.jpg(image, "Initial Image"))
      aspect = image.getHeight.toDouble / image.getWidth
      PyramidUtil.initImagePyramids(log, "Image", tileSize, image)
      S3Util.upload(log)

      val startLevel = Math.ceil(Math.log(image.getWidth.toDouble / tileSize) / Math.log(2)).toInt
      val localPrefix = "tile_0_"
      val magLevels: Int = 1
      val destLevel = startLevel + magLevels
      val finalTotalWidth = (tileSize * Math.pow(2, destLevel)).toInt
      val targetSize = ((tileSize + 2 * padding) * Math.pow(2, magLevels)).toInt

      val hadoopPrefix = if (null != log.getArchiveHome) {
        log.getArchiveHome.resolve("etc/" + localPrefix).toString.replaceAll("^s3:", "s3a:")
      } else {
        "file:///" + log.getResourceDir.getAbsolutePath + "/" + localPrefix
      }
      val pyramid = new ImagePyramid(tileSize, startLevel, aspect, hadoopPrefix)
      WorkerRunner.distribute((childLog: NotebookOutput, i: Long) => {
        LocalAppSettings.read().get("worker.index").foreach(idx => {
          System.setProperty("CUDA_DEVICES", idx)
        })
      })
      val tileRdd = sc.parallelize(pyramid.getImageTiles(padding, false).asScala)
      WorkerRunner.mapPartitions(tileRdd, (log, tiles: Iterator[ImageTile]) => {
        val imageFunction = getImageEnlargingFunction(
          log,
          targetSize,
          targetSize,
          trainingMinutes,
          maxIterations,
          verbose,
          magLevels,
          padding,
          styleSources: _*
        )
        tiles.map(tile => {
          new ImageTile(tile.getRow(), tile.getCol, imageFunction.apply(tile.getImage))
        })
      }).collect().foreach(pyramid.collect(magLevels, padding, hadoopPrefix, _))
      new ImagePyramid(tileSize, destLevel, aspect, localPrefix).writeViewer(log)
      log.p(log.jpg(new ImagePyramid(tileSize, destLevel, aspect, hadoopPrefix).assemble(finalTotalWidth), "Full Image"))
    } catch {
      case throwable: Throwable =>
        log.eval(() => {
          throwable
        })
    }
    null
  }

  def getImageEnlargingFunction(log: NotebookOutput, width: Int, height: Int, trainingMinutes: Int, maxIterations: Int, verbose: Boolean, magLevels: Int, padding: Int, styleSources: CharSequence*): Function[BufferedImage, BufferedImage] = {
    val tileLayout = new ImageArtUtil.TileLayout(600, padding, 0, 0, Array[Int](width, height))
    //styleSources.foreach(source => log.p(log.jpg(ArtistryUtil.load(source, style_resolution), source)))
    val styleTransfer = PyramidUtil.getStyleTransfer()
    val styleSetup = getStyleSetup_SegmentedStyleTransfer(Precision.Float, styleSources, style_resolution)
    val opParams = new ImageArtUtil.ImageArtOpParams(log, trainingMinutes, maxIterations, verbose)
    val transformer = new ImageArtUtil.StyleTransformer(opParams, styleTransfer, tileLayout, padding, 0, 0, styleSetup)
    PyramidUtil.getImageEnlargingFunction(log, trainingMinutes, maxIterations, verbose, magLevels, padding, tileLayout, styleTransfer, styleSetup, transformer)
  }


}
