/*
 * Copyright (c) 2018 by Andrew Charneski.
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

import com.simiacryptus.mindseye.applications.{ArtistryUtil, ImageArtUtil}
import com.simiacryptus.mindseye.lang.cudnn.Precision
import com.simiacryptus.mindseye.pyramid.{ImagePyramid, PyramidUtil}
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.sparkbook.Java8Util._
import com.simiacryptus.util.io.{MarkdownNotebookOutput, NotebookOutput, ScalaJson}
import com.simiacryptus.util.lang.SerializableConsumer

abstract class InitialPyramid
(
  initialContent: String,
  styleSources: Array[CharSequence]
) extends SerializableConsumer[NotebookOutput] with StyleTransferParams {

  val styleSize = -1
  var aspect: Double = 1.0

  val tileSize: Int = 512
  val padding: Int = 20
  val trainingMinutes: Int = 10
  val maxIterations: Int = 20
  val verbose: Boolean = false
  val style_resolution = -1

  override def accept(log: NotebookOutput): Unit = {
    TestUtil.addGlobalHandlers(log.getHttpd)
    log.asInstanceOf[MarkdownNotebookOutput].setMaxImageSize(8 * 1024)
    log.eval(() => {
      ScalaJson.toJson(InitialPyramid.this)
    })
    try {
      PyramidUtil.initJS(log)
      val image = ArtistryUtil.load(initialContent, styleSize)
      log.p(log.jpg(image, "Initial Image"))
      aspect = image.getHeight.toDouble / image.getWidth
      PyramidUtil.initImagePyramids(log, "Image", tileSize, image)

      val startLevel = Math.ceil(Math.log(image.getWidth.toDouble / tileSize) / Math.log(2)).toInt
      val localPrefix = "tile_0_"
      val magLevels: Int = 1
      val destLevel = startLevel + magLevels
      val hadoopPrefix = ("file:///" + log.getResourceDir.getAbsolutePath) + "/" + localPrefix
      val finalTotalWidth = (tileSize * Math.pow(2, destLevel)).toInt

      val targetSize = ((tileSize + 2 * padding) * Math.pow(2, magLevels)).toInt
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
      new ImagePyramid(tileSize, startLevel, aspect, hadoopPrefix).buildNewImagePyramidLayer(magLevels, padding, hadoopPrefix, imageFunction, false)
      new ImagePyramid(tileSize, destLevel, aspect, localPrefix).writeViewer(log)
      log.p(log.jpg(new ImagePyramid(tileSize, destLevel, aspect, hadoopPrefix).assemble(finalTotalWidth), "Full Image"))
    } catch {
      case throwable: Throwable =>
        log.eval(() => {
          throwable
        })
    }
  }

  def getImageEnlargingFunction(log: NotebookOutput, width: Int, height: Int, trainingMinutes: Int, maxIterations: Int, verbose: Boolean, magLevels: Int, padding: Int, styleSources: CharSequence*): Function[BufferedImage, BufferedImage] = {
    val tileLayout = new ImageArtUtil.TileLayout(600, padding, 0, 0, Array[Int](width, height))
    styleSources.foreach(source => log.p(log.jpg(ArtistryUtil.load(source, style_resolution), source)))
    val styleTransfer = PyramidUtil.getStyleTransfer()
    val styleSetup = getStyleSetup_SegmentedStyleTransfer(Precision.Float, styleSources, style_resolution)
    val opParams = new ImageArtUtil.ImageArtOpParams(log, trainingMinutes, maxIterations, verbose)
    val transformer = new ImageArtUtil.StyleTransformer(opParams, styleTransfer, tileLayout, padding, 0, 0, styleSetup)
    PyramidUtil.getImageEnlargingFunction(log, trainingMinutes, maxIterations, verbose, magLevels, padding, tileLayout, styleTransfer, styleSetup, transformer)
  }


}
