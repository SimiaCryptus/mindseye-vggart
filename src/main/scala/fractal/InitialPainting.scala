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

import java.util.concurrent.atomic.AtomicReference

import com.simiacryptus.mindseye.applications._
import com.simiacryptus.mindseye.lang.Tensor
import com.simiacryptus.mindseye.lang.cudnn.Precision
import com.simiacryptus.mindseye.models.CVPipe_VGG19
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.sparkbook.Java8Util._
import com.simiacryptus.util.io.{MarkdownNotebookOutput, NotebookOutput, ScalaJson}
import com.simiacryptus.util.lang.SerializableConsumer

import scala.collection.JavaConversions._


abstract class InitialPainting
(
  styleSources: Seq[CharSequence]
) extends SerializableConsumer[NotebookOutput] with StyleTransferParams {
  require(!styleSources.isEmpty)

  def plasma_magnitude: Double = 1.0

  override def accept(log: NotebookOutput): Unit = {
    TestUtil.addGlobalHandlers(log.getHttpd)
    log.asInstanceOf[MarkdownNotebookOutput].setMaxImageSize(10000)
    log.eval(() => {
      ScalaJson.toJson(InitialPainting.this)
    })
    for (styleSource <- styleSources) {
      log.p(log.png(ArtistryUtil.load(styleSource, style_resolution), styleSource))
    }
    val colorAligned: Tensor = log.subreport("Init", (output: NotebookOutput) => init(output))
    log.p(log.png(colorAligned.toImage, "Seed"))
    val painting = log.subreport("Paint", (output: NotebookOutput) =>
      paint(output, colorAligned, r => getStyleSetup_TextureGeneration(precision, styleSources, style_resolution)).toImage)
    log.p(log.png(painting, "painting"))
  }

  def precision: Precision = Precision.Float

  def style_resolution: Int = 1280

  def paint
  (
    log: NotebookOutput,
    inputCanvas: Tensor,
    styleSetup: Int => TextureGeneration.StyleSetup[CVPipe_VGG19.Layer]
  ): Tensor = {
    val canvasCopy: AtomicReference[Tensor] = new AtomicReference[Tensor](inputCanvas.copy)
    for (width <- resolutionSchedule) {
      val textureGeneration: TextureGeneration.VGG19 = new TextureGeneration.VGG19
      textureGeneration.parallelLossFunctions = true
      val height = (aspect_ratio * width).toInt
      val tiling = Math.max(Math.min((2.0 * Math.pow(600, 2)) / (width * height), 9), 2).toInt
      textureGeneration.setTiling(tiling)
      canvasCopy.set(Tensor.fromRGB(TestUtil.resize(canvasCopy.get.toImage, width, height)))
      log.p("Input Parameters:")
      val style = styleSetup(width)
      log.eval(() => {
        ArtistryUtil.toJson(style)
      })
      val fingerprint = textureGeneration.measureStyle(style)
      val network = textureGeneration.fitnessNetwork(fingerprint)
      val file = log.asInstanceOf[MarkdownNotebookOutput].resolveResource("style_network.zip")
      network.writeZip(file)
      log.p(log.link(file, "Artist Network"))
      canvasCopy.set(TextureGeneration.optimize(log, network, canvasCopy.get, trainingMinutes, maxIterations, isVerbose, style.precision, tiling))
    }
    canvasCopy.get
  }

  def resolutionSchedule: Array[Int] = Array(100, 160, 220, 300, 400, 512)

  def aspect_ratio: Double = 1.0

  def trainingMinutes: Int = 10

  def maxIterations: Int = 20

  def init(log: NotebookOutput) = {
    val width = resolutionSchedule(0)
    val height = (aspect_ratio * width).toInt
    colorAlign(log, ArtistryUtil.paint_Plasma(3, 1000.0, 1.1, width, height).scale(plasma_magnitude))
  }

  def isVerbose: Boolean = false

  def colorAlign
  (
    log: NotebookOutput,
    inputCanvas: Tensor
  ): Tensor = {
    val contentColorTransform: ColorTransfer[CVPipe_VGG19.Layer, CVPipe_VGG19] = new ColorTransfer.VGG19() {}.setOrtho(false).setUnit(true)
    val width = 600
    val height = (aspect_ratio * width).toInt
    val resizedCanvas: Tensor = Tensor.fromRGB(TestUtil.resize(inputCanvas.toImage, width, height))
    val empty = new java.util.HashMap[CharSequence, ColorTransfer[CVPipe_VGG19.Layer, CVPipe_VGG19]]
    val styleImages = ImageArtUtil.getStyleImages(styleSources.toArray, empty, width, height)
    val styleSetup = ImageArtUtil.getColorAnalogSetup(styleSources.toList, precision, resizedCanvas, styleImages, CVPipe_VGG19.Layer.Layer_0)
    val styleFingerprint = contentColorTransform.measureStyle(styleSetup)
    contentColorTransform.transfer(log, resizedCanvas, styleSetup, trainingMinutes, styleFingerprint, maxIterations, isVerbose)
    contentColorTransform.forwardTransform(inputCanvas)
  }

}
