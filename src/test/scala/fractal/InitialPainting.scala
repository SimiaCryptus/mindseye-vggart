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

import com.simiacryptus.aws.Tendril
import com.simiacryptus.mindseye.applications.{ArtistryUtil, ColorTransfer, ImageArtUtil, TextureGeneration}
import com.simiacryptus.mindseye.lang.Tensor
import com.simiacryptus.mindseye.lang.cudnn.Precision
import com.simiacryptus.mindseye.models.CVPipe_VGG19
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.sparkbook.Java8Util._
import com.simiacryptus.util.io.NotebookOutput

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._


abstract class InitialPainting(
                                styleSources: Seq[CharSequence]
                              ) extends Tendril.SerializableConsumer[NotebookOutput] {

  def coeff_style_mean: Double = 1.0

  def coeff_style_cov: Double = 1.0

  def dreamCoeff: Double = 0.0

  override def accept(log: NotebookOutput): Unit = {
    TestUtil.addGlobalHandlers(log.getHttpd)
    for (styleSource <- styleSources) {
      log.p(log.png(ArtistryUtil.load(styleSource, style_resolution), "Style Image"))
    }
    val colorAligned: Tensor = log.subreport("Init", (output: NotebookOutput) => init(output))
    log.p(log.png(colorAligned.toImage, "Seed"))
    val painting = log.subreport("Paint", (output: NotebookOutput) => paint(output, colorAligned, getStyleSetup()).toImage)
    log.p(log.png(painting, layers.map(_.name).reduce(_ + "_" + _)))
  }

  def layers: List[CVPipe_VGG19.Layer] = List(
    CVPipe_VGG19.Layer.Layer_0,
    CVPipe_VGG19.Layer.Layer_1a,
    CVPipe_VGG19.Layer.Layer_1c
  )

  def init(log: NotebookOutput) = {
    val width = resolutionSchedule(0)
    val height = (aspect_ratio * width).toInt
    colorAlign(log, ArtistryUtil.paint_Plasma(3, 1000.0, 1.1, width, height).scale(plasma_magnitude))
  }

  def plasma_magnitude: Double = 1.0

  def colorAlign(log: NotebookOutput, inputCanvas: Tensor): Tensor = {
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

  def precision: Precision = Precision.Float

  def getStyleSetup(layers: List[CVPipe_VGG19.Layer] = layers,
                    sources: Seq[CharSequence] = styleSources,
                    coeff_mean: Double = coeff_style_mean,
                    coeff_cov: Double = coeff_style_cov,
                    coeff_dream: Double = dreamCoeff
                   ) = {
    new TextureGeneration.StyleSetup[CVPipe_VGG19.Layer](
      precision,
      mapAsJavaMap(sources.toList.map(file => file -> ArtistryUtil.load(file, style_resolution)).toMap),
      mapAsJavaMap({
        val styleCoefficients: TextureGeneration.StyleCoefficients[CVPipe_VGG19.Layer] = new TextureGeneration.StyleCoefficients[CVPipe_VGG19.Layer](TextureGeneration.CenteringMode.Origin)
        for (layer <- layers) {
          styleCoefficients.set(layer, coeff_mean, coeff_cov, coeff_dream)
        }
        Map(styleSources.toList.asJava -> styleCoefficients)
      }))
  }

  def style_resolution: Int = 1280

  def paint(log: NotebookOutput,
            inputCanvas: Tensor,
            styleSetup: TextureGeneration.StyleSetup[CVPipe_VGG19.Layer]
           ) = {
    val canvasCopy: AtomicReference[Tensor] = new AtomicReference[Tensor](inputCanvas.copy)
    for (width <- resolutionSchedule) {
      val textureGeneration: TextureGeneration.VGG19 = new TextureGeneration.VGG19
      textureGeneration.parallelLossFunctions = true
      val height = (aspect_ratio * width).toInt
      textureGeneration.setTiling(Math.max(Math.min((2.0 * Math.pow(600, 2)) / (width * height), 9), 2).toInt)
      canvasCopy.set(Tensor.fromRGB(TestUtil.resize(canvasCopy.get.toImage, width, height)))
      log.p("Input Parameters:")
      log.eval(() => {
        ArtistryUtil.toJson(styleSetup)
      })
      val fingerprint = textureGeneration.measureStyle(styleSetup)
      val newImage = textureGeneration.generate(log, canvasCopy.get, trainingMinutes, fingerprint, maxIterations, isVerbose, styleSetup.precision)
      canvasCopy.set(newImage)
    }
    canvasCopy.get
  }

  def resolutionSchedule: Array[Int] = Array(200, 600)

  def aspect_ratio: Double = 1.0

  def trainingMinutes: Int = 10

  def maxIterations: Int = 20

  def isVerbose: Boolean = false
}
