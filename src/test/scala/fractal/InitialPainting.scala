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
import com.simiacryptus.util.io.NotebookOutput

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._
import scala.collection.immutable

object InitialPainting {
  def wrap(xx: List[CVPipe_VGG19.Layer]): List[List[CVPipe_VGG19.Layer]] = xx.map((x: CVPipe_VGG19.Layer) => List(x))

  def join(a: List[List[CVPipe_VGG19.Layer]], b: List[List[CVPipe_VGG19.Layer]]): List[List[CVPipe_VGG19.Layer]] = a.flatMap((layerA: List[CVPipe_VGG19.Layer]) => b.map((layerB: List[CVPipe_VGG19.Layer]) => {
    (layerA ++ layerB).distinct.sorted
  }))

  def reduce(combined: List[List[CVPipe_VGG19.Layer]], size: Int): List[List[CVPipe_VGG19.Layer]] = combined.map(_.distinct.sorted).distinct.filter(_.size >= size).sortWith((a, b) => {
    var compare: Int = Integer.compare(a.size, b.size)
    var i: Int = 0
    while ( {
      0 == compare && i < a.size
    }) {
      val _i: Int = {
        i += 1;
        i - 1
      }
      compare = a(_i).name.compareTo(b(_i).name)
    }
    compare < 0
  })

}

abstract class InitialPainting(
                                coeff_style_mean: Double = 1.0,
                                coeff_style_cov: Double = 1.0,
                                dreamCoeff: Double = 0.0,
                                resolutionSchedule: Array[Int] = Array(200, 600),
                                style_resolution: Int = 1200,
                                aspect_ratio: Double = 1.0,
                                plasma_magnitude: Double = 1.0,
                                trainingMinutes: Int = 10,
                                maxIterations: Int = 20,
                                isVerbose: Boolean = false,
                                styleSources: Seq[CharSequence]
                              ) extends Tendril.SerializableConsumer[NotebookOutput] {
  def getLayers: List[List[CVPipe_VGG19.Layer]]

  override def accept(log: NotebookOutput): Unit = {
    val precision = Precision.Float
    log.p("Style Source:")
    for (styleSource <- styleSources) {
      log.p(log.png(ArtistryUtil.load(styleSource, style_resolution), "Style Image"))
    }
    val canvas = new AtomicReference[Tensor](ArtistryUtil.paint_Plasma(3, 1000.0, 1.1, resolutionSchedule(0), (aspect_ratio * resolutionSchedule(0)).toInt).scale(plasma_magnitude))
    canvas.set(log.subreport("Color_Space_Analog", (sublog: NotebookOutput) => {
      val contentColorTransform: ColorTransfer[CVPipe_VGG19.Layer, CVPipe_VGG19] = new ColorTransfer.VGG19() {}.setOrtho(false).setUnit(true)
      //colorSyncContentCoeffMap.set(CVPipe_VGG19.Layer.Layer_1a, 1e-1);
      val colorSyncResolution: Int = 600
      val resizedCanvas: Tensor = Tensor.fromRGB(TestUtil.resize(canvas.get.toImage, colorSyncResolution, (aspect_ratio * colorSyncResolution).toInt))
      val styleSetup: ColorTransfer.StyleSetup[CVPipe_VGG19.Layer] = ImageArtUtil.getColorAnalogSetup(styleSources.toList.asJava, precision, resizedCanvas, ImageArtUtil.getStyleImages(styleSources.toArray, new java.util.HashMap[CharSequence, ColorTransfer[CVPipe_VGG19.Layer, CVPipe_VGG19]], colorSyncResolution, (aspect_ratio * colorSyncResolution).toInt), CVPipe_VGG19.Layer.Layer_0)
      contentColorTransform.transfer(sublog, resizedCanvas, styleSetup, trainingMinutes, contentColorTransform.measureStyle(styleSetup), maxIterations, isVerbose)
      contentColorTransform.forwardTransform(canvas.get)
    }))
    for (layers: immutable.Seq[CVPipe_VGG19.Layer] <- getLayers) {
      val reportName = layers.map((x: CVPipe_VGG19.Layer) => x.name).reduce(_ + "_" + _)
      log.h1(reportName)
      val subresult: Tensor = log.subreport(reportName, (subreport: NotebookOutput) => {
        val styles = {
          val styleCoefficients: TextureGeneration.StyleCoefficients[CVPipe_VGG19.Layer] = new TextureGeneration.StyleCoefficients[CVPipe_VGG19.Layer](TextureGeneration.CenteringMode.Origin)
          for (layer <- layers) {
            styleCoefficients.set(layer, coeff_style_mean, coeff_style_cov, dreamCoeff)
          }
          Map(styleSources.toList.asJava -> styleCoefficients)
        }
        val styleSetup: TextureGeneration.StyleSetup[CVPipe_VGG19.Layer] = new TextureGeneration.StyleSetup[CVPipe_VGG19.Layer](precision,
          mapAsJavaMap(styles.keySet.flatten.map(file => file -> ArtistryUtil.load(file, style_resolution)).toMap),
          mapAsJavaMap(styles))
        val canvasCopy: AtomicReference[Tensor] = new AtomicReference[Tensor](canvas.get.copy)
        for (width <- resolutionSchedule) {
          val textureGeneration: TextureGeneration.VGG19 = new TextureGeneration.VGG19
          textureGeneration.parallelLossFunctions = true
          val height = (aspect_ratio * width).toInt
          textureGeneration.setTiling(Math.max(Math.min((2.0 * Math.pow(600, 2)) / (width * height), 9), 2).toInt)
          canvasCopy.set(Tensor.fromRGB(TestUtil.resize(canvasCopy.get.toImage, width, height)))
          subreport.p("Input Parameters:")
          subreport.eval(() => {
            ArtistryUtil.toJson(styleSetup)
          })
          canvasCopy.set(textureGeneration.generate(subreport, canvasCopy.get, trainingMinutes, textureGeneration.measureStyle(styleSetup), maxIterations, isVerbose, styleSetup.precision))
        }
        canvasCopy.get
      })
      log.p(log.png(subresult.toImage, reportName))
    }
  }
}
