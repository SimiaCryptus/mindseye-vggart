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

import com.simiacryptus.mindseye.applications.{ImageArtUtil, SegmentedStyleTransfer, TextureGeneration}
import com.simiacryptus.mindseye.lang.cudnn.Precision
import com.simiacryptus.mindseye.models.CVPipe_Inception

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

trait StyleTransferParams {

  def getStyleSetup_TextureGeneration(precision: Precision, styleSources: Seq[CharSequence], style_resolution: Int): TextureGeneration.StyleSetup[CVPipe_Inception.Strata] = {
    val styleCoefficients = new TextureGeneration.StyleCoefficients[CVPipe_Inception.Strata](TextureGeneration.CenteringMode.Origin)
    CVPipe_Inception.Strata.values().foreach((layer: CVPipe_Inception.Strata) => styleCoefficients.set(layer, coeff_style_mean(layer), coeff_style_cov(layer), dreamCoeff(layer)))
    new TextureGeneration.StyleSetup[CVPipe_Inception.Strata](
      precision,
      ImageArtUtil.getStyleImages(style_resolution, styleSources: _*).mapValues(_.toImage),
      Map(styleSources.toList.asJava -> styleCoefficients)
    )
  }

  def getStyleSetup_TextureGeneration2(precision: Precision, styleSources: Seq[CharSequence], style_resolution: Int): TextureGeneration.StyleSetup[CVPipe_Inception.Strata] = {
    val styleCoefficients = new TextureGeneration.StyleCoefficients[CVPipe_Inception.Strata](TextureGeneration.CenteringMode.Origin)
    CVPipe_Inception.Strata.values().foreach((layer: CVPipe_Inception.Strata) => styleCoefficients.set(layer, coeff_style_mean(layer), coeff_style_cov(layer), dreamCoeff(layer)))
    new TextureGeneration.StyleSetup[CVPipe_Inception.Strata](
      precision,
      ImageArtUtil.getStyleImages(style_resolution, styleSources: _*).mapValues(_.toImage),
      Map(styleSources.toList.asJava -> styleCoefficients)
    )
  }

  def getStyleSetup_SegmentedStyleTransfer(precision: Precision, styleSources: Seq[CharSequence], style_resolution: Int): SegmentedStyleTransfer.StyleSetup[CVPipe_Inception.Strata] = {
    val contentCoefficients: SegmentedStyleTransfer.ContentCoefficients[CVPipe_Inception.Strata] = new SegmentedStyleTransfer.ContentCoefficients[CVPipe_Inception.Strata]
    CVPipe_Inception.Strata.values().foreach((layer: CVPipe_Inception.Strata) => contentCoefficients.set(CVPipe_Inception.Strata.Layer_1, coeff_content(layer)))
    val styleCoefficients = new SegmentedStyleTransfer.StyleCoefficients[CVPipe_Inception.Strata](SegmentedStyleTransfer.CenteringMode.Origin)
    CVPipe_Inception.Strata.values().foreach((layer: CVPipe_Inception.Strata) => styleCoefficients.set(layer, coeff_style_mean(layer), coeff_style_cov(layer), dreamCoeff(layer)))
    new SegmentedStyleTransfer.StyleSetup[CVPipe_Inception.Strata](
      precision,
      null,
      contentCoefficients,
      ImageArtUtil.getStyleImages(style_resolution, styleSources: _*),
      Map(styleSources.toList.asJava -> styleCoefficients)
    )
  }

  def dreamCoeff(layer: CVPipe_Inception.Strata) = 5e-1 * style_layers(layer)

  //  def dreamCoeff(layer: CVPipe_Inception.Strata) = 5e-1 * style_layers(layer)

  //  def coeff_style_cov(layer: CVPipe_Inception.Strata) = 1e0 * style_layers(layer)

  //  def style_layers(layer: CVPipe_Inception.Strata): Double = layer match {
  //    case CVPipe_Inception.Strata.Layer_1 => 1e0
  //    case CVPipe_Inception.Strata.Layer_2 => 1e0
  //    case _ => 0.0
  //  }

  def style_layers(layer: CVPipe_Inception.Strata): Double = layer match {
    case CVPipe_Inception.Strata.Layer_1 => 1e0
    case CVPipe_Inception.Strata.Layer_2 => 1e0
    case _ => 0.0
  }

  //  def coeff_style_mean(layer: CVPipe_Inception.Strata) = 1e0 * style_layers(layer)

  def coeff_style_cov(layer: CVPipe_Inception.Strata) = 1e0 * style_layers(layer)

  def coeff_style_mean(layer: CVPipe_Inception.Strata) = 1e0 * style_layers(layer)

  def coeff_content(layer: CVPipe_Inception.Strata) = layer match {
    case CVPipe_Inception.Strata.Layer_1 => 1e-1
    case CVPipe_Inception.Strata.Layer_2 => 1e-1
  }

}
