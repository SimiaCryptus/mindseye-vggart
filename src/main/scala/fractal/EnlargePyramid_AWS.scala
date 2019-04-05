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

import com.simiacryptus.aws.exe.EC2NodeSettings
import com.simiacryptus.mindseye.models.CVPipe_Inception
import com.simiacryptus.sparkbook.{AWSNotebookRunner, EC2Runner}

object EnlargePyramid_AWS_EC2 extends EnlargePyramid_AWS with EC2Runner[Object] {

  override val s3bucket: String = envTuple._2

  override def nodeSettings: EC2NodeSettings = EC2NodeSettings.P2_XL

  override def maxHeap: Option[String] = Option("55g")
}


abstract class EnlargePyramid_AWS extends EnlargePyramid(
  styleSources = Array("s3a://simiacryptus/photos/shutterstock_781159663.jpg")
) with AWSNotebookRunner[Object] {
  override val s3bucket: String = super.s3bucket

  override val startLevel: Int = 2
  override val style_resolution: Int = 1200
  override val maxIterations: Int = 10
  override val aspect = 632.0 / 1024.0
  override val inputHref: String = "https://" + bucket + ".s3.us-west-2.amazonaws.com/" + reportPath + "/etc/" + imagePrefix
  override val inputHadoop: String = "s3a://" + bucket + "/" + reportPath + "/etc/" + imagePrefix

  override def imagePrefix: String = "tile_"

  def bucket: String = "data-cb03c"

  def reportPath: String = "reports/201903275317"

  override def style_layers(layer: CVPipe_Inception.Strata): Double = layer match {
    //    case CVPipe_Inception.Strata.Layer_2 => 1e0
    //    case CVPipe_Inception.Strata.Layer_3a => 1e0
    case CVPipe_Inception.Strata.Layer_3b => 1e0
    case CVPipe_Inception.Strata.Layer_4a => 1e0
    case CVPipe_Inception.Strata.Layer_4b => 1e0
    case _ => 0.0
  }

  override def coeff_content(layer: CVPipe_Inception.Strata): Double = layer match {
    case CVPipe_Inception.Strata.Layer_1 => 1e-1
    case _ => 0e0
  }

  override def dreamCoeff(layer: CVPipe_Inception.Strata): Double = layer match {
    case CVPipe_Inception.Strata.Layer_3b => 0e0
    case CVPipe_Inception.Strata.Layer_3a => 5e-1
    case CVPipe_Inception.Strata.Layer_4a => 1e0
    case CVPipe_Inception.Strata.Layer_4b => 1e0
    case CVPipe_Inception.Strata.Layer_4c => 1e0
    case _ => 0e0
  }
}

