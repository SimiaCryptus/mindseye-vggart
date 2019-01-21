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
import com.simiacryptus.mindseye.models.CVPipe_VGG19
import com.simiacryptus.sparkbook.{AWSNotebookRunner, EC2Runner}

object InitialPyramid_AWS_EC2 extends InitialPyramid_AWS with EC2Runner[Object] {

  override def maxHeap: Option[String] = Option("55g")

  override val s3bucket: String = envTuple._2

  override def nodeSettings: EC2NodeSettings = EC2NodeSettings.P2_XL

}

abstract class InitialPyramid_AWS extends InitialPyramid(
    initialContent = "https://mindseye-art-7f168.s3.us-west-2.amazonaws.com/reports/201809145834/etc/fractal.InitialPainting_AWS.6.png",
    styleSources = Array("s3a://simiacryptus/photos/shutterstock_157227299.jpg")
  ) with AWSNotebookRunner[Object] {

  override val maxIterations: Int = 10

  override def style_layers(layer: CVPipe_VGG19.Layer): Double = layer match {
    case CVPipe_VGG19.Layer.Layer_1a => 1e0
    case CVPipe_VGG19.Layer.Layer_1b => 1e0
    case _ => 0.0
  }

  override def coeff_content(layer: CVPipe_VGG19.Layer): Double = layer match {
    case CVPipe_VGG19.Layer.Layer_0 => 1e-1
    case _ => 0e0
  }

  override def dreamCoeff(layer: CVPipe_VGG19.Layer): Double = layer match {
    case CVPipe_VGG19.Layer.Layer_0 => 0e0
    case CVPipe_VGG19.Layer.Layer_1a => 1e-1
    case CVPipe_VGG19.Layer.Layer_1b => 4e-1
    case _ => 0e0
  }

}


