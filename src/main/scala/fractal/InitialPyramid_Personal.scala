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

import com.simiacryptus.aws.exe.EC2NodeSettings
import com.simiacryptus.mindseye.models.CVPipe_VGG19
import com.simiacryptus.sparkbook.{AWSNotebookRunner, EC2Runner, LocalRunner, NotebookRunner}

object InitialPyramid_Personal extends InitialPyramid_Personal with EC2Runner with AWSNotebookRunner {
  override def nodeSettings: EC2NodeSettings = EC2NodeSettings.DeepLearningAMI

  override def style_layers(layer: CVPipe_VGG19.Layer): Double = layer match {
    case CVPipe_VGG19.Layer.Layer_1a => 1e0
    case CVPipe_VGG19.Layer.Layer_1b => 1e0
    case _ => 0.0
  }

  override def coeff_content(layer: CVPipe_VGG19.Layer): Double = layer match {
    case CVPipe_VGG19.Layer.Layer_0 => 1e0
    case _ => 0e0
  }

  override def dreamCoeff(layer: CVPipe_VGG19.Layer): Double = layer match {
    case CVPipe_VGG19.Layer.Layer_0 => 0e0
    case CVPipe_VGG19.Layer.Layer_1a => 1e-1
    case CVPipe_VGG19.Layer.Layer_1b => 1e1
    case _ => 0e0
  }
}

object InitialPyramid_Personal_Local extends InitialPyramid_Personal with LocalRunner with NotebookRunner {

  override def style_resolution: Int = 600

  override def trainingMinutes: Int = 1

  override def maxIterations: Int = 1

}

class InitialPyramid_Personal extends InitialPyramid(
  initialContent = "https://mindseye-art-7f168.s3.us-west-2.amazonaws.com/reports/20180824155832/etc/fractal.InitialPainting_Personal.6.png",
  styleSources = Array("s3a://simiacryptus/photos/shutterstock_1065730331.jpg")
)
