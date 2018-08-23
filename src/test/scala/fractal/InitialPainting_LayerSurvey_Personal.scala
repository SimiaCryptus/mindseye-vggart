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
import com.simiacryptus.sparkbook.EC2NotebookRunner


object InitialPainting_LayerSurvey_Personal extends EC2NotebookRunner(EC2NodeSettings.DeepLearningAMI, classOf[InitialPainting_LayerSurvey_Personal])

class InitialPainting_LayerSurvey_Personal extends InitialPainting_LayerSurvey(
  styleSources = Seq("s3a://simiacryptus/photos/shutterstock_1073629553.jpg")
) {
  override def dreamCoeff = 2e-1

  override def resolutionSchedule = Array[Int](200, 600)

  override def aspect_ratio = 0.61803398875

  override def plasma_magnitude = 1e-1

  override def getLayers = {
    InitialPainting_LayerSurvey.reduce((0 to 5).map(i =>
      List(
        CVPipe_VGG19.Layer.Layer_0,
        CVPipe_VGG19.Layer.Layer_1a,
        CVPipe_VGG19.Layer.Layer_1b,
        CVPipe_VGG19.Layer.Layer_1c)
    )
      .map(InitialPainting_LayerSurvey.wrap)
      .reduce(InitialPainting_LayerSurvey.join),
      1)
  }
}