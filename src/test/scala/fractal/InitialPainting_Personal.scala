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

object InitialPainting_Personal extends EC2NotebookRunner(EC2NodeSettings.DeepLearningAMI, classOf[InitialPainting_Personal])

class InitialPainting_Personal extends InitialPainting(
  coeff_style_mean = 1e0,
  coeff_style_cov = 1e0,
  dreamCoeff = 2e-1,
  resolutionSchedule = Array[Int](200, 600),
  style_resolution = 1280,
  aspect_ratio = 0.61803398875,
  plasma_magnitude = 1e-1,
  styleSources = Seq("s3a://simiacryptus/photos/shutterstock_1073629553.jpg")
) {
  override def getLayers = {
    InitialPainting.reduce((0 to 5).map(i =>
      List(
        CVPipe_VGG19.Layer.Layer_0,
        CVPipe_VGG19.Layer.Layer_1a,
        CVPipe_VGG19.Layer.Layer_1b,
        CVPipe_VGG19.Layer.Layer_1c)
    )
      .map(InitialPainting.wrap)
      .reduce(InitialPainting.join),
      1)
  }
}