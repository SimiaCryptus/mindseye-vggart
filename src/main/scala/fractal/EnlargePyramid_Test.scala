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

import java.awt.image.BufferedImage
import java.util.function.Function

import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.NotebookRunner
import com.simiacryptus.sparkbook.util.Java8Util.toJavaFunction
import com.simiacryptus.sparkbook.util.LocalRunner

object EnlargePyramid_Test extends EnlargePyramid(
  styleSources = Array("s3a://simiacryptus/photos/shutterstock_1065730331.jpg")
) with LocalRunner[Object] with NotebookRunner[Object] {

  override val startLevel: Int = 1

  override val aspect = 632.0 / 1024.0

  override val inputHref: String = "https://" + bucket + ".s3.us-west-2.amazonaws.com/" + reportPath + "/etc/" + imagePrefix
  override val inputHadoop: String = "s3a://" + bucket + "/" + reportPath + "/etc/" + imagePrefix

  def imagePrefix: String = "tile_0_"

  def reportPath: String = "reports/20180826125743"

  def bucket: String = "mindseye-art-7f168"

  override def getImageEnlargingFunction(log: NotebookOutput, width: Int, height: Int, trainingMinutes: Int, maxIterations: Int, verbose: Boolean, magLevels: Int, padding: Int, styleSources: CharSequence*): Function[BufferedImage, BufferedImage] = {
    toJavaFunction((img: BufferedImage) => TestUtil.resize(img, img.getWidth * 2, true))
  }

}
