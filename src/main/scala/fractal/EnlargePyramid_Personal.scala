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
import com.simiacryptus.sparkbook.{AWSNotebookRunner, EC2Runner}

object EnlargePyramid_Personal extends EnlargePyramid_Personal with EC2Runner with AWSNotebookRunner {
  override def nodeSettings: EC2NodeSettings = EC2NodeSettings.DeepLearningAMI
}

class EnlargePyramid_Personal extends EnlargePyramid(
  styleSources = Array("s3a://simiacryptus/photos/shutterstock_1065730331.jpg")
) {
  override val aspect = .59353

  override def inputHref: String = "https://" + bucket + ".s3.us-west-2.amazonaws.com/" + reportPath + "/etc/" + imagePrefix

  def imagePrefix: String = "tile_1_"

  def reportPath: String = "reports/20180812222258"

  def bucket: String = "mindseye-art-7f168"

  override def inputHadoop: String = "s3a://" + bucket + "/" + reportPath + "/etc/" + imagePrefix
}
