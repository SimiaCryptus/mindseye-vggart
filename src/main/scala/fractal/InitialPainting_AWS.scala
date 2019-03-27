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
import com.simiacryptus.mindseye.lang.cudnn.CudaMemory
import com.simiacryptus.sparkbook.util.LocalRunner
import com.simiacryptus.sparkbook.{AWSNotebookRunner, EC2Runner, NotebookRunner}

object InitialPainting_AWS extends InitialPainting(
  styleSources = Seq(
    "s3a://simiacryptus/photos/shutterstock_1060865300.jpg"
  )
) with EC2Runner[Object] with AWSNotebookRunner[Object] {
  override def nodeSettings: EC2NodeSettings = EC2NodeSettings.P2_XL

  override def maxHeap: Option[String] = Option("60g")

  override def javaProperties: Map[String, String] = Map(
    "spark.master" -> "local[4]",
    "MAX_TOTAL_MEMORY" -> (8 * CudaMemory.GiB).toString,
    "MAX_DEVICE_MEMORY" -> (8 * CudaMemory.GiB).toString,
    "MAX_IO_ELEMENTS" -> (2 * CudaMemory.MiB).toString,
    "CONVOLUTION_WORKSPACE_SIZE_LIMIT" -> (1 * 512 * CudaMemory.MiB).toString,
    "MAX_FILTER_ELEMENTS" -> (1 * 512 * CudaMemory.MiB).toString
  )

  override def aspect_ratio = 0.61803398875

  override def plasma_magnitude = 1e-1
}

object InitialPainting_Local extends InitialPainting(
  styleSources = Seq(
    "s3a://simiacryptus/photos/shutterstock_781159663.jpg"
  )
) with LocalRunner[Object] with NotebookRunner[Object] {

  override def aspect_ratio = 0.61803398875

  override def plasma_magnitude = 1e-1
}
