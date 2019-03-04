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

import java.io.{ByteArrayOutputStream, OutputStream, PrintStream}

import com.simiacryptus.aws.exe.EC2NodeSettings
import com.simiacryptus.mindseye.lang.CoreSettings
import com.simiacryptus.mindseye.lang.cudnn.{CudaMemory, CudaSettings, CudaSystem}
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.{EC2SparkRunner, WorkerRunner}
import com.simiacryptus.util.JsonUtil
import util.Java8Util._

import scala.reflect.ClassTag

object EnlargePyramid_AWS_Spark extends EnlargePyramid_AWS with EC2SparkRunner[Object] {

  override def hiveRoot: Option[String] = super.hiveRoot

  override def sparkProperties: Map[String, String] = super.sparkProperties

  override def javaProperties: Map[String, String] = super.javaProperties ++ Map(
    "MAX_TOTAL_MEMORY" -> (8 * CudaMemory.GiB).toString,
    "MAX_DEVICE_MEMORY" -> (8 * CudaMemory.GiB).toString,
    "MAX_IO_ELEMENTS" -> (2 * CudaMemory.MiB).toString,
    "CONVOLUTION_WORKSPACE_SIZE_LIMIT" -> (1 * 512 * CudaMemory.MiB).toString,
    "MAX_FILTER_ELEMENTS" -> (1 * 512 * CudaMemory.MiB).toString,
    "java.util.concurrent.ForkJoinPool.common.parallelism" -> Integer.toString(CoreSettings.INSTANCE().jvmThreads)
  )


  override val s3bucket: String = envTuple._2

  override def masterSettings: EC2NodeSettings = EC2NodeSettings.M5_XL

  override def workerSettings: EC2NodeSettings = EC2NodeSettings.P2_8XL

  override val driverMemory: String = "15g"

  override val workerMemory: String = "60g"

  override val numberOfWorkersPerNode: Int = 8

  override val numberOfWorkerNodes: Int = 1

  override val workerCores: Int = 1

  override def apply(log: NotebookOutput): AnyRef = {
    log.getHttpd.addGET("gpu.txt", "text/plain", (out: OutputStream) => {
      val printStream = new PrintStream(out)
      WorkerRunner.distributeEval(idx => {
        Thread.sleep(5000)
        val outputStream = new ByteArrayOutputStream()
        CudaSystem.printHeader(new PrintStream(outputStream))
        s"""CudaSettings.INSTANCE = ${JsonUtil.toJson(CudaSettings.INSTANCE)}
            |Cuda System Header = ${outputStream.toString}
            |""".stripMargin
      })(ClassTag.apply(classOf[String]), spark).foreach(printStream.println(_))
      printStream.close()
    }: Unit)
    super.apply(log)
  }
}

