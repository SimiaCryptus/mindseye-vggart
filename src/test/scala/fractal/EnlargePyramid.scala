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

import com.simiacryptus.aws.Tendril
import com.simiacryptus.mindseye.pyramid.PyramidUtil
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.util.io.{JsonUtil, MarkdownNotebookOutput, NotebookOutput}
import util.Java8Util._

class EnlargePyramid(var tileSize: Int = 512,
                     var magLevels: Int = 1,
                     var padding: Int = 20,
                     val startLevel: Int = 3,
                     val bucket: String,
                     val reportPath: String,
                     val localPrefix: String,
                     val aspect: Double = 1.0,
                     val styleSources: Array[CharSequence],
                     val trainingMinutes: Int = 10,
                     val maxIterations: Int = 20,
                     var verbose: Boolean = false
                    ) extends Tendril.SerializableConsumer[NotebookOutput] {


  override def accept(log: NotebookOutput): Unit = {
    TestUtil.addGlobalHandlers(log.getHttpd)
    log.asInstanceOf[MarkdownNotebookOutput].setMaxImageSize(10000)
    log.run(() => {
      JsonUtil.toJson(EnlargePyramid.this)
    }: Unit)
    try {
      PyramidUtil.initJS(log)
      val hrefPrefix: String = "https://" + bucket + ".s3.us-west-2.amazonaws.com/" + reportPath + "/etc/" + localPrefix
      val hadoopPrefix: String = "s3a://" + bucket + "/" + reportPath + "/etc/" + localPrefix
      PyramidUtil.writeViewer(log, "source", PyramidUtil.getTilesource(tileSize, 0, startLevel, hrefPrefix, aspect))
      val imageFunction = PyramidUtil.getImageEnlargingFunction(log,
        ((tileSize + 2 * padding) * Math.pow(2, magLevels)).toInt,
        ((tileSize + 2 * padding) * Math.pow(2, magLevels)).toInt,
        trainingMinutes, maxIterations, verbose, magLevels, padding, styleSources)
      PyramidUtil.buildNewImagePyramidLayer(startLevel, magLevels, padding, tileSize, hadoopPrefix, imageFunction, aspect, false)
      val totalSize: Double = tileSize * Math.pow(2, startLevel + magLevels)
      PyramidUtil.writeViewer(log, localPrefix + "%d_%d".format(startLevel, magLevels + startLevel),
        s"""{
           |        height: ${(totalSize * aspect).toInt},
           |        width:  ${totalSize.toInt},
           |        tileSize: $tileSize,
           |        minLevel: 0,
           |        maxLevel: ${(startLevel + magLevels)},
           |        getTileUrl: function( level, x, y ){
           |            return "$hrefPrefix" + level + "_" + y + "_" + x + ".jpg";
           |        }
           |    }""".stripMargin)
    } catch {
      case throwable: Throwable =>
        log.eval(() => {
          throwable
        })
    }
  }


}
