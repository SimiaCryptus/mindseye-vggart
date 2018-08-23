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

class EnlargePyramid(val bucket: String,
                     val reportPath: String,
                     val imagePrefix: String,
                     val styleSources: Array[CharSequence]
                    ) extends Tendril.SerializableConsumer[NotebookOutput] {

  override def accept(log: NotebookOutput): Unit = {
    TestUtil.addGlobalHandlers(log.getHttpd)
    log.asInstanceOf[MarkdownNotebookOutput].setMaxImageSize(10000)
    log.run(() => {
      JsonUtil.toJson(EnlargePyramid.this)
    }: Unit)
    try {
      PyramidUtil.initJS(log)
      val hrefPrefix: String = "https://" + bucket + ".s3.us-west-2.amazonaws.com/" + reportPath + "/etc/" + imagePrefix
      val hadoopPrefix: String = "s3a://" + bucket + "/" + reportPath + "/etc/" + imagePrefix
      PyramidUtil.writeViewer(log, "source", PyramidUtil.getTilesource(tileSize, 0, startLevel, hrefPrefix, aspect))
      val imageFunction = PyramidUtil.getImageEnlargingFunction(log,
        ((tileSize + 2 * padding) * Math.pow(2, magLevels)).toInt,
        ((tileSize + 2 * padding) * Math.pow(2, magLevels)).toInt,
        trainingMinutes, maxIterations, verbose, magLevels, padding, styleSources)
      PyramidUtil.buildNewImagePyramidLayer(startLevel, magLevels, padding, tileSize, hadoopPrefix, imageFunction, aspect, false)
      val totalSize: Double = tileSize * Math.pow(2, startLevel + magLevels)
      PyramidUtil.writeViewer(log, imagePrefix + "%d_%d".format(startLevel, magLevels + startLevel),
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

  def tileSize: Int = 512

  def magLevels: Int = 1

  def padding: Int = 20

  def startLevel: Int = 3

  def aspect: Double = 1.0

  def trainingMinutes: Int = 10

  def maxIterations: Int = 20

  def verbose: Boolean = false


}
