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

import com.simiacryptus.lang.SerializableFunction
import com.simiacryptus.mindseye.pyramid._
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.notebook.{MarkdownNotebookOutput, NotebookOutput}
import com.simiacryptus.sparkbook.NotebookRunner
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.util.{LocalRunner, ScalaJson}

object GoogleMarsPyramidImport extends GoogleMarsPyramidImport with LocalRunner[Unit] with NotebookRunner[Unit]

class GoogleMarsPyramidImport() extends SerializableFunction[NotebookOutput, Unit] {

  val level = 2

  override def apply(log: NotebookOutput): Unit = {
    TestUtil.addGlobalHandlers(log.getHttpd)
    log.asInstanceOf[MarkdownNotebookOutput].setMaxImageSize(8 * 1024)
    log.eval(() => {
      ScalaJson.toJson(GoogleMarsPyramidImport.this)
    })
    try {
      PyramidUtil.initJS(log)
      val sampler = GoogleMaps.Mars.Elevation.getValueSampler(level)
      out(log, sampler, new ImagePyramid(512, level, 1.0, "mars_primary"),
        wrapVertical = false)
      out(log, sampler.offset(0.25, 0.25).rotate(Math.PI / 6).zoom(0.1, 0.1),
        new ImagePyramid(512, level, 1.0, "mars_zoom"),
        wrapHorizontal = false, wrapVertical = false)
    } catch {
      case throwable: Throwable =>
        log.eval(() => {
          throwable
        })
    }
  }

  private def out(log: NotebookOutput, sampler: ValueSampler, primary: ImagePyramid, wrapHorizontal: Boolean = true, wrapVertical: Boolean = true) = {
    primary.write(log, sampler).rebuild(0)
    PyramidUtil.writeViewer(log, primary.getPrefix, wrapHorizontal, wrapVertical, primary)
    log.p(log.jpg(primary.logRelative(log).assemble(2 * 1024), primary.getPrefix))
  }
}
