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

package fractal.etc

import com.simiacryptus.aws.exe.EC2NodeSettings
import com.simiacryptus.mindseye.applications.ArtistryUtil
import com.simiacryptus.mindseye.lang.Tensor
import com.simiacryptus.mindseye.models.CVPipe_VGG19
import com.simiacryptus.mindseye.test.TestUtil
import com.simiacryptus.notebook.NotebookOutput
import com.simiacryptus.sparkbook.util.Java8Util._
import com.simiacryptus.sparkbook.{AWSNotebookRunner, EC2Runner}
import fractal.InitialPainting


object InitialPainting_LayerSurvey extends InitialPainting_LayerSurvey(
  styleSources = Seq("s3a://simiacryptus/photos/shutterstock_1073629553.jpg")
) with EC2Runner[Object] with AWSNotebookRunner[Object] {
  override def nodeSettings: EC2NodeSettings = EC2NodeSettings.P3_2XL

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

  def wrap(xx: List[CVPipe_VGG19.Layer]): List[List[CVPipe_VGG19.Layer]] = xx.map((x: CVPipe_VGG19.Layer) => List(x))

  def join(a: List[List[CVPipe_VGG19.Layer]], b: List[List[CVPipe_VGG19.Layer]]): List[List[CVPipe_VGG19.Layer]] = a.flatMap((layerA: List[CVPipe_VGG19.Layer]) => b.map((layerB: List[CVPipe_VGG19.Layer]) => {
    (layerA ++ layerB).distinct.sorted
  }))

  def reduce(combined: List[List[CVPipe_VGG19.Layer]], size: Int): List[List[CVPipe_VGG19.Layer]] = combined.map(_.distinct.sorted).distinct.filter(_.size >= size).sortWith((a, b) => {
    var compare: Int = Integer.compare(a.size, b.size)
    var i: Int = 0
    while ( {
      0 == compare && i < a.size
    }) {
      val _i: Int = {
        i += 1;
        i - 1
      }
      compare = a(_i).name.compareTo(b(_i).name)
    }
    compare < 0
  })

}

abstract class InitialPainting_LayerSurvey(
                                            styleSources: Seq[CharSequence]
                                          ) extends InitialPainting(styleSources) {
  def getLayers: List[List[CVPipe_VGG19.Layer]]

  override def apply(log: NotebookOutput): Object = {
    TestUtil.addGlobalHandlers(log.getHttpd)
    for (styleSource <- styleSources) {
      log.p(log.png(ArtistryUtil.load(styleSource, style_resolution), "Style Image"))
    }
    val colorAligned: Tensor = log.subreport("Init", (output: NotebookOutput) => init(output))
    log.p(log.png(colorAligned.toImage, "Seed"))
    for (layers <- getLayers) {
      val reportName = layers.map((x: CVPipe_VGG19.Layer) => x.name).reduce(_ + "_" + _)
      log.h1(reportName)
      val painting = log.subreport(reportName, (output: NotebookOutput) => paint(output, colorAligned,
        r => getStyleSetup_TextureGeneration(precision, styleSources, style_resolution)).toImage)
      log.p(log.png(painting, reportName))
    }
    null
  }

}
