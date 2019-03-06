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

package com.simiacryptus.mindseye.deep_dream;

import com.simiacryptus.aws.exe.EC2NotebookRunner;
import com.simiacryptus.aws.exe.LocalNotebookRunner;
import com.simiacryptus.mindseye.ImageScript;
import com.simiacryptus.mindseye.applications.ArtistryUtil;
import com.simiacryptus.mindseye.applications.DeepDream;
import com.simiacryptus.mindseye.applications.TiledTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.models.CVPipe_VGG19;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * The type Hi def.
 */
public class HiDef extends ImageScript {

  /**
   * The Resolution.
   */
  public int resolution = 1200;
  /**
   * The Content sources.
   */
  public String[] contentSources = {
      "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/The_Earth_seen_from_Apollo_17.jpg/1024px-The_Earth_seen_from_Apollo_17.jpg"
  };

  public void accept(@Nonnull NotebookOutput log) {
    DeepDream.VGG19 styleTransfer = new DeepDream.VGG19() {
      @Nonnull
      @Override
      public Trainable getTrainable(final PipelineNetwork network, final Tensor canvas) {
        return new TiledTrainable(network, canvas, 0);
      }
    };
    Precision precision = Precision.Float;
    styleTransfer.setTiled(false);
    Arrays.stream(contentSources).forEach(contentSource ->
    {
      log.p("Content Source:");
      log.p(log.png(ArtistryUtil.load(contentSource, resolution), "Content Image"));
      BufferedImage canvasImage;
      canvasImage = ArtistryUtil.load(contentSource, resolution);
      Map<CVPipe_VGG19.Layer, DeepDream.ContentCoefficients> dreamCoeff = new HashMap<>();
      dreamCoeff.put(CVPipe_VGG19.Layer.Layer_1d, new DeepDream.ContentCoefficients(0, 1e-1));
      dreamCoeff.put(CVPipe_VGG19.Layer.Layer_2b, new DeepDream.ContentCoefficients(0, 1e0));
      DeepDream.StyleSetup<CVPipe_VGG19.Layer> styleSetup = new DeepDream.StyleSetup<>(precision,
          ArtistryUtil.loadTensor(
              contentSource,
              canvasImage.getWidth(),
              canvasImage.getHeight()
          ), dreamCoeff
      );
      styleTransfer.deepDream(log.getHttpd(), log, Tensor.fromRGB(canvasImage), styleSetup, getTrainingMinutes(), getMaxIterations(), isVerbose());
    });
  }

  /**
   * The type Local.
   */
  public static class Local {
    /**
     * The entry point of application.
     *
     * @param args the input arguments
     * @throws Exception the exception
     */
    public static void main(String... args) throws Exception {
      LocalNotebookRunner.run(LocalNotebookRunner.getTask(HiDef.class));
    }
  }

  /**
   * The type Ec 2.
   */
  public static class EC2 {
    /**
     * The entry point of application.
     *
     * @param args the input arguments
     * @throws Exception the exception
     */
    public static void main(String... args) throws Exception {
      EC2NotebookRunner.run(LocalNotebookRunner.getTask(HiDef.class));
    }
  }

}
