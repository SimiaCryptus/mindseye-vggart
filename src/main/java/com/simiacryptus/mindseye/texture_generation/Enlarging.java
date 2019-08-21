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

package com.simiacryptus.mindseye.texture_generation;

import com.simiacryptus.aws.exe.EC2NotebookRunner;
import com.simiacryptus.aws.exe.LocalNotebookRunner;
import com.simiacryptus.mindseye.ImageScript;
import com.simiacryptus.mindseye.applications.ArtistryUtil;
import com.simiacryptus.mindseye.applications.TextureGeneration;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.models.CVPipe_Inception;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

public class Enlarging extends ImageScript {


  public int startImageSize = 200;
  public double coeff_style_mean = 1e1;
  public double coeff_style_cov = 1e0;
  public String[] styleSources = {
      "git://github.com/jcjohnson/fast-neural-style.git/master/images/styles/starry_night_crop.jpg"
  };

  @Nonnull
  public BufferedImage init(final int width) {
    return ArtistryUtil.paint_Plasma(3, 1000.0, 1.1, width).toImage();
  }

  public DoubleStream resolutionStream() {
    return TestUtil.geometricStream(startImageSize, 800, 4).get();
  }

  public void accept(@Nonnull NotebookOutput log) {

    TextureGeneration.Inception textureGeneration = new TextureGeneration.Inception();
    Precision precision = Precision.Float;
    textureGeneration.parallelLossFunctions = true;
    textureGeneration.setTiling(3);
    log.p("Style Source:");
    for (final CharSequence styleSource : styleSources) {
      log.p(log.png(ArtistryUtil.load(styleSource, startImageSize), "Style Image"));
    }
    double dreamCoeff = 1e1;
    final Map<List<CharSequence>, TextureGeneration.StyleCoefficients<CVPipe_Inception.Strata>> styles = TestUtil.buildMap(x ->
        x.put(
            Arrays.asList(styleSources),
            new TextureGeneration.StyleCoefficients<CVPipe_Inception.Strata>(
                TextureGeneration.CenteringMode.Origin)
                .set(CVPipe_Inception.Strata.Layer_0, coeff_style_mean, coeff_style_cov, dreamCoeff)
                .set(CVPipe_Inception.Strata.Layer_2, coeff_style_mean, coeff_style_cov, dreamCoeff)
                .set(CVPipe_Inception.Strata.Layer_1b, coeff_style_mean, coeff_style_cov, dreamCoeff)
                .set(CVPipe_Inception.Strata.Layer_1c, coeff_style_mean, coeff_style_cov, dreamCoeff)
        ));
    Tensor canvasImage = null;
    for (final double resolution : resolutionStream().toArray()) {
      int size = (int) resolution;
      if (null == canvasImage) {
        canvasImage = Tensor.fromRGB(init(size));
      } else {
        canvasImage = Tensor.fromRGB(ImageUtil.resize(canvasImage.toImage(), size, true));
      }
      TextureGeneration.StyleSetup<CVPipe_Inception.Strata> styleSetup = new TextureGeneration.StyleSetup<CVPipe_Inception.Strata>(precision,
          TestUtil.buildMap(y -> y.putAll(styles.keySet().stream().flatMap(x1 -> x1.stream())
              .collect(Collectors.toMap(x1 -> x1, file -> ArtistryUtil.load(file, size))))), styles);
      log.p("Input Parameters:");
      log.eval(() -> {
        return ArtistryUtil.toJson(styleSetup);
      });
      PipelineNetwork network = textureGeneration.fitnessNetwork(textureGeneration.measureStyle(styleSetup));
      canvasImage = TextureGeneration.optimize(
          log,
          network,
          canvasImage,
          getTrainingMinutes(),
          getMaxIterations(),
          isVerbose(),
          styleSetup.precision,
          textureGeneration.getTiling()
      );
    }
  }

  public static class Local {
    public static void main(String... args) throws Exception {
      LocalNotebookRunner.run(LocalNotebookRunner.getTask(Enlarging.class));
    }
  }

  public static class EC2 {
    public static void main(String... args) throws Exception {
      EC2NotebookRunner.run(LocalNotebookRunner.getTask(Enlarging.class));
    }
  }

}
