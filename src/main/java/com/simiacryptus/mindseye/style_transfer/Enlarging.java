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

package com.simiacryptus.mindseye.style_transfer;

import com.simiacryptus.aws.exe.EC2NotebookRunner;
import com.simiacryptus.aws.exe.LocalNotebookRunner;
import com.simiacryptus.mindseye.ImageScript;
import com.simiacryptus.mindseye.applications.ArtistryUtil;
import com.simiacryptus.mindseye.applications.StyleTransfer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.models.CVPipe_Inception;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

public class Enlarging extends ImageScript {

  public int startImageSize = 250;
  public double coeff_style_mean = 1e1;
  public double coeff_style_cov = 1e0;
  public String[] styleSources = {
      "git://github.com/jcjohnson/fast-neural-style.git/master/images/styles/starry_night_crop.jpg"
  };
  public String[] contentSources = {
      "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/The_Earth_seen_from_Apollo_17.jpg/1024px-The_Earth_seen_from_Apollo_17.jpg"
  };

  @Nonnull
  public BufferedImage init(final CharSequence contentSource, final int width) {
    BufferedImage canvasImage;
    canvasImage = ArtistryUtil.load(contentSource, width);
    canvasImage = TestUtil.resize(canvasImage, width, true);
    canvasImage = ArtistryUtil.expandPlasma(Tensor.fromRGB(
        TestUtil.resize(canvasImage, 16, true)),
        1000.0, 1.1, width
    ).scale(0.9).toImage();
    return canvasImage;
  }

  public DoubleStream resolutionStream() {
    return TestUtil.geometricStream(startImageSize, 800, 3).get();
  }

  public void accept(@Nonnull NotebookOutput log) {

    StyleTransfer.Inception styleTransfer = new StyleTransfer.Inception();
    Precision precision = Precision.Float;
    styleTransfer.parallelLossFunctions = true;
    styleTransfer.setTiled(false);
    Arrays.stream(contentSources).forEach(contentSource ->
    {
      log.p("Content Source:");
      log.p(log.png(ArtistryUtil.load(contentSource, startImageSize), "Content Image"));
      log.p("Style Source:");
      for (final CharSequence styleSource : styleSources) {
        log.p(log.png(ArtistryUtil.load(styleSource, startImageSize), "Style Image"));
      }
      //.set(CVPipe_Inception.Strata.Layer_1d, coeff_style_mean, coeff_style_cov, dreamCoeff)
      double contentMixingCoeff = 1e1;
      double dreamCoeff = 1e-1;
      styleTransfer(log, styleTransfer, precision,
          contentSource,
          TestUtil.buildMap(x ->
              x.put(
                  Arrays.asList(styleSources),
                  new StyleTransfer.StyleCoefficients<CVPipe_Inception.Strata>(StyleTransfer.CenteringMode.Origin)
                      .set(CVPipe_Inception.Strata.Layer_0, coeff_style_mean, coeff_style_cov, dreamCoeff)
                      .set(CVPipe_Inception.Strata.Layer_2, coeff_style_mean, coeff_style_cov, dreamCoeff)
                      .set(CVPipe_Inception.Strata.Layer_1b, coeff_style_mean, coeff_style_cov, dreamCoeff)
                      .set(CVPipe_Inception.Strata.Layer_1c, coeff_style_mean, coeff_style_cov, dreamCoeff)
                  //.set(CVPipe_Inception.Strata.Layer_1d, coeff_style_mean, coeff_style_cov, dreamCoeff)
              )),
          new StyleTransfer.ContentCoefficients<CVPipe_Inception.Strata>()
              .set(CVPipe_Inception.Strata.Layer_2, contentMixingCoeff * 1e-1)
              .set(CVPipe_Inception.Strata.Layer_1c, contentMixingCoeff)
              .set(CVPipe_Inception.Strata.Layer_1d, contentMixingCoeff),
          getTrainingMinutes(), getMaxIterations());

    });
  }

  public Tensor styleTransfer(
      @Nonnull final NotebookOutput log,
      final StyleTransfer.Inception styleTransfer,
      final Precision precision,
      final CharSequence contentSource,
      final Map<List<CharSequence>, StyleTransfer.StyleCoefficients<CVPipe_Inception.Strata>> styles,
      final StyleTransfer.ContentCoefficients<CVPipe_Inception.Strata> contentCoefficients,
      final int trainingMinutes,
      final int maxIterations
  ) {
    Tensor canvasImage = null;
    for (final Double resolution : resolutionStream().mapToObj(x -> x).collect(Collectors.toList())) {
      int size = (int) (double) resolution;
      if (null == canvasImage) {
        canvasImage = Tensor.fromRGB(init(contentSource, size));
      } else {
        canvasImage = Tensor.fromRGB(TestUtil.resize(canvasImage.toImage(), size, true));
      }
      StyleTransfer.StyleSetup<CVPipe_Inception.Strata> styleSetup = new StyleTransfer.StyleSetup<>(precision,
          ArtistryUtil.loadTensor(
              contentSource,
              canvasImage.getDimensions()[0],
              canvasImage.getDimensions()[1]
          ),
          contentCoefficients, TestUtil.buildMap(y -> y.putAll(styles.keySet().stream().flatMap(x -> x.stream())
          .collect(Collectors.toMap(x -> x, file -> ArtistryUtil.load(file, size))))), styles);
      canvasImage = styleTransfer.transfer(log, canvasImage, styleSetup,
          trainingMinutes, styleTransfer.measureStyle(styleSetup), maxIterations, isVerbose()
      );
    }
    return canvasImage;
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
