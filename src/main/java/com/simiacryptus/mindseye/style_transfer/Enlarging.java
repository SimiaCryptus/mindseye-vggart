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
import com.simiacryptus.mindseye.models.CVPipe_VGG19;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

/**
 * The type Enlarging.
 */
public class Enlarging extends ImageScript {

  /**
   * The Start png size.
   */
  public int startImageSize = 250;
  /**
   * The Coeff style mean.
   */
  public double coeff_style_mean = 1e1;
  /**
   * The Coeff style bandCovariance.
   */
  public double coeff_style_cov = 1e0;
  /**
   * The Style sources.
   */
  public String[] styleSources = {
      "git://github.com/jcjohnson/fast-neural-style.git/master/images/styles/starry_night_crop.jpg"
  };
  /**
   * The Content sources.
   */
  public String[] contentSources = {
      "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/The_Earth_seen_from_Apollo_17.jpg/1024px-The_Earth_seen_from_Apollo_17.jpg"
  };

  /**
   * Init buffered png.
   *
   * @param contentSource the content source
   * @param width         the width
   * @return the buffered png
   */
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

  /**
   * Resolution stream double stream.
   *
   * @return the double stream
   */
  public DoubleStream resolutionStream() {
    return TestUtil.geometricStream(startImageSize, 800, 3).get();
  }

  public void accept(@Nonnull NotebookOutput log) {

    StyleTransfer.VGG19 styleTransfer = new StyleTransfer.VGG19();
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
      //.set(CVPipe_VGG19.Layer.Layer_1d, coeff_style_mean, coeff_style_cov, dreamCoeff)
      double contentMixingCoeff = 1e1;
      double dreamCoeff = 1e-1;
      styleTransfer(log, styleTransfer, precision,
          contentSource,
          TestUtil.buildMap(x ->
              x.put(
                  Arrays.asList(styleSources),
                  new StyleTransfer.StyleCoefficients<CVPipe_VGG19.Layer>(StyleTransfer.CenteringMode.Origin)
                      .set(CVPipe_VGG19.Layer.Layer_0, coeff_style_mean, coeff_style_cov, dreamCoeff)
                      .set(CVPipe_VGG19.Layer.Layer_1a, coeff_style_mean, coeff_style_cov, dreamCoeff)
                      .set(CVPipe_VGG19.Layer.Layer_1b, coeff_style_mean, coeff_style_cov, dreamCoeff)
                      .set(CVPipe_VGG19.Layer.Layer_1c, coeff_style_mean, coeff_style_cov, dreamCoeff)
                  //.set(CVPipe_VGG19.Layer.Layer_1d, coeff_style_mean, coeff_style_cov, dreamCoeff)
              )),
          new StyleTransfer.ContentCoefficients<CVPipe_VGG19.Layer>()
              .set(CVPipe_VGG19.Layer.Layer_1a, contentMixingCoeff * 1e-1)
              .set(CVPipe_VGG19.Layer.Layer_1c, contentMixingCoeff)
              .set(CVPipe_VGG19.Layer.Layer_1d, contentMixingCoeff),
          getTrainingMinutes(), getMaxIterations());

    });
  }

  /**
   * Style transfer buffered png.
   *
   * @param log                 the log
   * @param styleTransfer       the style transfer
   * @param precision           the precision
   * @param contentSource       the content source
   * @param styles              the styles
   * @param contentCoefficients the content coefficients
   * @param trainingMinutes     the training minutes
   * @param maxIterations       the max iterations
   * @return the buffered png
   */
  public Tensor styleTransfer(
      @Nonnull final NotebookOutput log,
      final StyleTransfer.VGG19 styleTransfer,
      final Precision precision,
      final CharSequence contentSource,
      final Map<List<CharSequence>, StyleTransfer.StyleCoefficients<CVPipe_VGG19.Layer>> styles,
      final StyleTransfer.ContentCoefficients<CVPipe_VGG19.Layer> contentCoefficients,
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
      StyleTransfer.StyleSetup<CVPipe_VGG19.Layer> styleSetup = new StyleTransfer.StyleSetup<>(precision,
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
      LocalNotebookRunner.run(LocalNotebookRunner.getTask(Enlarging.class));
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
      EC2NotebookRunner.run(LocalNotebookRunner.getTask(Enlarging.class));
    }
  }

}
