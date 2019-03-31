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
import com.simiacryptus.notebook.TableOutput;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

/**
 * The type Parameter sweep.
 */
public class ParameterSweep extends ImageScript {

  /**
   * The Resolution.
   */
  public int resolution = 400;
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
      "https://upload.wikimedia.org/wikipedia/commons/f/fb/Lightmatter_chimp.jpg"
  };

  /**
   * Content coeff stream double stream.
   *
   * @return the double stream
   */
  public DoubleStream contentCoeffStream() {
    return TestUtil.geometricStream(1e-1, 1e2, 3).get();
  }

  /**
   * Dream coeff stream double stream.
   *
   * @return the double stream
   */
  public DoubleStream dreamCoeffStream() {
    return TestUtil.geometricStream(1e-1, 1e1, 3).get();
  }

  public void accept(@Nonnull NotebookOutput log) {

    StyleTransfer.Inception styleTransfer = new StyleTransfer.Inception();
    Precision precision = Precision.Float;
    styleTransfer.parallelLossFunctions = true;
    styleTransfer.setTiled(false);
    TableOutput experimentTable = new TableOutput();
    Arrays.stream(contentSources).forEach(contentSource -> {
      log.p("Content Source:");
      log.p(log.png(ArtistryUtil.load(contentSource, resolution), "Content Image"));
      Stream<CharSequence> styleStream = Arrays.stream(styleSources);
      styleStream.map(x -> Arrays.asList((CharSequence) x)).forEach(sources -> {
        log.p("Style Source:");
        for (final CharSequence styleSource : sources) {
          log.p(log.png(ArtistryUtil.load(styleSource, resolution), "Style Image"));
        }
        //.set(CVPipe_Inception.Strata.Layer_1d, coeff_style_mean, coeff_style_cov, dreamCoeff)
        BufferedImage[] imgs = dreamCoeffStream().mapToObj(x -> x).flatMap(dreamCoeff -> {
          return contentCoeffStream().mapToObj(contentMixingCoeff -> {
            final Map<List<CharSequence>, StyleTransfer.StyleCoefficients<CVPipe_Inception.Strata>> styles = TestUtil.buildMap(
                x ->
                    x.put(
                        sources,
                        new StyleTransfer.StyleCoefficients<CVPipe_Inception.Strata>(
                            StyleTransfer.CenteringMode.Origin)
                            .set(
                                CVPipe_Inception.Strata.Layer_0,
                                coeff_style_mean,
                                coeff_style_cov,
                                dreamCoeff
                            )
                            .set(
                                CVPipe_Inception.Strata.Layer_2,
                                coeff_style_mean,
                                coeff_style_cov,
                                dreamCoeff
                            )
                            .set(
                                CVPipe_Inception.Strata.Layer_1b,
                                coeff_style_mean,
                                coeff_style_cov,
                                dreamCoeff
                            )
                            .set(
                                CVPipe_Inception.Strata.Layer_1c,
                                coeff_style_mean,
                                coeff_style_cov,
                                dreamCoeff
                            )
                        //.set(CVPipe_Inception.Strata.Layer_1d, coeff_style_mean, coeff_style_cov, dreamCoeff)
                    ));
            Tensor canvasImage = ArtistryUtil.loadTensor(
                contentSource,
                resolution
            );
            canvasImage = Tensor.fromRGB(TestUtil.resize(
                canvasImage.toImage(),
                resolution,
                true
            ));
            canvasImage = ArtistryUtil.expandPlasma(Tensor.fromRGB(
                TestUtil.resize(canvasImage.toImage(), 16, true)),
                1000.0, 1.1, resolution
            ).scale(0.9);
            StyleTransfer.StyleSetup<CVPipe_Inception.Strata> styleSetup = new StyleTransfer.StyleSetup<>(
                precision,
                ArtistryUtil.loadTensor(
                    contentSource,
                    canvasImage.getDimensions()[0],
                    canvasImage.getDimensions()[1]
                ),
                new StyleTransfer.ContentCoefficients<CVPipe_Inception.Strata>()
                    .set(CVPipe_Inception.Strata.Layer_2, contentMixingCoeff * 1e-1)
                    .set(CVPipe_Inception.Strata.Layer_1c, contentMixingCoeff)
                    .set(CVPipe_Inception.Strata.Layer_1d, contentMixingCoeff),
                TestUtil.buildMap(y -> y.putAll(styles.keySet().stream().flatMap(x -> x.stream())
                    .collect(Collectors.toMap(x -> x, file -> ArtistryUtil.load(file, resolution))))),
                styles
            );
            Tensor image = styleTransfer.transfer(
                log,
                canvasImage,
                styleSetup,
                getTrainingMinutes(),
                styleTransfer.measureStyle(
                    styleSetup),
                getMaxIterations(),
                isVerbose()
            );
            HashMap<CharSequence, Object> row = new HashMap<>();
            row.put(
                "Description",
                String.format(
                    "contentMixingCoeff=%s\ndreamCoeff=%s",
                    contentMixingCoeff,
                    dreamCoeff
                )
            );
            row.put("Image", log.png(image.toImage(), "image"));
            experimentTable.putRow(row);
            return image.toImage();
          });
        }).toArray(i -> new BufferedImage[i]);
        log.p("Summary Table:");
        log.p(experimentTable.toMarkdownTable());
        log.p("Animated Sequence:");
        log.p(TestUtil.animatedGif(log, imgs));
      });

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
      LocalNotebookRunner.run(LocalNotebookRunner.getTask(ParameterSweep.class));
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
      EC2NotebookRunner.run(LocalNotebookRunner.getTask(ParameterSweep.class));
    }
  }

}
