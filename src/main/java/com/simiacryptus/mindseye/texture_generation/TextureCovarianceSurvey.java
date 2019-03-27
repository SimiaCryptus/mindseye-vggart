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

import com.simiacryptus.mindseye.ImageScript;
import com.simiacryptus.mindseye.applications.ArtistryUtil;
import com.simiacryptus.mindseye.applications.ColorTransfer;
import com.simiacryptus.mindseye.applications.ImageArtUtil;
import com.simiacryptus.mindseye.applications.TextureGeneration;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.models.CVPipe_Inception;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

/**
 * The type Hi def.
 */
public abstract class TextureCovarianceSurvey extends ImageScript {

  public final double coeff_style_mean;
  /**
   * The Coeff style bandCovariance.
   */
  public final double[] coeff_style_covs;
  /**
   * The Style sources.
   */
  public final String[] styleSources;
  /**
   * The Resolution.
   */
  public final int[] resolutionSchedule;
  /**
   * The Dream coeff.
   */
  public final double dreamCoeff;
  private final int style_resolution;

  public TextureCovarianceSurvey(
      final double coeff_style_mean,
      final double[] coeff_style_covs,
      final double dreamCoeff,
      final int[] resolutionSchedule,
      final int style_resolution,
      final String... styleSources
  ) {
    this.coeff_style_mean = coeff_style_mean;
    this.coeff_style_covs = coeff_style_covs;
    this.dreamCoeff = dreamCoeff;
    this.resolutionSchedule = resolutionSchedule;
    this.style_resolution = style_resolution;
    this.styleSources = styleSources;
  }

  public void accept(@Nonnull NotebookOutput log) {

    Precision precision = Precision.Float;
    log.p("Style Source:");
    for (final CharSequence styleSource : styleSources) {
      log.p(log.png(ArtistryUtil.load(styleSource, style_resolution), "Style Image"));
    }

    final AtomicReference<Tensor> canvas = new AtomicReference<>(ArtistryUtil.paint_Plasma(3, 1000.0, 1.1, resolutionSchedule[0]));

    canvas.set(log.subreport("Color_Space_Analog", sublog -> {
      ColorTransfer<CVPipe_Inception.Strata, CVPipe_Inception> contentColorTransform = new ColorTransfer.Inception() {
      }.setOrtho(false).setUnit(true);
      //colorSyncContentCoeffMap.set(CVPipe_Inception.Strata.Layer_1a, 1e-1);
      int colorSyncResolution = 600;
      Tensor resizedCanvas = Tensor.fromRGB(TestUtil.resize(canvas.get().toImage(), colorSyncResolution));
      final ColorTransfer.StyleSetup<CVPipe_Inception.Strata> styleSetup = ImageArtUtil.getColorAnalogSetup(
          Arrays.asList(styleSources),
          precision,
          resizedCanvas,
          ImageArtUtil.getStyleImages(
              colorSyncResolution, styleSources
          ),
          CVPipe_Inception.Strata.Layer_1
      );
      contentColorTransform.transfer(
          sublog,
          resizedCanvas,
          styleSetup,
          getTrainingMinutes(),
          contentColorTransform.measureStyle(styleSetup),
          getMaxIterations(),
          isVerbose()
      );
      return contentColorTransform.forwardTransform(canvas.get());
    }));


    for (final double coeff_style_cov : coeff_style_covs) {
      final List<CVPipe_Inception.Strata> layers = getLayers();
      String reportName = String.format("Cov_%s", coeff_style_cov);
      log.h1(reportName);
      Tensor subresult = log.subreport(reportName, subreport -> {
        final Map<List<CharSequence>, TextureGeneration.StyleCoefficients<CVPipe_Inception.Strata>> styles = TestUtil.buildMap(x -> {
          TextureGeneration.StyleCoefficients<CVPipe_Inception.Strata> styleCoefficients = new TextureGeneration.StyleCoefficients<>(
              TextureGeneration.CenteringMode.Origin);
          for (final CVPipe_Inception.Strata layer : layers) {
            styleCoefficients.set(
                layer,
                coeff_style_mean,
                coeff_style_cov,
                dreamCoeff
            );
          }
          x.put(
              Arrays.asList(styleSources),
              styleCoefficients
          );
        });
        TextureGeneration.StyleSetup<CVPipe_Inception.Strata> styleSetup = new TextureGeneration.StyleSetup<>(
            precision,
            TestUtil.buildMap(y -> y.putAll(
                styles.keySet().stream().flatMap(
                    x -> x.stream())
                    .collect(Collectors.toMap(
                        x -> x,
                        file -> ArtistryUtil.load(
                            file,
                            style_resolution
                        )
                    )))),
            styles
        );
        AtomicReference<Tensor> canvasCopy = new AtomicReference<>(canvas.get().copy());
        for (final Integer resolution : resolutionSchedule) {
          TextureGeneration.Inception textureGeneration = new TextureGeneration.Inception();
          textureGeneration.parallelLossFunctions = true;
          textureGeneration.setTiling((int) Math.max(Math.min((2.0 * Math.pow(600, 2)) / (resolution * resolution), 9), 2));
          canvasCopy.set(Tensor.fromRGB(TestUtil.resize(canvasCopy.get().toImage(), resolution)));
          subreport.p("Input Parameters:");
          subreport.eval(() -> {
            return ArtistryUtil.toJson(styleSetup);
          });
          Precision precision1 = styleSetup.precision;
          canvasCopy.set(textureGeneration.optimize(
              subreport,
              textureGeneration.measureStyle(styleSetup), canvasCopy.get(),
              getTrainingMinutes(),
              getMaxIterations(), isVerbose(), precision1
          ));
        }
        return canvasCopy.get();
      });
      log.p(log.png(subresult.toImage(), reportName));
    }

  }

  @Nonnull
  public abstract List<CVPipe_Inception.Strata> getLayers();

}
