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
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

public abstract class HiDef extends ImageScript {

  public final double coeff_style_mean;
  public final double coeff_style_cov;
  public final String[] styleSources;
  public final int resolution;
  public final double dreamCoeff;

  public HiDef(
      final double coeff_style_mean,
      final double coeff_style_cov,
      final double dreamCoeff,
      final int resolution,
      final String... styleSources
  ) {
    this.coeff_style_mean = coeff_style_mean;
    this.coeff_style_cov = coeff_style_cov;
    this.dreamCoeff = dreamCoeff;
    this.resolution = resolution;
    this.styleSources = styleSources;
  }

  public void accept(@Nonnull NotebookOutput log) {

    TextureGeneration.Inception textureGeneration = new TextureGeneration.Inception();
    Precision precision = Precision.Float;
    textureGeneration.parallelLossFunctions = true;
    textureGeneration.setTiling(3);
    log.p("Style Source:");
    for (final CharSequence styleSource : styleSources) {
      log.p(log.png(ArtistryUtil.load(styleSource, resolution), "Style Image"));
    }
    final Map<List<CharSequence>, TextureGeneration.StyleCoefficients<CVPipe_Inception.Strata>> styles = TestUtil.buildMap(x -> {
      TextureGeneration.StyleCoefficients<CVPipe_Inception.Strata> styleCoefficients = new TextureGeneration.StyleCoefficients<>(
          TextureGeneration.CenteringMode.Origin);
      for (final CVPipe_Inception.Strata layer : getLayers()) {
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

    final AtomicReference<Tensor> canvas = new AtomicReference<>(ArtistryUtil.paint_Plasma(3, 1000.0, 1.1, resolution));

    canvas.set(log.subreport(sublog -> {
      ColorTransfer<CVPipe_Inception.Strata, CVPipe_Inception> contentColorTransform = new ColorTransfer.Inception() {
      }.setOrtho(false);
      //colorSyncContentCoeffMap.set(CVPipe_Inception.Strata.Layer_1a, 1e-1);
      int colorSyncResolution = 600;
      Tensor resizedCanvas = Tensor.fromRGB(ImageUtil.resize(canvas.get().toImage(), colorSyncResolution));
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
    }, log.getName() + "_" + "Color_Space_Analog"));

    TextureGeneration.StyleSetup<CVPipe_Inception.Strata> styleSetup = new TextureGeneration.StyleSetup<>(
        precision,
        TestUtil.buildMap(y -> y.putAll(
            styles.keySet().stream().flatMap(
                x -> x.stream())
                .collect(Collectors.toMap(
                    x -> x,
                    file -> ArtistryUtil.load(
                        file,
                        resolution
                    )
                )))),
        styles
    );
    log.p("Input Parameters:");
    log.eval(() -> {
      return ArtistryUtil.toJson(styleSetup);
    });
    textureGeneration.optimize(
        log,
        textureGeneration.measureStyle(styleSetup), canvas.get(),
        getTrainingMinutes(),
        getMaxIterations(), isVerbose(), styleSetup.precision
    );
  }

  @Nonnull
  public abstract List<CVPipe_Inception.Strata> getLayers();

}
