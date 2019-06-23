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

import com.simiacryptus.mindseye.ImageScript;
import com.simiacryptus.mindseye.applications.ArtistryUtil;
import com.simiacryptus.mindseye.applications.ColorTransfer;
import com.simiacryptus.mindseye.applications.ImageArtUtil;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.models.CVPipe_Inception;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.util.JsonUtil;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

public class ColorEnhancementSurvey extends ImageScript {

  public final int minStyleWidth;
  public int maxResolution;
  public int startResolution;
  public int plasmaResolution;
  public CharSequence[] styleSources;
  public String[] contentSources;

  public ColorEnhancementSurvey(final String[] contentSources, final CharSequence[] styleSources) {
    this.contentSources = contentSources;
    this.styleSources = styleSources;
    this.verbose = true;
    this.maxIterations = 30;
    this.trainingMinutes = 30;
    this.maxResolution = 1400;
    startResolution = 600;
    this.minStyleWidth = 600;
    plasmaResolution = startResolution / 8;
  }


  public void accept(@Nonnull NotebookOutput log) {
    TestUtil.addGlobalHandlers(log.getHttpd());

    log.eval(() -> {
      return JsonUtil.toJson(ColorEnhancementSurvey.this);
    });

    Precision precision = Precision.Float;
    Arrays.stream(contentSources).forEach(contentSource -> {
      try {

        log.h1("Task Initialization");
        final AtomicInteger resolution = new AtomicInteger(startResolution);
        log.p("Content Source:");
        log.p(log.png(ArtistryUtil.load(contentSource, resolution.get()), "Content Image"));
        log.p("Style Source:");
        for (final CharSequence styleSource : styleSources) {
          log.p(log.png(ArtistryUtil.load(styleSource, resolution.get()), "Style Image"));
        }


        log.subreport("Color_Space_Enhancement_1", sublog -> {
          ColorTransfer.StyleCoefficients<CVPipe_Inception.Strata> coefficients = new ColorTransfer.StyleCoefficients<>(ColorTransfer.CenteringMode.Origin);
          coefficients.set(CVPipe_Inception.Strata.Layer_2, 1e0, 1e0, 1e-1);
          return ImageArtUtil.getColorStyleEnhance(
              new ImageArtUtil.ImageArtOpParams(sublog, getMaxIterations(), getTrainingMinutes(), isVerbose()),
              precision,
              resolution,
              minStyleWidth,
              coefficients,
              styleSources
          );
        });

        log.subreport("Color_Space_Enhancement_2", sublog -> {
          ColorTransfer.StyleCoefficients<CVPipe_Inception.Strata> coefficients = new ColorTransfer.StyleCoefficients<>(ColorTransfer.CenteringMode.Origin);
          coefficients.set(CVPipe_Inception.Strata.Layer_2, 1e0, 0, 1e-1);
          return ImageArtUtil.getColorStyleEnhance(
              new ImageArtUtil.ImageArtOpParams(sublog, getMaxIterations(), getTrainingMinutes(), isVerbose()),
              precision,
              resolution,
              minStyleWidth,
              coefficients,
              styleSources
          );
        });

        log.subreport("Color_Space_Enhancement_3", sublog -> {
          ColorTransfer.StyleCoefficients<CVPipe_Inception.Strata> coefficients = new ColorTransfer.StyleCoefficients<>(ColorTransfer.CenteringMode.Origin);
          coefficients.set(CVPipe_Inception.Strata.Layer_2, 1e0, 0, -1e-1);
          return ImageArtUtil.getColorStyleEnhance(
              new ImageArtUtil.ImageArtOpParams(sublog, getMaxIterations(), getTrainingMinutes(), isVerbose()),
              precision,
              resolution,
              minStyleWidth,
              coefficients,
              styleSources
          );
        });

        log.subreport("Color_Space_Enhancement_4", sublog -> {
          ColorTransfer.StyleCoefficients<CVPipe_Inception.Strata> coefficients = new ColorTransfer.StyleCoefficients<>(ColorTransfer.CenteringMode.Origin);
          coefficients.set(CVPipe_Inception.Strata.Layer_2, 1e0, 1e0, 1e1);
          return ImageArtUtil.getColorStyleEnhance(
              new ImageArtUtil.ImageArtOpParams(sublog, getMaxIterations(), getTrainingMinutes(), isVerbose()),
              precision,
              resolution,
              minStyleWidth,
              coefficients,
              styleSources
          );
        });

        log.subreport("Color_Space_Enhancement_5", sublog -> {
          ColorTransfer.StyleCoefficients<CVPipe_Inception.Strata> coefficients = new ColorTransfer.StyleCoefficients<>(ColorTransfer.CenteringMode.Origin);
          coefficients.set(CVPipe_Inception.Strata.Layer_2, 1e0, 0, 1e1);
          return ImageArtUtil.getColorStyleEnhance(
              new ImageArtUtil.ImageArtOpParams(sublog, getMaxIterations(), getTrainingMinutes(), isVerbose()),
              precision,
              resolution,
              minStyleWidth,
              coefficients,
              styleSources
          );
        });

        log.subreport("Color_Space_Enhancement_6", sublog -> {
          ColorTransfer.StyleCoefficients<CVPipe_Inception.Strata> coefficients = new ColorTransfer.StyleCoefficients<>(ColorTransfer.CenteringMode.Origin);
          coefficients.set(CVPipe_Inception.Strata.Layer_2, 1e0, 0, -1e1);
          return ImageArtUtil.getColorStyleEnhance(
              new ImageArtUtil.ImageArtOpParams(sublog, getMaxIterations(), getTrainingMinutes(), isVerbose()),
              precision,
              resolution,
              minStyleWidth,
              coefficients,
              styleSources
          );
        });

        log.subreport("Color_Space_Enhancement_7", sublog -> {
          ColorTransfer.StyleCoefficients<CVPipe_Inception.Strata> coefficients = new ColorTransfer.StyleCoefficients<>(ColorTransfer.CenteringMode.Origin);
          coefficients.set(CVPipe_Inception.Strata.Layer_2, 1e0, 1e0, -1e-1);
          return ImageArtUtil.getColorStyleEnhance(
              new ImageArtUtil.ImageArtOpParams(sublog, getMaxIterations(), getTrainingMinutes(), isVerbose()),
              precision,
              resolution,
              minStyleWidth,
              coefficients,
              styleSources
          );
        });

      } catch (Throwable throwable) {
        log.eval(() -> {
          return throwable;
        });
      }
    });
  }

}
