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
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer;
import com.simiacryptus.mindseye.models.CVPipe_Inception;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.orient.LBFGS;
import com.simiacryptus.mindseye.opt.orient.OrientationStrategy;
import com.simiacryptus.mindseye.opt.orient.TrustRegionStrategy;
import com.simiacryptus.mindseye.opt.region.OrthonormalConstraint;
import com.simiacryptus.mindseye.opt.region.TrustRegion;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.util.JsonUtil;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * The type Style survey.
 */
public class ColorSyncSurvey extends ImageScript {

  public final int minStyleWidth;
  public int maxResolution;
  public int startResolution;
  public int plasmaResolution;
  public CharSequence[] styleSources;
  public String[] contentSources;

  public ColorSyncSurvey(final String[] contentSources, final CharSequence[] styleSources) {
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
      return JsonUtil.toJson(ColorSyncSurvey.this);
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
        Tensor canvasBufferedImage = Tensor.fromRGB(TestUtil.resize(
            ArtistryUtil.load(contentSource),
            resolution.get(),
            true
        ));

        log.p(log.png(log.subreport("Color_Space_Analog_1", sublog -> {
          canvasBufferedImage.assertAlive();
          return ImageArtUtil.colorTransfer(
              new ImageArtUtil.ImageArtOpParams(sublog, getMaxIterations(), getTrainingMinutes(), isVerbose()),
              new ColorTransfer.Inception() {
                @Nonnull
                @Override
                public OrientationStrategy<LineSearchCursor> getOrientation() {
                  return new TrustRegionStrategy(new LBFGS()) {
                    @Override
                    public TrustRegion getRegionPolicy(final Layer layer) {
                      if (layer instanceof SimpleConvolutionLayer) {
                        return new OrthonormalConstraint(ColorTransfer.getIndexMap((SimpleConvolutionLayer) layer)).setOrtho(false);
                      } else {
                        return null;
                      }
                    }
                  };
                }
              },
              new ColorTransfer.StyleSetup<CVPipe_Inception.Strata>(
                  precision,
                  canvasBufferedImage.copy(),
                  new ColorTransfer.ContentCoefficients<>(),
                  ImageArtUtil.getStyleImages(
                      Math.max(resolution.get(), minStyleWidth), styleSources
                  ),
                  TestUtil.buildMap(map -> {
                    ColorTransfer.StyleCoefficients<CVPipe_Inception.Strata> styleCoefficients =
                        new ColorTransfer.StyleCoefficients<>(ColorTransfer.CenteringMode.Origin);
                    styleCoefficients.set(
                        CVPipe_Inception.Strata.Layer_0,
                        1e0,
                        1e0,
                        (double) 0
                    );
                    map.put(Arrays.asList(styleSources), styleCoefficients);
                  })
              ),
              contentSource,
              startResolution,
              canvasBufferedImage.copy()
          );
        }).toImage(), "Style-Aligned Content Color"));

        log.p(log.png(log.subreport("Color_Space_Analog_2", sublog -> {
          canvasBufferedImage.assertAlive();
          return ImageArtUtil.colorTransfer(
              new ImageArtUtil.ImageArtOpParams(sublog, getMaxIterations(), getTrainingMinutes(), isVerbose()),
              new ColorTransfer.Inception() {
                @Nonnull
                @Override
                public OrientationStrategy<LineSearchCursor> getOrientation() {
                  return new TrustRegionStrategy(new LBFGS()) {
                    @Override
                    public TrustRegion getRegionPolicy(final Layer layer) {
                      if (layer instanceof SimpleConvolutionLayer) {
                        return new OrthonormalConstraint(ColorTransfer.getIndexMap((SimpleConvolutionLayer) layer)).setUnit(false);
                      } else {
                        return null;
                      }
                    }
                  };
                }
              },
              new ColorTransfer.StyleSetup<CVPipe_Inception.Strata>(
                  precision,
                  canvasBufferedImage.copy(),
                  new ColorTransfer.ContentCoefficients<>(),
                  ImageArtUtil.getStyleImages(
                      Math.max(resolution.get(), minStyleWidth), styleSources
                  ),
                  TestUtil.buildMap(map -> {
                    ColorTransfer.StyleCoefficients<CVPipe_Inception.Strata> styleCoefficients =
                        new ColorTransfer.StyleCoefficients<>(ColorTransfer.CenteringMode.Origin);
                    styleCoefficients.set(
                        CVPipe_Inception.Strata.Layer_0,
                        1e0,
                        1e0,
                        (double) 0
                    );
                    map.put(Arrays.asList(styleSources), styleCoefficients);
                  })
              ),
              contentSource,
              startResolution,
              canvasBufferedImage.copy()
          );
        }).toImage(), "Style-Aligned Content Color"));

        log.p(log.png(log.subreport("Color_Space_Analog_3", sublog -> {
          canvasBufferedImage.assertAlive();
          return ImageArtUtil.colorTransfer(
              new ImageArtUtil.ImageArtOpParams(sublog, getMaxIterations(), getTrainingMinutes(), isVerbose()),
              new ColorTransfer.Inception() {
                @Nonnull
                @Override
                public OrientationStrategy<LineSearchCursor> getOrientation() {
                  return new TrustRegionStrategy(new LBFGS()) {
                    @Override
                    public TrustRegion getRegionPolicy(final Layer layer) {
                      if (layer instanceof SimpleConvolutionLayer) {
                        return new OrthonormalConstraint(ColorTransfer.getIndexMap((SimpleConvolutionLayer) layer));
                      } else {
                        return null;
                      }
                    }
                  };
                }
              },
              new ColorTransfer.StyleSetup<CVPipe_Inception.Strata>(
                  precision,
                  canvasBufferedImage.copy(),
                  new ColorTransfer.ContentCoefficients<>(),
                  ImageArtUtil.getStyleImages(
                      Math.max(resolution.get(), minStyleWidth), styleSources
                  ),
                  TestUtil.buildMap(map -> {
                    ColorTransfer.StyleCoefficients<CVPipe_Inception.Strata> styleCoefficients =
                        new ColorTransfer.StyleCoefficients<>(ColorTransfer.CenteringMode.Origin);
                    styleCoefficients.set(
                        CVPipe_Inception.Strata.Layer_0,
                        1e0,
                        0,
                        (double) 0
                    );
                    map.put(Arrays.asList(styleSources), styleCoefficients);
                  })
              ),
              contentSource,
              startResolution,
              canvasBufferedImage.copy()
          );
        }).toImage(), "Style-Aligned Content Color"));

        log.p(log.png(log.subreport("Color_Space_Analog_4", sublog -> {
          canvasBufferedImage.assertAlive();
          return ImageArtUtil.colorTransfer(
              new ImageArtUtil.ImageArtOpParams(sublog, getMaxIterations(), getTrainingMinutes(), isVerbose()),
              new ColorTransfer.Inception() {
                @Nonnull
                @Override
                public OrientationStrategy<LineSearchCursor> getOrientation() {
                  return new TrustRegionStrategy(new LBFGS()) {
                    @Override
                    public TrustRegion getRegionPolicy(final Layer layer) {
                      if (layer instanceof SimpleConvolutionLayer) {
                        return new OrthonormalConstraint(ColorTransfer.getIndexMap((SimpleConvolutionLayer) layer)).setOrtho(false);
                      } else {
                        return null;
                      }
                    }
                  };
                }
              },
              new ColorTransfer.StyleSetup<CVPipe_Inception.Strata>(
                  precision,
                  canvasBufferedImage.copy(),
                  new ColorTransfer.ContentCoefficients<>(),
                  ImageArtUtil.getStyleImages(
                      Math.max(resolution.get(), minStyleWidth), styleSources
                  ),
                  TestUtil.buildMap(map -> {
                    ColorTransfer.StyleCoefficients<CVPipe_Inception.Strata> styleCoefficients =
                        new ColorTransfer.StyleCoefficients<>(ColorTransfer.CenteringMode.Origin);
                    styleCoefficients.set(
                        CVPipe_Inception.Strata.Layer_1a,
                        1e0,
                        1e0,
                        (double) 0
                    );
                    map.put(Arrays.asList(styleSources), styleCoefficients);
                  })
              ),
              contentSource,
              startResolution,
              canvasBufferedImage.copy()
          );
        }).toImage(), "Style-Aligned Content Color"));

        log.p(log.png(log.subreport("Color_Space_Analog_5", sublog -> {
          canvasBufferedImage.assertAlive();
          return ImageArtUtil.colorTransfer(
              new ImageArtUtil.ImageArtOpParams(sublog, getMaxIterations(), getTrainingMinutes(), isVerbose()),
              new ColorTransfer.Inception() {
                @Nonnull
                @Override
                public OrientationStrategy<LineSearchCursor> getOrientation() {
                  return new TrustRegionStrategy(new LBFGS()) {
                    @Override
                    public TrustRegion getRegionPolicy(final Layer layer) {
                      if (layer instanceof SimpleConvolutionLayer) {
                        return new OrthonormalConstraint(ColorTransfer.getIndexMap((SimpleConvolutionLayer) layer)).setOrtho(false);
                      } else {
                        return null;
                      }
                    }
                  };
                }
              },
              new ColorTransfer.StyleSetup<CVPipe_Inception.Strata>(
                  precision,
                  canvasBufferedImage.copy(),
                  new ColorTransfer.ContentCoefficients<>(),
                  ImageArtUtil.getStyleImages(
                      Math.max(resolution.get(), minStyleWidth), styleSources
                  ),
                  TestUtil.buildMap(map -> {
                    ColorTransfer.StyleCoefficients<CVPipe_Inception.Strata> styleCoefficients =
                        new ColorTransfer.StyleCoefficients<>(ColorTransfer.CenteringMode.Dynamic);
                    styleCoefficients.set(
                        CVPipe_Inception.Strata.Layer_0,
                        1e0,
                        1e0,
                        (double) 0
                    );
                    map.put(Arrays.asList(styleSources), styleCoefficients);
                  })
              ),
              contentSource,
              startResolution,
              canvasBufferedImage.copy()
          );
        }).toImage(), "Style-Aligned Content Color"));

      } catch (Throwable throwable) {
        log.eval(() -> {
          return throwable;
        });
      }
    });
  }

}
