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
import com.simiacryptus.mindseye.applications.SegmentedStyleTransfer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.models.CVPipe_Inception;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.util.JsonUtil;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

/**
 * The type Style survey.
 */
public class FullyFeaturedArtJob extends ImageScript {

  public final int minStyleWidth;
  public int maxResolution;
  public int startResolution;
  public int plasmaResolution;
  public CharSequence[] styleSources;
  public String[] contentSources;

  public FullyFeaturedArtJob(final String[] contentSources, final CharSequence[] styleSources) {
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
      return JsonUtil.toJson(FullyFeaturedArtJob.this);
    });

    SegmentedStyleTransfer<CVPipe_Inception.Strata, CVPipe_Inception> styleTransfer = new SegmentedStyleTransfer.Inception();
    Precision precision = Precision.Float;
    int image_clusters = 3;
    styleTransfer.setStlye_colorClusters(image_clusters);
    styleTransfer.setStyle_textureClusters(image_clusters);
    styleTransfer.setStyle_masks(image_clusters);
    styleTransfer.setContent_colorClusters(image_clusters);
    styleTransfer.setContent_textureClusters(image_clusters);
    styleTransfer.setContent_masks(image_clusters);
    styleTransfer.parallelLossFunctions = true;
    styleTransfer.setTiled(false);
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

        // Enhance color scheme:
        Map<CharSequence, ColorTransfer<CVPipe_Inception.Strata, CVPipe_Inception>> styleColorTransforms = log.subreport("Color_Space_Enhancement", sublog -> {
          ColorTransfer.StyleCoefficients<CVPipe_Inception.Strata> coefficients = new ColorTransfer.StyleCoefficients<>(ColorTransfer.CenteringMode.Origin);
          coefficients.set(CVPipe_Inception.Strata.Layer_2, 1e0, 1e0, -1e-1);
//          coefficients.set(CVPipe_Inception.Strata.Layer_1b, 1e0, 1e-1, -1e0);
//          coefficients.set(CVPipe_Inception.Strata.Layer_1d, (double) 1e-1, (double) 1e-1, 1e0);
//          coefficients.set(CVPipe_Inception.Strata.Layer_1e, (double) 1e-1, (double) 1e-1, 1e1);
          return ImageArtUtil.getColorStyleEnhance(
              new ImageArtUtil.ImageArtOpParams(sublog, getMaxIterations(), getTrainingMinutes(), isVerbose()),
              precision,
              resolution,
              minStyleWidth,
              coefficients,
              styleSources
          );
        });

        Tensor canvasBufferedImage = Tensor.fromRGB(TestUtil.resize(
            ArtistryUtil.load(contentSource),
            resolution.get(),
            true
        ));
        canvasBufferedImage.assertAlive();
        final AtomicReference<Tensor> canvasImage = new AtomicReference<>(canvasBufferedImage);
        ColorTransfer<CVPipe_Inception.Strata, CVPipe_Inception> contentColorTransform = new ColorTransfer.Inception().setOrtho(false);
        // Transfer color scheme:
        Tensor color_space_analog = log.subreport("Color_Space_Analog", sublog -> {
          //colorSyncContentCoeffMap.set(CVPipe_Inception.Strata.Layer_1a, 1e-1);
          return ImageArtUtil.colorTransfer(
              new ImageArtUtil.ImageArtOpParams(sublog, getMaxIterations(), getTrainingMinutes(), isVerbose()),
              contentColorTransform,
              ImageArtUtil.getColorAnalogSetup(
                  Arrays.asList(styleSources),
                  precision,
                  canvasBufferedImage,
                  ImageArtUtil.getStyleImages(
                      styleColorTransforms, Math.max(resolution.get(), minStyleWidth), styleSources
                  ),
                  CVPipe_Inception.Strata.Layer_2
              ),
              contentSource,
              startResolution,
              canvasImage.get().copy()
          );
        });
        log.p(log.png(color_space_analog.toImage(), "Style-Aligned Content Color"));
        canvasImage.set(color_space_analog);

        // Seed initialization / degradation
        canvasImage.set(ImageArtUtil.degrade(
            plasmaResolution, resolution.get(), canvasImage.get()
        ));

        Map<CVPipe_Inception.Strata, Double> styleLayers = TestUtil.buildMap(m -> {
          m.put(CVPipe_Inception.Strata.Layer_2, 1e-1);
          m.put(CVPipe_Inception.Strata.Layer_1b, 1e0);
          m.put(CVPipe_Inception.Strata.Layer_1c, 1e0);
          m.put(CVPipe_Inception.Strata.Layer_1d, 1e1);
        });
        SegmentedStyleTransfer.ContentCoefficients<CVPipe_Inception.Strata> contentCoefficients =
            new SegmentedStyleTransfer.ContentCoefficients<CVPipe_Inception.Strata>()
                .set(CVPipe_Inception.Strata.Layer_2, 1e-3)
                .set(CVPipe_Inception.Strata.Layer_1b, 1e-2)
                .set(CVPipe_Inception.Strata.Layer_1c, 1e1);

        {
          log.h1("Phase 0 - Dreaming");
          double coeff_style_mean = 1e1;
          double contentMixingCoeff = 1e0;
          double dreamCoeff = 1e1;
          double coeff_style_cov = 1e1;
          canvasImage.set(log.subreport("Phase_0", sublog -> {
            final Tensor canvasImage1 = canvasImage.get();
            int padding = 20;
            final int torroidalOffsetX = false ? -padding : 0;
            final int torroidalOffsetY = false ? -padding : 0;
            final ImageArtUtil.ImageArtOpParams imageArtOpParams = new ImageArtUtil.ImageArtOpParams(
                sublog,
                getTrainingMinutes(),
                getMaxIterations(),
                isVerbose()
            );
            final SegmentedStyleTransfer.StyleSetup<CVPipe_Inception.Strata> styleSetup = new SegmentedStyleTransfer.StyleSetup<>(
                precision,
                contentColorTransform.forwardTransform(
                    ArtistryUtil.loadTensor(
                        contentSource,
                        canvasImage1.getDimensions()[0],
                        canvasImage1.getDimensions()[1]
                    )),
                ImageArtUtil.scale(
                    contentCoefficients,
                    contentMixingCoeff
                ),
                ImageArtUtil.getStyleImages(
                    styleColorTransforms, Math.max(resolution.get(), minStyleWidth), styleSources
                ),
                TestUtil.buildMap(x -> {
                  x.put(
                      Arrays.asList(styleSources),
                      ImageArtUtil.getStyleCoefficients(
                          styleLayers,
                          coeff_style_mean,
                          coeff_style_cov,
                          dreamCoeff
                      )
                  );
                })
            );
            final ImageArtUtil.TileLayout tileLayout = new ImageArtUtil.TileLayout(
                startResolution,
                canvasImage1,
                padding,
                torroidalOffsetX,
                torroidalOffsetY
            );
            ImageArtUtil.TileTransformer transformer = new ImageArtUtil.StyleTransformer(
                imageArtOpParams,
                styleTransfer,
                tileLayout,
                padding,
                torroidalOffsetX,
                torroidalOffsetY,
                styleSetup
            );

            final HashMap<SegmentedStyleTransfer.MaskJob, Set<Tensor>> originalCache = new HashMap<>(styleTransfer.getMaskCache());
            final Tensor result;
            final Tensor content = ArtistryUtil.loadTensor(contentSource, tileLayout.getCanvasDimensions()[0], tileLayout.getCanvasDimensions()[1]);
            if (tileLayout.getCols() > 1 || tileLayout.getRows() > 1) {
              result = ImageArtUtil.tiledTransfer(imageArtOpParams,
                  canvasImage1,
                  padding,
                  torroidalOffsetX,
                  torroidalOffsetY, tileLayout, transformer, content
              );
            } else {
              result = styleTransfer.transfer(
                  imageArtOpParams.getLog(),
                  styleSetup,
                  imageArtOpParams.getMaxIterations(),
                  styleTransfer.measureStyle(imageArtOpParams.getLog(), styleSetup),
                  imageArtOpParams.getTrainingMinutes(),
                  imageArtOpParams.isVerbose(),
                  canvasImage1
              );
            }
            styleTransfer.getMaskCache().clear();
            styleTransfer.getMaskCache().putAll(originalCache);
            return result;
          }));
          log.eval(() -> {
            return contentColorTransform.inverseTransform(canvasImage.get()).toImage();
          });
        }

        {
          log.h1("Phase 1 - Waking Up");
          double contentMixingCoeff = 1e2;
          double dreamCoeff = 1e0;
          double coeff_style_mean = 1e1;
          double coeff_style_cov = 1e1;
          canvasImage.set(log.subreport("Phase_1", log2 -> {
            final Tensor canvasImage1 = canvasImage.get();
            int padding = 20;
            final int torroidalOffsetX = false ? -padding : 0;
            final int torroidalOffsetY = false ? -padding : 0;
            final ImageArtUtil.ImageArtOpParams imageArtOpParams = new ImageArtUtil.ImageArtOpParams(
                log2,
                getTrainingMinutes(),
                getMaxIterations(),
                isVerbose()
            );
            final SegmentedStyleTransfer.StyleSetup<CVPipe_Inception.Strata> styleSetup = new SegmentedStyleTransfer.StyleSetup<>(
                precision,
                contentColorTransform.forwardTransform(
                    ArtistryUtil.loadTensor(
                        contentSource,
                        canvasImage1.getDimensions()[0],
                        canvasImage1.getDimensions()[1]
                    )),
                ImageArtUtil.scale(
                    contentCoefficients,
                    contentMixingCoeff
                ),
                ImageArtUtil.getStyleImages(
                    styleColorTransforms, Math.max(resolution.get(), minStyleWidth), styleSources
                ),
                TestUtil.buildMap(x -> {
                  x.put(
                      Arrays.asList(styleSources),
                      ImageArtUtil.getStyleCoefficients(
                          styleLayers,
                          coeff_style_mean,
                          coeff_style_cov,
                          dreamCoeff
                      )
                  );
                })
            );
            final ImageArtUtil.TileLayout tileLayout = new ImageArtUtil.TileLayout(
                startResolution,
                canvasImage1,
                padding,
                torroidalOffsetX,
                torroidalOffsetY
            );
            ImageArtUtil.TileTransformer transformer = new ImageArtUtil.StyleTransformer(
                imageArtOpParams,
                styleTransfer,
                tileLayout,
                padding,
                torroidalOffsetX,
                torroidalOffsetY,
                styleSetup
            );

            final HashMap<SegmentedStyleTransfer.MaskJob, Set<Tensor>> originalCache = new HashMap<>(styleTransfer.getMaskCache());
            final Tensor result;
            final Tensor content = ArtistryUtil.loadTensor(contentSource, tileLayout.getCanvasDimensions()[0], tileLayout.getCanvasDimensions()[1]);
            if (tileLayout.getCols() > 1 || tileLayout.getRows() > 1) {
              result = ImageArtUtil.tiledTransfer(imageArtOpParams,
                  canvasImage1,
                  padding,
                  torroidalOffsetX,
                  torroidalOffsetY, tileLayout, transformer, content
              );
            } else {
              result = styleTransfer.transfer(
                  imageArtOpParams.getLog(),
                  styleSetup,
                  imageArtOpParams.getMaxIterations(),
                  styleTransfer.measureStyle(imageArtOpParams.getLog(), styleSetup),
                  imageArtOpParams.getTrainingMinutes(),
                  imageArtOpParams.isVerbose(),
                  canvasImage1
              );
            }
            styleTransfer.getMaskCache().clear();
            styleTransfer.getMaskCache().putAll(originalCache);
            return result;
          }));
          log.eval(() -> {
            return contentColorTransform.inverseTransform(canvasImage.get()).toImage();
          });
        }

        //contentCoefficients0.set(CVPipe_Inception.Strata.Layer_1a, 1e-1);

        while (resolution.updateAndGet(v -> (int) (v * Math.pow(3, 0.5))) < maxResolution) {
          log.h1("Phase n+1 - Enlarge to " + resolution.get());
          double coeff_style_mean;
          double contentMixingCoeff;
          double dreamCoeff;
          double coeff_style_cov;
          if (resolution.get() <= 1600) {
            dreamCoeff = 1e0;
            contentMixingCoeff = 1e0;
            coeff_style_mean = 1e1;
            coeff_style_cov = 1e1;
          } else {
            styleLayers.remove(CVPipe_Inception.Strata.Layer_2);
            styleLayers.remove(CVPipe_Inception.Strata.Layer_1b);
            styleLayers.put(CVPipe_Inception.Strata.Layer_1e, 1.0);
            dreamCoeff = 1e1;
            contentMixingCoeff = 1e0;
            coeff_style_mean = 1e1;
            coeff_style_cov = 1e2;
          }
          canvasImage.set(Tensor.fromRGB(TestUtil.resize(canvasImage.get().toImage(), resolution.get(), true)));
          canvasImage.set(log.subreport("Phase_" + resolution.get(), log2 -> {
            final Tensor canvasImage1 = canvasImage.get();
            int padding = 20;
            final int torroidalOffsetX = false ? -padding : 0;
            final int torroidalOffsetY = false ? -padding : 0;
            final ImageArtUtil.ImageArtOpParams imageArtOpParams = new ImageArtUtil.ImageArtOpParams(
                log2,
                getTrainingMinutes(),
                getMaxIterations(),
                isVerbose()
            );
            final SegmentedStyleTransfer.StyleSetup<CVPipe_Inception.Strata> styleSetup = new SegmentedStyleTransfer.StyleSetup<>(
                precision,
                contentColorTransform.forwardTransform(
                    ArtistryUtil.loadTensor(
                        contentSource,
                        canvasImage1.getDimensions()[0],
                        canvasImage1.getDimensions()[1]
                    )),
                ImageArtUtil.scale(
                    contentCoefficients,
                    contentMixingCoeff
                ),
                ImageArtUtil.getStyleImages(
                    styleColorTransforms, Math.max(resolution.get(), minStyleWidth), styleSources
                ),
                TestUtil.buildMap(x -> {
                  x.put(
                      Arrays.asList(styleSources),
                      ImageArtUtil.getStyleCoefficients(
                          styleLayers,
                          coeff_style_mean,
                          coeff_style_cov,
                          dreamCoeff
                      )
                  );
                })
            );
            final ImageArtUtil.TileLayout tileLayout = new ImageArtUtil.TileLayout(
                startResolution,
                canvasImage1,
                padding,
                torroidalOffsetX,
                torroidalOffsetY
            );
            ImageArtUtil.TileTransformer transformer = new ImageArtUtil.StyleTransformer(
                imageArtOpParams,
                styleTransfer,
                tileLayout,
                padding,
                torroidalOffsetX,
                torroidalOffsetY,
                styleSetup
            );

            final HashMap<SegmentedStyleTransfer.MaskJob, Set<Tensor>> originalCache = new HashMap<>(styleTransfer.getMaskCache());
            final Tensor result;
            final Tensor content = ArtistryUtil.loadTensor(contentSource, tileLayout.getCanvasDimensions()[0], tileLayout.getCanvasDimensions()[1]);
            if (tileLayout.getCols() > 1 || tileLayout.getRows() > 1) {
              result = ImageArtUtil.tiledTransfer(imageArtOpParams,
                  canvasImage1,
                  padding,
                  torroidalOffsetX,
                  torroidalOffsetY, tileLayout, transformer, content
              );
            } else {
              result = styleTransfer.transfer(
                  imageArtOpParams.getLog(),
                  styleSetup,
                  imageArtOpParams.getMaxIterations(),
                  styleTransfer.measureStyle(imageArtOpParams.getLog(), styleSetup),
                  imageArtOpParams.getTrainingMinutes(),
                  imageArtOpParams.isVerbose(),
                  canvasImage1
              );
            }
            styleTransfer.getMaskCache().clear();
            styleTransfer.getMaskCache().putAll(originalCache);
            return result;
          }));
          log.eval(() -> {
            return contentColorTransform.inverseTransform(canvasImage.get()).toImage();
          });
        }
      } catch (Throwable throwable) {
        log.eval(() -> {
          return throwable;
        });
      }
    });
  }

}
