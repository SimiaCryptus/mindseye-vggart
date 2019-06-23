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
import com.simiacryptus.mindseye.pyramid.PyramidUtil;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.notebook.MarkdownNotebookOutput;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.util.JsonUtil;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public class TextureFinishing extends ImageScript {

  private final int steps;
  public int resolution;
  public CharSequence[] styleSources;
  public String[] contentSources;

  public TextureFinishing(final String[] contentSources, final CharSequence[] styleSources) {
    this.contentSources = contentSources;
    this.styleSources = styleSources;
    this.verbose = true;
    this.maxIterations = 20;
    this.trainingMinutes = 20;
    resolution = 3400;
    steps = 4;
  }

  public void accept(@Nonnull NotebookOutput log) {
    TestUtil.addGlobalHandlers(log.getHttpd());
    ((MarkdownNotebookOutput) log).setMaxImageSize(resolution);

    log.eval(() -> {
      return JsonUtil.toJson(TextureFinishing.this);
    });
    PyramidUtil.initJS(log);

    SegmentedStyleTransfer<CVPipe_Inception.Strata, CVPipe_Inception> styleTransfer = new SegmentedStyleTransfer.Inception();
    Precision precision = Precision.Float;
    int imageClusters = 1;
    styleTransfer.setStyle_masks(imageClusters);
    styleTransfer.setStyle_textureClusters(imageClusters);
    styleTransfer.setContent_colorClusters(imageClusters);
    styleTransfer.setContent_textureClusters(imageClusters);
    styleTransfer.setContent_masks(imageClusters);
    styleTransfer.parallelLossFunctions = true;
    styleTransfer.setTiled(false);
    for (String contentSource : contentSources) {
      try {
        log.h1("Task Initialization");
        log.p("Content Source:");
        log.p(log.png(ArtistryUtil.load(contentSource, -1), "Content Image"));
        log.p("Style Source:");
        for (final CharSequence styleSource : styleSources) {
          log.p(log.png(ArtistryUtil.load(styleSource, -1), "Style Image"));
        }

        // Enhance color scheme:
        final AtomicReference<Tensor> canvasImage = new AtomicReference<>(ArtistryUtil.loadTensor(contentSource, -1));
        final AtomicInteger index = new AtomicInteger();

        double[] resolutions = TestUtil.geometricStream(canvasImage.get().getDimensions()[0], resolution, steps + 1).get().skip(1).toArray();
        Arrays.stream(resolutions).forEach(res -> {
          canvasImage.set(Tensor.fromRGB(TestUtil.resize(canvasImage.get().toImage(), (int) res, true)));
          canvasImage.set(log.subreport(String.format("Phase_%s", index.incrementAndGet()), sublog -> {
            SegmentedStyleTransfer.ContentCoefficients<CVPipe_Inception.Strata> contentCoefficients = new SegmentedStyleTransfer.ContentCoefficients<>();
            contentCoefficients.set(CVPipe_Inception.Strata.Layer_1, 1e-1);
            Map<CVPipe_Inception.Strata, Double> styleLayers = new HashMap<>();
            styleLayers.put(CVPipe_Inception.Strata.Layer_2, 1e0);
            styleLayers.put(CVPipe_Inception.Strata.Layer_3a, 1e0);
            final Tensor canvasImage1 = canvasImage.get();
            int padding = 20;
            final int torroidalOffsetX = true ? -padding : 0;
            final int torroidalOffsetY = true ? -padding : 0;
            final ImageArtUtil.ImageArtOpParams imageArtOpParams = new ImageArtUtil.ImageArtOpParams(
                sublog,
                getTrainingMinutes(),
                getMaxIterations(),
                isVerbose()
            );
            final SegmentedStyleTransfer.StyleSetup<CVPipe_Inception.Strata> styleSetup = new SegmentedStyleTransfer.StyleSetup<>(
                precision,
                new ColorTransfer.Inception().forwardTransform(
                    ArtistryUtil.loadTensor(
                        contentSource,
                        canvasImage1.getDimensions()[0],
                        canvasImage1.getDimensions()[1]
                    )),
                ImageArtUtil.scale(
                    contentCoefficients,
                    1e0
                ),
                ImageArtUtil.getStyleImages(
                    (int) Math.max(res, 1200), styleSources
                ),
                TestUtil.buildMap(x -> {
                  x.put(
                      Arrays.asList(styleSources),
                      ImageArtUtil.getStyleCoefficients(
                          styleLayers,
                          1e0,
                          1e0,
                          5e-1
                      )
                  );
                })
            );
            final ImageArtUtil.TileLayout tileLayout = new ImageArtUtil.TileLayout(600, canvasImage1, padding, torroidalOffsetX, torroidalOffsetY);
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
          BufferedImage image = log.eval(() -> {
            return canvasImage.get().toImage();
          });
          PyramidUtil.initImagePyramids(log, String.format("Phase_%s", index.get()), 512, image);
        });


      } catch (Throwable throwable) {
        log.eval(() -> {
          return throwable;
        });
      }
    }
  }


}
