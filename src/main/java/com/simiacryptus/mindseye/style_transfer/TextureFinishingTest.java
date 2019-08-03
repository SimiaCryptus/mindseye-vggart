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
import com.simiacryptus.mindseye.applications.ImageArtUtil;
import com.simiacryptus.mindseye.applications.SegmentedStyleTransfer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.models.CVPipe_Inception;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.notebook.MarkdownNotebookOutput;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.util.JsonUtil;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public class TextureFinishingTest extends ImageScript {

  private final int steps;
  public int resolution;
  public CharSequence[] styleSources;
  public String[] contentSources;

  public TextureFinishingTest(final String[] contentSources, final CharSequence[] styleSources) {
    this.contentSources = contentSources;
    this.styleSources = styleSources;
    this.verbose = true;
    this.maxIterations = 30;
    this.trainingMinutes = 30;
    resolution = 3400;
    steps = 2;
  }

  public void accept(@Nonnull NotebookOutput log) {
    TestUtil.addGlobalHandlers(log.getHttpd());
    ((MarkdownNotebookOutput) log).setMaxImageSize(10000);
    log.eval(() -> {
      return JsonUtil.toJson(TextureFinishingTest.this);
    });

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

        TestUtil.geometricStream(canvasImage.get().getDimensions()[0], resolution, steps + 1).get().skip(1).forEach(res -> {
          canvasImage.set(Tensor.fromRGB(TestUtil.resize(canvasImage.get().toImage(), (int) res, true)));
          canvasImage.set(log.subreport(sublog -> {
            SegmentedStyleTransfer.ContentCoefficients<CVPipe_Inception.Strata> contentCoefficients = new SegmentedStyleTransfer.ContentCoefficients<>();
            contentCoefficients.set(CVPipe_Inception.Strata.Layer_0, 1e-1);
            int padding = 20;
            final ImageArtUtil.ImageArtOpParams imageArtOpParams = new ImageArtUtil.ImageArtOpParams(
                sublog,
                getTrainingMinutes(),
                getMaxIterations(),
                isVerbose()
            );
            final int torroidalOffsetX = true ? -padding : 0;
            final int torroidalOffsetY = true ? -padding : 0;
            final ImageArtUtil.TileLayout tileLayout = new ImageArtUtil.TileLayout(
                600,
                canvasImage.get(),
                padding,
                torroidalOffsetX,
                torroidalOffsetY
            );
            final Tensor content = ArtistryUtil.loadTensor(contentSource, tileLayout.getCanvasDimensions()[0], tileLayout.getCanvasDimensions()[1]);
            if (tileLayout.getCols() > 1 || tileLayout.getRows() > 1) {
              return ImageArtUtil.tiledTransfer(imageArtOpParams,
                  canvasImage.get(),
                  padding, torroidalOffsetX, torroidalOffsetY, tileLayout, (contentTile, canvasTile, i) -> {
                    imageArtOpParams.getLog().p(String.format("Processing Tile %s with size %s", i, Arrays.toString(contentTile.getDimensions())));
                    imageArtOpParams.getLog().p(imageArtOpParams.getLog().png(contentTile.toImage(), ""));
                    imageArtOpParams.getLog().p(imageArtOpParams.getLog().png(canvasTile.toImage(), ""));
                    return canvasTile;
                  }, content
              );
            } else {
              return content;
            }
          }, log.getName() + "_" + String.format("Phase_%s", index.getAndIncrement())));
          log.eval(() -> {
            return canvasImage.get().toImage();
          });
        });


      } catch (Throwable throwable) {
        log.eval(() -> {
          return throwable;
        });
      }
    }
  }


}
