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
import com.simiacryptus.mindseye.applications.ColorTransfer;
import com.simiacryptus.mindseye.applications.ImageArtUtil;
import com.simiacryptus.mindseye.applications.SegmentedStyleTransfer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.models.CVPipe_Inception;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.util.JsonUtil;

import javax.annotation.Nonnull;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

public class LayerStyleSurvey extends ImageScript {

  public final int minStyleWidth;
  public int maxResolution;
  public int startResolution;
  public int plasmaResolution;
  public CharSequence[] styleSources;
  public String[] contentSources;

  public LayerStyleSurvey() {
    this(
        new String[]{"https://data-cb03c.s3.amazonaws.com/reports/20190328060818/etc/com.simiacryptus.mindseye.style_transfer.StyleSurvey.1.png"},
        new String[]{"https://s3-us-west-2.amazonaws.com/simiacryptus/photos/vangogh-starry-night-ballance1.jpg"});
  }

  public LayerStyleSurvey(final String[] contentSources, final CharSequence[] styleSources) {
    this.contentSources = contentSources;
    this.styleSources = styleSources;
    this.verbose = true;
    this.maxIterations = 50;
    this.trainingMinutes = 30;
    this.maxResolution = 1400;
    startResolution = 600;
    this.minStyleWidth = 1200;
    plasmaResolution = startResolution / 8;
  }


  public void accept(@Nonnull NotebookOutput log) {
    TestUtil.addGlobalHandlers(log.getHttpd());

    log.eval(() -> {
      return JsonUtil.toJson(LayerStyleSurvey.this);
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
        Map<CharSequence, ColorTransfer<CVPipe_Inception.Strata, CVPipe_Inception>> styleColorTransforms = new HashMap<>();

        Tensor canvasBufferedImage = Tensor.fromRGB(ImageUtil.resize(
            ArtistryUtil.load(contentSource),
            resolution.get(),
            true
        ));
        canvasBufferedImage.assertAlive();
        final AtomicReference<Tensor> canvasImage = new AtomicReference<>(canvasBufferedImage);
        List<CharSequence> styleKeys = Arrays.asList(styleSources);
        ColorTransfer<CVPipe_Inception.Strata, CVPipe_Inception> contentColorTransform = new ColorTransfer.Inception();
        // Transfer color scheme:
        Tensor color_space_analog = log.subreport(sublog -> {
          //colorSyncContentCoeffMap.set(CVPipe_Inception.Strata.Layer_1a, 1e-1);
          return ImageArtUtil.colorTransfer(
              new ImageArtUtil.ImageArtOpParams(sublog, getMaxIterations(), getTrainingMinutes(), isVerbose()),
              contentColorTransform,
              ImageArtUtil.getColorAnalogSetup(
                  styleKeys,
                  precision,
                  canvasBufferedImage,
                  ImageArtUtil.getStyleImages(
                      styleColorTransforms, Math.max(resolution.get(), minStyleWidth), styleSources
                  ), CVPipe_Inception.Strata.Layer_0
              ),
              contentSource,
              startResolution,
              canvasImage.get().copy()
          );
        }, log.getName() + "_" + "Color_Space_Analog");
        log.p(log.png(color_space_analog.toImage(), "Style-Aligned Content Color"));
        canvasImage.set(color_space_analog);

//        // Seed initialization / degradation
//        canvasImage.set(ImageArtUtil.degrade(
//            plasmaResolution, resolution.get(), canvasImage.get()
//        ));

        List<CVPipe_Inception.Strata> styleLayers = Arrays.asList(
            CVPipe_Inception.Strata.Layer_1,
            CVPipe_Inception.Strata.Layer_2,
            CVPipe_Inception.Strata.Layer_3a,
//            CVPipe_Inception.Strata.Layer_3b,
            CVPipe_Inception.Strata.Layer_4a,
            CVPipe_Inception.Strata.Layer_4b,
//            CVPipe_Inception.Strata.Layer_4c,
//            CVPipe_Inception.Strata.Layer_4d,
            CVPipe_Inception.Strata.Layer_4e,
            CVPipe_Inception.Strata.Layer_5a
//            CVPipe_Inception.Strata.Layer_5b
        );
//        List<CVPipe_Inception.Strata> contentLayers = styleLayers;
        List<CVPipe_Inception.Strata> contentLayers = Arrays.asList(
            CVPipe_Inception.Strata.Layer_1,
            CVPipe_Inception.Strata.Layer_2,
            CVPipe_Inception.Strata.Layer_4b
        );
        for (CVPipe_Inception.Strata contentLayer : contentLayers) {
          log.h1("Content Strata " + contentLayer.name());
          for (CVPipe_Inception.Strata styleLayer : styleLayers) {
            log.h2("Style Strata " + styleLayer.name());
            Tensor result = log.subreport(sublog -> {
              final Tensor canvasImage1 = canvasImage.get().copy();
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
                      new SegmentedStyleTransfer.ContentCoefficients<CVPipe_Inception.Strata>().set(contentLayer, 1e0),
                      1e-1
                  ),
                  ImageArtUtil.getStyleImages(
                      styleColorTransforms, Math.max(resolution.get(), minStyleWidth), styleSources
                  ),
                  TestUtil.buildMap(x -> {
                    x.put(
                        styleKeys,
                        ImageArtUtil.getStyleCoefficients(
                            TestUtil.buildMap(m -> {
                              m.put(styleLayer, 1e0);
                            }),
                            1e0,
                            1e0,
                            0e-2
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
              final Tensor result1;
              final Tensor content = ArtistryUtil.loadTensor(contentSource, tileLayout.getCanvasDimensions()[0], tileLayout.getCanvasDimensions()[1]);
              if (tileLayout.getCols() > 1 || tileLayout.getRows() > 1) {
                result1 = ImageArtUtil.tiledTransfer(imageArtOpParams,
                    canvasImage1,
                    padding,
                    torroidalOffsetX,
                    torroidalOffsetY, tileLayout, transformer, content
                );
              } else {
                result1 = styleTransfer.transfer(
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
              return result1;
            }, log.getName() + "_" + String.format("%s_%s", styleLayer.name(), contentLayer.name()));
            log.eval(() -> {
              return contentColorTransform.inverseTransform(result).toImage();
            });
          }
        }
      } catch (Throwable throwable) {
        log.eval(() -> {
          return throwable;
        });
      }
    });
  }


  public static class EC2 {


    public static void main(String... args) throws Exception {
      EC2NotebookRunner.run(LocalNotebookRunner.getTask(LayerStyleSurvey.class));
    }
  }

}
