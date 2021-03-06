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

package com.simiacryptus.mindseye.applications.dev.vgg19;

import com.simiacryptus.mindseye.applications.ArtistryAppBase_VGG19;
import com.simiacryptus.mindseye.applications.ArtistryData;
import com.simiacryptus.mindseye.applications.ArtistryUtil;
import com.simiacryptus.mindseye.applications.StyleTransfer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.models.CVPipe_VGG19;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.Stream;

public class ArtisticGradient extends ArtistryAppBase_VGG19 {

  @Nonnull
  public static BufferedImage init(final CharSequence contentSource, final int width) {
    BufferedImage canvasImage;
    canvasImage = ArtistryUtil.load(contentSource, width);
    canvasImage = ImageUtil.resize(canvasImage, width, true);
    canvasImage = ArtistryUtil.expandPlasma(Tensor.fromRGB(
        ImageUtil.resize(canvasImage, 16, true)),
        1000.0, 1.1, width
    ).toImage();
    return canvasImage;
  }

  @Nonnull
  public static <K, V> Map<K, V> create(Consumer<Map<K, V>> configure) {
    Map<K, V> map = new HashMap<>();
    configure.accept(map);
    return map;
  }

  public void run(@Nonnull NotebookOutput log) {
    StyleTransfer.VGG19 styleTransfer = new StyleTransfer.VGG19();
    init(log);
    Precision precision = Precision.Float;
    styleTransfer.parallelLossFunctions = true;
    styleTransfer.setTiled(false);
    int phases = 1;
    int geometricEnd = 4;
    int maxIterations = 10;
    int trainingMinutes = 90;
    int startImageSize = 400;
    double coeff_style_mean = 1e0;
    double coeff_style_cov = 0;
    Supplier<DoubleStream> contentCoeffStream = () -> DoubleStream.iterate(5e-1, x -> x * Math.pow(1e2, 1.0 / 4)).skip(2).limit(1);
    Supplier<DoubleStream> dreamCoeffStream = () -> DoubleStream.iterate(0, x -> 0).limit(1);
//    Supplier<DoubleStream> dreamCoeffStream = () -> DoubleStream.iterate(1e-3, x -> x * Math.pow(1e1, 1.0 / 4)).skip(2).limit(1);

    //ArtistryUtil.getHadoopFiles("file:///H:/SimiaCryptus/Artistry//space/");

    Stream.concat(
        ArtistryData.PLANETS.stream(),
        ArtistryData.CLASSIC_CONTENT.stream()
    ).forEach(contentSource -> {
      log.p(log.png(ArtistryUtil.load(contentSource), "Content Image"));
      writeGif(
          log,
          ArtistryData.CLASSIC_STYLES.stream().map(x -> Arrays.asList(x)).collect(Collectors.toList()).stream()
              .flatMap(styleSources -> {
                for (final CharSequence styleSource : styleSources) {
                  log.p(log.png(
                      ArtistryUtil.load(
                          styleSource),
                      "Style Image"
                  ));
                }
                return dreamCoeffStream.get()
                    .mapToObj(x -> x)
                    .flatMap(
                        dreamCoeff ->
                            contentCoeffStream.get().mapToObj(
                                contentMixingCoeff ->
                                    styleTransfer(
                                        log,
                                        styleTransfer,
                                        precision,
                                        new AtomicInteger(startImageSize),
                                        Math.pow(geometricEnd, 1.0 / (2 * phases)),
                                        contentSource,
                                        create(x -> x.put(
                                            styleSources,
                                            new StyleTransfer.StyleCoefficients<CVPipe_VGG19.Layer>(
                                                StyleTransfer.CenteringMode.Origin)
                                                .set(
                                                    CVPipe_VGG19.Layer.Layer_1a,
                                                    coeff_style_mean,
                                                    coeff_style_cov,
                                                    dreamCoeff
                                                )
                                                .set(
                                                    CVPipe_VGG19.Layer.Layer_1b,
                                                    coeff_style_mean,
                                                    coeff_style_cov,
                                                    dreamCoeff
                                                )
                                                .set(
                                                    CVPipe_VGG19.Layer.Layer_1c,
                                                    coeff_style_mean,
                                                    coeff_style_cov,
                                                    dreamCoeff
                                                )
                                            //.set(CVPipe_VGG19.Strata.Layer_1d, 1e0, 1e0, dreamCoeff)
                                        )),
                                        new StyleTransfer.ContentCoefficients<CVPipe_VGG19.Layer>()
                                            .set(
                                                CVPipe_VGG19.Layer.Layer_1c,
                                                contentMixingCoeff
                                            )
                                            .set(
                                                CVPipe_VGG19.Layer.Layer_1d,
                                                contentMixingCoeff
                                            ),
                                        trainingMinutes,
                                        maxIterations,
                                        phases
                                    ).toImage()));
              })
      );
    });

    log.setFrontMatterProperty("status", "OK");
  }

  public void writeGif(@Nonnull final NotebookOutput log, final Stream<BufferedImage> imageStream) {
    log.p(TestUtil.animatedGif(log, imageStream.toArray(i -> new BufferedImage[i])));
  }

  public Tensor styleTransfer(
      @Nonnull final NotebookOutput log,
      final StyleTransfer.VGG19 styleTransfer,
      final Precision precision,
      final AtomicInteger imageSize,
      final double growthFactor,
      final CharSequence contentSource,
      final Map<List<CharSequence>, StyleTransfer.StyleCoefficients<CVPipe_VGG19.Layer>> styles,
      final StyleTransfer.ContentCoefficients<CVPipe_VGG19.Layer> contentCoefficients,
      final int trainingMinutes,
      final int maxIterations,
      final int phases
  ) {
    Tensor canvasImage = Tensor.fromRGB(init(contentSource, imageSize.get()));
    for (int i = 0; i < phases; i++) {
      if (0 < i) {
        imageSize.set((int) (imageSize.get() * growthFactor));
        canvasImage = Tensor.fromRGB(ImageUtil.resize(canvasImage.toImage(), imageSize.get(), true));
      }
      StyleTransfer.StyleSetup<CVPipe_VGG19.Layer> styleSetup = new StyleTransfer.StyleSetup<CVPipe_VGG19.Layer>(
          precision,
          ArtistryUtil.loadTensor(
              contentSource,
              canvasImage.getDimensions()[0],
              canvasImage.getDimensions()[1]
          ),
          contentCoefficients,
          create(y -> y.putAll(
              styles.keySet().stream().flatMap(x -> x.stream())
                  .collect(Collectors.toMap(x -> x, file -> ArtistryUtil.load(file, imageSize.get()))))),
          styles
      );
      canvasImage = styleTransfer.transfer(log, canvasImage, styleSetup,
          trainingMinutes, styleTransfer.measureStyle(styleSetup), maxIterations, true
      );
    }
    return canvasImage;
  }

}
