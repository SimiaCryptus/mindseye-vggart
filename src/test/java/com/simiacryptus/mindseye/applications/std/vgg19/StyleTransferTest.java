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

package com.simiacryptus.mindseye.applications.std.vgg19;

import com.simiacryptus.mindseye.applications.ArtistryAppBase_VGG19;
import com.simiacryptus.mindseye.applications.ArtistryData;
import com.simiacryptus.mindseye.applications.ArtistryUtil;
import com.simiacryptus.mindseye.applications.StyleTransfer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.models.CVPipe_VGG19;
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

public class StyleTransferTest extends ArtistryAppBase_VGG19 {

  public void run(@Nonnull NotebookOutput log) {
    StyleTransfer.VGG19 styleTransfer = new StyleTransfer.VGG19();
    init(log);
    Precision precision = Precision.Float;
    final AtomicInteger imageSize = new AtomicInteger(256);
    styleTransfer.parallelLossFunctions = true;
    double growthFactor = Math.sqrt(4);

    Map<List<CharSequence>, StyleTransfer.StyleCoefficients<CVPipe_VGG19.Layer>> styles = new HashMap<>();
    List<CharSequence> styleSources = ArtistryData.CLASSIC_STYLES;
    styles.put(styleSources, new StyleTransfer.StyleCoefficients<CVPipe_VGG19.Layer>(StyleTransfer.CenteringMode.Origin)
        .set(CVPipe_VGG19.Layer.Layer_1a, 1e0, 1e0)
        .set(CVPipe_VGG19.Layer.Layer_1b, 1e0, 1e0)
        .set(CVPipe_VGG19.Layer.Layer_1c, 1e0, 1e0)
    );
    StyleTransfer.ContentCoefficients<CVPipe_VGG19.Layer> contentCoefficients = new StyleTransfer.ContentCoefficients<CVPipe_VGG19.Layer>()
        .set(CVPipe_VGG19.Layer.Layer_1b, 3e0)
        .set(CVPipe_VGG19.Layer.Layer_1c, 3e0);
    int trainingMinutes = 90;
    int maxIterations = 10;
    int phases = 2;

    log.h1("Phase 0");
    Tensor canvasImage = ArtistryUtil.loadTensor(ArtistryData.CLASSIC_STYLES.get(0), imageSize.get());
    canvasImage = Tensor.fromRGB(ImageUtil.resize(canvasImage.toImage(), imageSize.get(), true));
    canvasImage = ArtistryUtil.expandPlasma(Tensor.fromRGB(ImageUtil.resize(canvasImage.toImage(), 16, true)), 1000.0, 1.1, imageSize.get());
    Tensor contentImage = ArtistryUtil.loadTensor(
        ArtistryData.CLASSIC_CONTENT.get(0),
        canvasImage.getDimensions()[0],
        canvasImage.getDimensions()[1]
    );
    Map<CharSequence, BufferedImage> styleImages = new HashMap<>();
    StyleTransfer.StyleSetup<CVPipe_VGG19.Layer> styleSetup;

    styleImages.clear();
    styleImages.putAll(styles.keySet().stream().flatMap(x -> x.stream()).collect(Collectors.toMap(
        x -> x,
        file -> ArtistryUtil.load(file, imageSize.get())
    )));
    styleSetup = new StyleTransfer.StyleSetup<>(precision, contentImage, contentCoefficients, styleImages, styles);

    StyleTransfer.NeuralSetup measureStyle = styleTransfer.measureStyle(styleSetup);
    canvasImage = styleTransfer.transfer(log, canvasImage, styleSetup, trainingMinutes, measureStyle, maxIterations, true);
    for (int i = 1; i < phases; i++) {
      log.h1("Phase " + i);
      imageSize.set((int) (imageSize.get() * growthFactor));
      canvasImage = Tensor.fromRGB(ImageUtil.resize(canvasImage.toImage(), imageSize.get(), true));

      styleImages.clear();
      styleImages.putAll(styles.keySet().stream().flatMap(x -> x.stream()).collect(Collectors.toMap(
          x -> x,
          file -> ArtistryUtil.load(file, imageSize.get())
      )));
      styleSetup = new StyleTransfer.StyleSetup<>(precision, contentImage, contentCoefficients, styleImages, styles);

      canvasImage = styleTransfer.transfer(log, canvasImage, styleSetup, trainingMinutes, measureStyle, maxIterations, true);
    }
    log.setFrontMatterProperty("status", "OK");
  }

}
