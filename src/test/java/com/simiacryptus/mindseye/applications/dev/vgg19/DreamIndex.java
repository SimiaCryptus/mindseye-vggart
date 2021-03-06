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
import com.simiacryptus.mindseye.applications.DeepDream;
import com.simiacryptus.mindseye.applications.TextureGeneration;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.models.CVPipe_VGG19;
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;
import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

public class DreamIndex extends ArtistryAppBase_VGG19 {

  public void run(@Nonnull NotebookOutput log) {
    TextureGeneration.VGG19 styleTransfer = new TextureGeneration.VGG19();
    DeepDream.VGG19 deepDream = new DeepDream.VGG19();
    deepDream.setTiled(true);
    init(log);
    Precision precision = Precision.Float;
    styleTransfer.parallelLossFunctions = true;
    double growthFactor = Math.sqrt(2);
    int iterations = 10;
    int trainingMinutes = 90;

    for (CharSequence file : ArtistryData.CLASSIC_STYLES) {
      log.h2("Image: " + file);
      try {
        log.p(log.png(ImageIO.read(new File(file.toString())), "Input Image"));
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
      for (final CVPipe_VGG19.Layer layer : CVPipe_VGG19.Layer.values()) {
        Tensor canvas = Tensor.fromRGB(TextureGeneration.initCanvas(new AtomicInteger(256)));
        log.h2("Strata: " + layer);
        Map<List<CharSequence>, TextureGeneration.StyleCoefficients<CVPipe_VGG19.Layer>> textureStyle = new HashMap<>();
        TextureGeneration.StyleCoefficients<CVPipe_VGG19.Layer> tStyleCoefficients = new TextureGeneration.StyleCoefficients<>(TextureGeneration.CenteringMode.Origin);
        tStyleCoefficients.set(layer, 1e0, 1e0);
        textureStyle.put(Arrays.asList(file), tStyleCoefficients);
        canvas = TextureGeneration.optimize(
            log,
            styleTransfer,
            precision,
            256,
            growthFactor,
            textureStyle,
            trainingMinutes,
            canvas,
            1,
            iterations,
            0
        );
        Map<CVPipe_VGG19.Layer, DeepDream.ContentCoefficients> dreamCoeff = new HashMap<>();
        dreamCoeff.put(layer, new DeepDream.ContentCoefficients(0, 1e0));
        deepDream.deepDream(
            log.getHttpd(),
            log,
            canvas,
            new DeepDream.StyleSetup<>(precision, canvas, dreamCoeff),
            trainingMinutes,
            iterations,
            true
        );
      }
    }

    log.setFrontMatterProperty("status", "OK");
  }

}
