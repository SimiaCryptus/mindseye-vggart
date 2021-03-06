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
import com.simiacryptus.notebook.FileHTTPD;
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.stream.Stream;

public class TextureDream extends ArtistryAppBase_VGG19 {

  public void run(@Nonnull NotebookOutput log) {
    TextureGeneration.VGG19 styleTransfer = new TextureGeneration.VGG19();
    DeepDream.VGG19 deepDream = new DeepDream.VGG19();
    deepDream.setTiled(true);
    init(log);
    Precision precision = Precision.Float;
    styleTransfer.parallelLossFunctions = true;

    double growthFactor = Math.sqrt(2);
    int imageSize = 256;
    int styleSize = 512;
    int phases = 1;
    int maxIterations = 10;
    int trainingMinutes = 90;


    Stream<List<CharSequence>> listStream = ArtistryData.CLASSIC_STYLES.stream().map(x -> Arrays.asList(x));
    listStream.forEach(styleSources -> {

      final FileHTTPD server = log.getHttpd();
      TextureGeneration.optimize(
          log,
          styleTransfer,
          precision,
          imageSize,
          growthFactor,
          create(map -> map.put(styleSources, new TextureGeneration.StyleCoefficients<CVPipe_VGG19.Layer>(TextureGeneration.CenteringMode.Origin)
              .set(CVPipe_VGG19.Layer.Layer_1a, 1e0, 1e0)
              .set(CVPipe_VGG19.Layer.Layer_1c, 1e0, 1e0, 1e0))),
          trainingMinutes,
          Tensor.fromRGB(TextureGeneration.initCanvas(new AtomicInteger(imageSize))),
          phases,
          maxIterations,
          styleSize
      );

      TextureGeneration.optimize(
          log,
          styleTransfer,
          precision,
          imageSize,
          growthFactor,
          create(map -> map.put(styleSources, new TextureGeneration.StyleCoefficients<CVPipe_VGG19.Layer>(TextureGeneration.CenteringMode.Origin)
              .set(CVPipe_VGG19.Layer.Layer_1b, 1e0, 1e0)
              .set(CVPipe_VGG19.Layer.Layer_1c, 1e0, 1e0, 1e0)
          )),
          trainingMinutes,
          Tensor.fromRGB(TextureGeneration.initCanvas(new AtomicInteger(imageSize))),
          phases,
          maxIterations,
          styleSize
      );

      TextureGeneration.optimize(
          log,
          styleTransfer,
          precision,
          imageSize,
          growthFactor,
          create(map -> map.put(styleSources, new TextureGeneration.StyleCoefficients<CVPipe_VGG19.Layer>(TextureGeneration.CenteringMode.Origin)
              .set(CVPipe_VGG19.Layer.Layer_1b, 1e0, 1e0, 1e0)
              .set(CVPipe_VGG19.Layer.Layer_1c, 1e0, 1e0)
          )),
          trainingMinutes,
          Tensor.fromRGB(TextureGeneration.initCanvas(new AtomicInteger(imageSize))),
          phases,
          maxIterations,
          styleSize
      );

      TextureGeneration.optimize(
          log,
          styleTransfer,
          precision,
          imageSize,
          growthFactor,
          create(map -> map.put(styleSources, new TextureGeneration.StyleCoefficients<CVPipe_VGG19.Layer>(TextureGeneration.CenteringMode.Origin)
              .set(CVPipe_VGG19.Layer.Layer_1b, 1e0, 1e0)
              .set(CVPipe_VGG19.Layer.Layer_1d, 1e0, 1e0, 1e0)
          )),
          trainingMinutes,
          Tensor.fromRGB(TextureGeneration.initCanvas(new AtomicInteger(imageSize))),
          phases,
          maxIterations,
          styleSize
      );

      TextureGeneration.optimize(
          log,
          styleTransfer,
          precision,
          imageSize,
          growthFactor,
          create(map -> map.put(styleSources, new TextureGeneration.StyleCoefficients<CVPipe_VGG19.Layer>(TextureGeneration.CenteringMode.Origin)
              .set(CVPipe_VGG19.Layer.Layer_1a, 1e0, 1e0)
              .set(CVPipe_VGG19.Layer.Layer_1c, 1e0, 1e0, 0)
          )),
          trainingMinutes,
          Tensor.fromRGB(TextureGeneration.initCanvas(new AtomicInteger(imageSize))),
          phases,
          maxIterations,
          styleSize
      );

      TextureGeneration.optimize(
          log,
          styleTransfer,
          precision,
          imageSize,
          growthFactor,
          create(map -> map.put(styleSources, new TextureGeneration.StyleCoefficients<CVPipe_VGG19.Layer>(TextureGeneration.CenteringMode.Origin)
              .set(CVPipe_VGG19.Layer.Layer_1b, 1e0, 1e0)
              .set(CVPipe_VGG19.Layer.Layer_1c, 1e0, 1e0, 0)
          )),
          trainingMinutes,
          Tensor.fromRGB(TextureGeneration.initCanvas(new AtomicInteger(imageSize))),
          phases,
          maxIterations,
          styleSize
      );

      TextureGeneration.optimize(
          log,
          styleTransfer,
          precision,
          imageSize,
          growthFactor,
          create(map -> map.put(styleSources, new TextureGeneration.StyleCoefficients<CVPipe_VGG19.Layer>(TextureGeneration.CenteringMode.Origin)
              .set(CVPipe_VGG19.Layer.Layer_1b, 1e0, 1e0, 0)
              .set(CVPipe_VGG19.Layer.Layer_1c, 1e0, 1e0)
          )),
          trainingMinutes,
          Tensor.fromRGB(TextureGeneration.initCanvas(new AtomicInteger(imageSize))),
          phases,
          maxIterations,
          styleSize
      );

      TextureGeneration.optimize(
          log,
          styleTransfer,
          precision,
          imageSize,
          growthFactor,
          create(map -> map.put(styleSources, new TextureGeneration.StyleCoefficients<CVPipe_VGG19.Layer>(TextureGeneration.CenteringMode.Origin)
              .set(CVPipe_VGG19.Layer.Layer_1b, 1e0, 1e0)
              .set(CVPipe_VGG19.Layer.Layer_1d, 1e0, 1e0, 0)
          )),
          trainingMinutes,
          Tensor.fromRGB(TextureGeneration.initCanvas(new AtomicInteger(imageSize))),
          phases,
          maxIterations,
          styleSize
      );
    });

    log.setFrontMatterProperty("status", "OK");
  }

  @Nonnull
  public <K, V> Map<K, V> create(Consumer<Map<K, V>> configure) {
    Map<K, V> map = new HashMap<>();
    configure.accept(map);
    return map;
  }

}
