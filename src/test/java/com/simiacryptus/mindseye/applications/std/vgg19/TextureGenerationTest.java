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
import com.simiacryptus.mindseye.applications.TextureGeneration;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.models.CVPipe_VGG19;
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;

public class TextureGenerationTest extends ArtistryAppBase_VGG19 {

  public void run(@Nonnull NotebookOutput log) {
    TextureGeneration.VGG19 styleTransfer = new TextureGeneration.VGG19();
    init(log);
    Precision precision = Precision.Float;
    styleTransfer.parallelLossFunctions = true;
    double growthFactor = Math.sqrt(2);
    int trainingMinutes = 90;
    final int maxIterations = 5;
    Tensor canvas = Tensor.fromRGB(TextureGeneration.initCanvas(new AtomicInteger(256)));
    Map<List<CharSequence>, TextureGeneration.StyleCoefficients<CVPipe_VGG19.Layer>> textureStyle = new HashMap<>();
    textureStyle.put(
        ArtistryData.CLASSIC_STYLES.subList(0, 1),
        new TextureGeneration.StyleCoefficients<CVPipe_VGG19.Layer>(TextureGeneration.CenteringMode.Origin)
            .set(CVPipe_VGG19.Layer.Layer_0, 1e0, 1e0)
            .set(CVPipe_VGG19.Layer.Layer_1b, 1e0, 1e0)
            .set(CVPipe_VGG19.Layer.Layer_1d, 1e0, 1e0)
    );
    TextureGeneration.optimize(
        log,
        styleTransfer,
        precision,
        256,
        growthFactor,
        textureStyle,
        trainingMinutes,
        canvas,
        2,
        maxIterations,
        256
    );

    log.setFrontMatterProperty("status", "OK");
  }

}
