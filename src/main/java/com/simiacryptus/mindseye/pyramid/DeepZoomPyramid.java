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

package com.simiacryptus.mindseye.pyramid;

import com.simiacryptus.mindseye.ImageScript;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.notebook.MarkdownNotebookOutput;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.util.JsonUtil;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.util.function.Function;

public class DeepZoomPyramid extends ImageScript {

  private final int tileSize;
  private final int magLevels;
  private final int padding;
  private final int startLevel;
  private final String bucket;
  private final String reportPath;
  private final String localPrefix;
  private final double aspect;
  private final CharSequence[] styleSources;

  public DeepZoomPyramid(
      final String bucket,
      final String reportPath,
      final String localPrefix,
      final double aspect,
      final int startLevel,
      final String... styleSources
  ) {
    this.bucket = bucket;
    this.reportPath = reportPath;
    this.localPrefix = localPrefix;
    tileSize = 512;
    magLevels = 1;
    padding = 20;
    this.startLevel = startLevel;
    this.aspect = aspect;
    this.styleSources = styleSources;
  }

  public void accept(@Nonnull NotebookOutput log) {
    TestUtil.addGlobalHandlers(log.getHttpd());
    ((MarkdownNotebookOutput) log).setMaxImageSize(10000);
    log.eval(() -> {
      return JsonUtil.toJson(DeepZoomPyramid.this);
    });
    try {
      PyramidUtil.initJS(log);
      String hrefPrefix = "https://" + bucket + ".s3.us-west-2.amazonaws.com/" + reportPath + "/etc/" + localPrefix;
      String hadoopPrefix = "s3a://" + bucket + "/" + reportPath + "/etc/" + localPrefix;
      ImagePyramid pyramidLevel = new ImagePyramid(tileSize, startLevel, aspect, hadoopPrefix);
      PyramidUtil.writeViewer(log, "source", pyramidLevel);
      Function<BufferedImage, BufferedImage> imageFunction = PyramidUtil.getImageEnlargingFunction(
          log,
          (int) ((tileSize + 2 * padding) * Math.pow(2, magLevels)),
          (int) ((tileSize + 2 * padding) * Math.pow(2, magLevels)), getTrainingMinutes(), getMaxIterations(), isVerbose(), magLevels, padding,
          -1, styleSources
      );
      pyramidLevel.buildNewImagePyramidLayer(magLevels, padding, imageFunction, false);
      ImagePyramid imagePyramid = new ImagePyramid(tileSize, startLevel + magLevels, aspect, hrefPrefix);
      PyramidUtil.writeViewer(log, localPrefix + String.format("%d_%d", startLevel, magLevels + startLevel), imagePyramid);
    } catch (Throwable throwable) {
      log.eval(() -> {
        return throwable;
      });
    }
  }

}
