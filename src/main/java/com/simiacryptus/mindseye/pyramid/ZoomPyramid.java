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

/**
 * The type Style survey.
 */
public class ZoomPyramid extends ImageScript {

  private final int tileSize;
  private final int magLevels;
  private final int padding;
  private final int startLevel;
  private final String bucket;
  private final String reportPath;
  private final String localPrefix;
  private final double aspect;

  public ZoomPyramid(final String bucket, final String reportPath, final String localPrefix, final double aspect, final int startLevel) {
    this.bucket = bucket;
    this.reportPath = reportPath;
    this.localPrefix = localPrefix;
    tileSize = 512;
    magLevels = 1;
    padding = 20;
    this.startLevel = startLevel;
    this.aspect = aspect;
  }

  public void accept(@Nonnull NotebookOutput log) {
    TestUtil.addGlobalHandlers(log.getHttpd());
    ((MarkdownNotebookOutput) log).setMaxImageSize(10000);
    log.eval(() -> {
      return JsonUtil.toJson(ZoomPyramid.this);
    });
    try {
      PyramidUtil.initJS(log);
      String hrefPrefix = "https://" + bucket + ".s3.us-west-2.amazonaws.com/" + reportPath + "/etc/" + localPrefix;
      String hadoopPrefix = "s3a://" + bucket + "/" + reportPath + "/etc/" + localPrefix;
      new ImagePyramid(tileSize, startLevel, aspect, hadoopPrefix).buildNewImagePyramidLayer(magLevels, padding,
          this::zoom,
          true
      );
      ImagePyramid imagePyramid = new ImagePyramid(tileSize, startLevel + magLevels, aspect, hrefPrefix);
      PyramidUtil.writeViewer(log, localPrefix + String.format("%d_%d", startLevel, magLevels + startLevel),
          imagePyramid
      );
    } catch (Throwable throwable) {
      log.eval(() -> {
        return throwable;
      });
    }
  }

  @Nonnull
  public BufferedImage zoom(final BufferedImage image) {
    return TestUtil.resize(image, (int) (image.getWidth() * Math.pow(2, magLevels)), true);
  }

}
