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

/**
 * The type Style survey.
 */
public class RebuildPyramid extends ImageScript {

  private final int tileSize;
  private final int startLevel;
  private final String bucket;
  private final String reportPath;
  private final String sourcePrefix;
  private final double aspect;
  private final String destPrefix;

  public RebuildPyramid(
      final String bucket,
      final String reportPath,
      final String sourcePrefix,
      final double aspect,
      final int startLevel, final String destPrefix
  ) {
    this.bucket = bucket;
    this.reportPath = reportPath;
    this.sourcePrefix = sourcePrefix;
    this.destPrefix = destPrefix;
    tileSize = 512;
    this.startLevel = startLevel;
    this.aspect = aspect;
  }

  public void accept(@Nonnull NotebookOutput log) {
    TestUtil.addGlobalHandlers(log.getHttpd());
    ((MarkdownNotebookOutput) log).setMaxImageSize(10000);
    log.eval(() -> {
      return JsonUtil.toJson(RebuildPyramid.this);
    });
    try {
      PyramidUtil.initJS(log);
//      String hadoopPrefix = "s3a://" + bucket + "/" + reportPath + "/etc/" + sourcePrefix;
      String hadoopPrefix = "https://" + bucket + ".s3.us-west-2.amazonaws.com/" + reportPath + "/etc/" + sourcePrefix;
      String filePrefix = ("file:///" + log.getResourceDir().getAbsolutePath()) + "/" + destPrefix;
      new ImagePyramid(tileSize, startLevel, aspect, hadoopPrefix).copyReducePyramid(filePrefix);
      new ImagePyramid(tileSize, startLevel, aspect, destPrefix).writeViewer(log);
      log.p(log.jpg(
          new ImagePyramid(tileSize, startLevel, aspect, filePrefix).assemble((int) (tileSize * Math.pow(2, this.startLevel))),
          "Full Image"
      ));
    } catch (Throwable throwable) {
      log.eval(() -> {
        return throwable;
      });
    }
  }

}
