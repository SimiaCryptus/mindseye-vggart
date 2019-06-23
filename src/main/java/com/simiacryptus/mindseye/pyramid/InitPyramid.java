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

import com.simiacryptus.mindseye.applications.ArtistryUtil;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.notebook.MarkdownNotebookOutput;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.util.JsonUtil;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.util.Arrays;

public class InitPyramid extends ViewJSTest {

  public InitPyramid(final CharSequence[] styleSources) {
    super(styleSources);
  }

  public void accept(@Nonnull NotebookOutput log) {
    TestUtil.addGlobalHandlers(log.getHttpd());
    ((MarkdownNotebookOutput) log).setMaxImageSize(10000);
    log.eval(() -> {
      return JsonUtil.toJson(InitPyramid.this);
    });
    try {
      PyramidUtil.initJS(log);
      int tileSize = 512;
      PyramidUtil.initImagePyramids(
          log,
          "Image",
          tileSize,
          Arrays.stream(styleSources).map(styleSource -> ArtistryUtil.load(styleSource, -1)).toArray(i -> new BufferedImage[i])
      );
    } catch (Throwable throwable) {
      log.eval(() -> {
        return throwable;
      });
    }
  }

}
