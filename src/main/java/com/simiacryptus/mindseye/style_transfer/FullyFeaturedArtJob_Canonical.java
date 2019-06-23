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

import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.simiacryptus.aws.exe.EC2NotebookRunner;
import com.simiacryptus.aws.exe.LocalNotebookRunner;

import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class FullyFeaturedArtJob_Canonical extends FullyFeaturedArtJob {

  public FullyFeaturedArtJob_Canonical() {
    super(new String[]{
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Taj_Mahal_%28Edited%29.jpeg/1920px-Taj_Mahal_%28Edited%29.jpeg"
    }, new CharSequence[]{
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a5/Tsunami_by_hokusai_19th_century.jpg/350px-Tsunami_by_hokusai_19th_century.jpg"
    });
    this.maxResolution = 800;
    startResolution = 400;
  }

  public static class Local {
    public static void main(String... args) throws Exception {
      Executors.newScheduledThreadPool(1, new ThreadFactoryBuilder().setDaemon(true).build()).scheduleAtFixedRate(System::gc, 1, 1, TimeUnit.MINUTES);
      LocalNotebookRunner.run(LocalNotebookRunner.getTask(FullyFeaturedArtJob_Canonical.class));
    }
  }

  public static class EC2 {
    public static void main(String... args) throws Exception {
      EC2NotebookRunner.run(LocalNotebookRunner.getTask(FullyFeaturedArtJob_Canonical.class));
    }
  }

}
