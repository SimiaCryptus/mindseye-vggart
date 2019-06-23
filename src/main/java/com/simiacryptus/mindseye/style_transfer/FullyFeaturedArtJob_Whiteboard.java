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

import com.simiacryptus.aws.exe.EC2NotebookRunner;
import com.simiacryptus.aws.exe.LocalNotebookRunner;

public class FullyFeaturedArtJob_Whiteboard extends FullyFeaturedArtJob {

  public FullyFeaturedArtJob_Whiteboard() {
    super(new String[]{
        "https://s3-us-west-2.amazonaws.com/simiacryptus/photos/IMG_20170423_200559814.jpg"
    }, new CharSequence[]{
        "https://s3-us-west-2.amazonaws.com/simiacryptus/photos/1280px-Nilssonlines2.png",
        "https://s3-us-west-2.amazonaws.com/simiacryptus/photos/Snub_dodecahedral_graph.png"
    });
    startResolution = 800;
    plasmaResolution = 800;
  }

  public static class Local {
    public static void main(String... args) throws Exception {
      LocalNotebookRunner.run(LocalNotebookRunner.getTask(FullyFeaturedArtJob_Whiteboard.class));
    }
  }

  public static class EC2 {
    public static void main(String... args) throws Exception {
      EC2NotebookRunner.run(LocalNotebookRunner.getTask(FullyFeaturedArtJob_Whiteboard.class));
    }
  }

}
