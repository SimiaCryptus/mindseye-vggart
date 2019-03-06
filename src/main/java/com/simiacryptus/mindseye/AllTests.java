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

package com.simiacryptus.mindseye;

import com.simiacryptus.aws.exe.EC2NotebookRunner;
import com.simiacryptus.aws.exe.LocalNotebookRunner;
import com.simiacryptus.mindseye.models.CVPipe_VGG19;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.List;

/**
 * The type All tests.
 */
public class AllTests {

  @Nonnull
  private static ImageScript[] getTests() {
    ImageScript[] scripts = {

        new com.simiacryptus.mindseye.deep_dream.Simple(),
        new com.simiacryptus.mindseye.style_transfer.Simple(),
        new com.simiacryptus.mindseye.texture_generation.Simple(),

        new com.simiacryptus.mindseye.deep_dream.HiDef(),

        new com.simiacryptus.mindseye.style_transfer.Enlarging(),
        new com.simiacryptus.mindseye.style_transfer.ParameterSweep(),
        new com.simiacryptus.mindseye.style_transfer.StyleSurvey(),
        new com.simiacryptus.mindseye.style_transfer.HiDef(),

        new com.simiacryptus.mindseye.texture_generation.Enlarging(),
        new com.simiacryptus.mindseye.texture_generation.ParameterSweep(),
        new com.simiacryptus.mindseye.texture_generation.HiDef(1e1,
            1e0,
            1e1,
            1600,
            "git://github.com/jcjohnson/fast-neural-style.git/master/images/styles/starry_night_crop.jpg"
        ) {

          @Nonnull
          @Override
          public List<CVPipe_VGG19.Layer> getLayers() {
            return Arrays.asList(
                CVPipe_VGG19.Layer.Layer_0,
                CVPipe_VGG19.Layer.Layer_1a,
                CVPipe_VGG19.Layer.Layer_1b,
                CVPipe_VGG19.Layer.Layer_1c
            );
          }
        }

    };
    for (final ImageScript script : scripts) {
      script.setVerbose(true).setMaxIterations(20).setTrainingMinutes(20);
    }
    return scripts;
  }

  /**
   * The type Local.
   */
  public static class Local {
    /**
     * The entry point of application.
     *
     * @param args the input arguments
     * @throws Exception the exception
     */
    public static void main(String... args) throws Exception {
      LocalNotebookRunner.run(
          getTests());
    }
  }

  /**
   * The type Ec 2.
   */
  public static class EC2 {
    /**
     * The entry point of application.
     *
     * @param args the input arguments
     * @throws Exception the exception
     */
    public static void main(String... args) throws Exception {
      EC2NotebookRunner.run(getTests());
    }
  }

}
