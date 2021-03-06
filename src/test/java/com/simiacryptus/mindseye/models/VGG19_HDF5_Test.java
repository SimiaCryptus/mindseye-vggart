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

package com.simiacryptus.mindseye.models;

import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;

public class VGG19_HDF5_Test extends ImageClassifierTestBase {

  @Override
  public ImageClassifier getImageClassifier(@Nonnull NotebookOutput log) {
//    @Nonnull PrintStream apiLog = new PrintStream(log.file("cuda.log"));
//    CudaSystem.addLog(apiLog);
//    log.p(log.file((String) null, "cuda.log", "GPU Log"));
    return log.eval(() -> {
      @Nonnull ImageClassifier vgg19_hdf5 = VGG19.fromHDF5();
      ((HasHDF5) vgg19_hdf5).getHDF5().print();
      return vgg19_hdf5;
    });
  }

  @Nonnull
  @Override
  protected Class<?> getTargetClass() {
    return VGG19.class;
  }

}
