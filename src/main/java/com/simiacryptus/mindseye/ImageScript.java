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

import com.simiacryptus.lang.SerializableConsumer;
import com.simiacryptus.notebook.NotebookOutput;

public abstract class ImageScript implements SerializableConsumer<NotebookOutput> {
  public boolean verbose = true;
  public int maxIterations = 20;
  public int trainingMinutes = 20;

  public boolean isVerbose() {
    return verbose;
  }

  public ImageScript setVerbose(boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  public int getMaxIterations() {
    return maxIterations;
  }

  public ImageScript setMaxIterations(int maxIterations) {
    this.maxIterations = maxIterations;
    return this;
  }

  public int getTrainingMinutes() {
    return trainingMinutes;
  }

  public ImageScript setTrainingMinutes(int trainingMinutes) {
    this.trainingMinutes = trainingMinutes;
    return this;
  }
}
