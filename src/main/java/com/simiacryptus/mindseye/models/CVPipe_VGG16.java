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

import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.util.Util;

import javax.annotation.Nonnull;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

public class CVPipe_VGG16 implements CVPipe<CVPipe_VGG16.Layer> {
  public static final CVPipe_VGG16 INSTANCE = build();
  private final Map<Layer, UUID> nodes = new HashMap<>();
  private final Map<Layer, PipelineNetwork> prototypes = new HashMap<>();
  private PipelineNetwork network = new PipelineNetwork();

  private static CVPipe_VGG16 build() {
    CVPipe_VGG16 obj = new CVPipe_VGG16();
    try {
      new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
        @Override
        protected void phase0() {
          super.phase0();
          obj.nodes.put(Layer.Layer_0, pipeline.getHeadId());
          obj.prototypes.put(Layer.Layer_0, pipeline.copy());
        }

        @Override
        protected void phase1a() {
          super.phase1a();
          obj.nodes.put(Layer.Layer_1a, pipeline.getHeadId());
          obj.prototypes.put(Layer.Layer_1a, pipeline.copy());
        }

        @Override
        protected void phase1b() {
          super.phase1b();
          obj.nodes.put(Layer.Layer_1b, pipeline.getHeadId());
          obj.prototypes.put(Layer.Layer_1b, pipeline.copy());
        }

        @Override
        protected void phase1c() {
          super.phase1c();
          obj.nodes.put(Layer.Layer_1c, pipeline.getHeadId());
          obj.prototypes.put(Layer.Layer_1c, pipeline.copy());
        }

        @Override
        protected void phase1d() {
          super.phase1d();
          obj.nodes.put(Layer.Layer_1d, pipeline.getHeadId());
          obj.prototypes.put(Layer.Layer_1d, pipeline.copy());
        }

        @Override
        protected void phase1e() {
          super.phase1e();
          obj.nodes.put(Layer.Layer_1e, pipeline.getHeadId());
          obj.prototypes.put(Layer.Layer_1e, pipeline.copy());
        }

        @Override
        protected void phase2a() {
          super.phase2a();
          obj.nodes.put(Layer.Layer_2a, pipeline.getHeadId());
          obj.prototypes.put(Layer.Layer_2a, pipeline.copy());
        }

        @Override
        protected void phase2b() {
          super.phase2b();
          obj.nodes.put(Layer.Layer_2b, pipeline.getHeadId());
          obj.prototypes.put(Layer.Layer_2b, pipeline.copy());
        }

        @Override
        protected void phase3a() {
          super.phase3a();
          obj.nodes.put(Layer.Layer_3a, pipeline.getHeadId());
          obj.prototypes.put(Layer.Layer_3a, pipeline.copy());
        }

        @Override
        protected void phase3b() {
          super.phase3b();
          obj.nodes.put(Layer.Layer_3b, pipeline.getHeadId());
          obj.prototypes.put(Layer.Layer_3b, pipeline.copy());
          obj.network = (PipelineNetwork) pipeline.freeze();
          throw new RuntimeException("Abort Network Construction");
        }
      }.getNetwork();
    } catch (@Nonnull final RuntimeException e1) {
    } catch (Throwable e11) {
      throw new RuntimeException(e11);
    }
    return obj;
  }

  @Override
  public Map<Layer, UUID> getNodes() {
    return Collections.unmodifiableMap(nodes);
  }

  @Override
  public Map<Layer, PipelineNetwork> getPrototypes() {
    return Collections.unmodifiableMap(prototypes);
  }

  @Override
  public PipelineNetwork getNetwork() {
    return network.copy();
  }

  public enum Layer implements LayerEnum<Layer> {
    Layer_0,
    Layer_1a,
    Layer_1b,
    Layer_1c,
    Layer_1d,
    Layer_1e,
    Layer_2a,
    Layer_2b,
    Layer_3a,
    Layer_3b;

    public final PipelineNetwork network() {
      PipelineNetwork pipelineNetwork = INSTANCE.getPrototypes().get(this);
      if (null == pipelineNetwork) throw new IllegalStateException(this.toString());
      return null == pipelineNetwork ? null : pipelineNetwork.copy();
    }
  }
}
