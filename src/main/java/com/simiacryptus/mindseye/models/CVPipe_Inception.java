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

import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.util.TFConverter;
import com.simiacryptus.tensorflow.ImageNetworkPipeline;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.*;
import java.util.stream.Collectors;

public class CVPipe_Inception implements CVPipe<com.simiacryptus.mindseye.models.CVPipe_Inception.Strata> {

  public static final Logger logger = LoggerFactory.getLogger(CVPipe_Inception.class);
  public static final CVPipe_Inception INSTANCE = build();

  private CVPipe_Inception() {
  }

  private static CVPipe_Inception build() {
    CVPipe_Inception obj = new CVPipe_Inception();
    final String abortMsg = "Abort Network Construction";
    try {
      obj.pipelineNetwork = obj.init();
      assert null != obj.prototypes;
      assert !obj.prototypes.isEmpty();
    } catch (@Nonnull final RuntimeException e1) {
      if (!e1.getMessage().equals(abortMsg)) {
        logger.warn("Err", e1);
        throw new RuntimeException(e1);
      }
    } catch (Throwable e11) {
      logger.warn("Error", e11);
      throw new RuntimeException(e11);
    }
    return obj;
  }

  @Override
  public Map<com.simiacryptus.mindseye.models.CVPipe_Inception.Strata, UUID> getNodes() {
    return Collections.unmodifiableMap(nodes);
  }

  @Override
  public Map<com.simiacryptus.mindseye.models.CVPipe_Inception.Strata, PipelineNetwork> getPrototypes() {
    assert null != prototypes;
    assert !prototypes.isEmpty();
    return Collections.unmodifiableMap(prototypes);
  }

  private PipelineNetwork pipelineNetwork = null;
  private final Map<com.simiacryptus.mindseye.models.CVPipe_Inception.Strata, UUID> nodes = new HashMap<>();
  private final Map<com.simiacryptus.mindseye.models.CVPipe_Inception.Strata, PipelineNetwork> prototypes = new HashMap<>();

  @Override
  public PipelineNetwork getNetwork() {
    return pipelineNetwork.copy();
  }

  @NotNull
  private PipelineNetwork init() {
    PipelineNetwork pipelineNetwork = new PipelineNetwork();
    ImageNetworkPipeline imageNetworkPipeline = ImageNetworkPipeline.inception5h();
    List<com.simiacryptus.mindseye.models.CVPipe_Inception.Strata> strataList = Arrays.stream(CVPipe_Inception.Strata.values()).sorted(Comparator.comparing(x -> x.index)).collect(Collectors.toList());
    strataList.stream().forEach(strata->{
      Layer tfLayer = new TFConverter().convert(TFConverter.getLayer(imageNetworkPipeline, strata.index));
      nodes.put(strata, pipelineNetwork.wrap(tfLayer).getId());
      prototypes.put(strata, pipelineNetwork.copy());
    });
    return pipelineNetwork;
  }

//  public enum Layer implements LayerEnum<Layer> {
//    X;
//
//    @Override
//    public PipelineNetwork network() {
//      return null;
//    }
//  }

  public enum Strata implements LayerEnum<com.simiacryptus.mindseye.models.CVPipe_Inception.Strata> {
    Layer_1("conv2d0", 0),
    Layer_2("localresponsenorm1", 1),
    Layer_3a("mixed3a", 2),
    Layer_3b("mixed3b", 3),
    Layer_4a("mixed4a", 4),
    Layer_4b("mixed4b", 5),
    Layer_4c("mixed4c", 6),
    Layer_4d("mixed4d", 7),
    Layer_4e("mixed4e", 8),
    Layer_5a("mixed5a", 9),
    Layer_5b("mixed5b", 10);

    public static final Strata Layer_0 = Layer_1;
    public static final Strata Layer_1a = Layer_2;
    public static final Strata Layer_1b = Layer_3a;
    public static final Strata Layer_1c = Layer_3b;
    public static final Strata Layer_1d = Layer_4a;
    public static final Strata Layer_1e = Layer_4b;
    public final String id;
    public final int index;

    Strata(String id, int index) {
      this.index = index;
      this.id = id;
    }

    public final PipelineNetwork network() {
      PipelineNetwork pipelineNetwork = INSTANCE.getPrototypes().get(this);
      if (null == pipelineNetwork) throw new IllegalStateException(this.toString());
      return null == pipelineNetwork ? null : pipelineNetwork.copy();
    }
  }
}
