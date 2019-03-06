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

package com.simiacryptus.mindseye.applications;

import com.simiacryptus.lang.ref.*;
import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.layers.PlaceholderLayer;
import com.simiacryptus.mindseye.layers.java.ImgTileAssemblyLayer;
import com.simiacryptus.mindseye.layers.java.ImgTileSelectLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.NullNotebookOutput;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Tiled trainable.
 */
public class TiledTrainable extends ReferenceCountingBase implements Trainable {
  private final PipelineNetwork network;
  private final Tensor canvas;
  private final int tileWidth = 600;
  private final int tileHeight = 600;
  private final int strideX = 600;
  private final int strideY = 600;
  private final NotebookOutput log;
  private final int padding;
  private boolean verbose;

  /**
   * Instantiates a new Tiled trainable.
   *
   * @param network the network
   * @param canvas  the canvas
   * @param padding
   */
  public TiledTrainable(final PipelineNetwork network, final Tensor canvas, final int padding, NotebookOutput log) {
    this.network = network;
    this.network.addRef();
    this.canvas = canvas;
    this.log = log;
    this.padding = padding;
    setVerbose(true);
  }

  public TiledTrainable(final PipelineNetwork network, final Tensor canvas, final int padding) {
    this(network, canvas, padding, new NullNotebookOutput());
  }

  @Override
  public PointSample measure(final TrainingMonitor monitor) {
    assert 3 == canvas.getDimensions().length;
    int width = canvas.getDimensions()[0];
    int height = canvas.getDimensions()[1];
    int cols = (int) (Math.ceil((width - tileWidth) * 1.0 / strideX) + 1);
    int rows = (int) (Math.ceil((height - tileHeight) * 1.0 / strideY) + 1);
    if (cols == 1 && rows == 1) {
      return new ArrayTrainable(network, 1).setVerbose(isVerbose()).setMask(true).setData(Arrays.asList(new Tensor[][]{{canvas}})).measure(monitor);
    } else {
      int tileSizeX = (cols <= 1) ? width : (int) Math.ceil(((double) (width - padding) / cols) + padding);
      int tileSizeY = (rows <= 1) ? height : (int) Math.ceil(((double) (height - padding) / rows) + padding);
      this.log.p(String.format(
          "Using Tile Size %s x %s to partition %s x %s png into %s x %s tiles",
          tileSizeX,
          tileSizeY,
          width,
          height,
          cols,
          rows
      ));
      Tensor[] tiles = ImgTileSelectLayer.toTiles(log, canvas, tileSizeX, tileSizeY, tileSizeX - padding, tileSizeY - padding, 0, 0);
      PointSample[] results = Arrays.stream(tiles).map(tile ->
          new ArrayTrainable(network, 1).setVerbose(true).setMask(true)
              .setData(Arrays.asList(new Tensor[][]{{tile}}))
              .measure(monitor))
          .toArray(i -> new PointSample[i]);
      List<Tensor> deltaList = IntStream.range(0, results.length).mapToObj(i -> {
        Delta<UUID> layerDelta = results[i].delta.stream().findAny().get();
        return new Tensor(layerDelta.getDelta(), tiles[i].getDimensions());
      }).collect(Collectors.toList());
      if (deltaList.size() != cols * rows) throw new AssertionError(deltaList.size() + " != " + cols + " * " + rows);
      final DeltaSet<UUID> delta = new DeltaSet<>();
      PlaceholderLayer<double[]> placeholderLayer = new PlaceholderLayer<>(canvas.getData());
      ImgTileAssemblyLayer assemblyLayer = new ImgTileAssemblyLayer(cols, rows)
          .setPaddingX(padding).setPaddingY(padding);
      Tensor assembled = assemblyLayer.eval(deltaList.toArray(new Tensor[]{})).getData().get(0);
      if (canvas.getData().length != assembled.getData().length) throw new IllegalStateException(
          String.format(
              "%d != %d (%s != %s)",
              canvas.getData().length,
              assembled.getData().length,
              Arrays.toString(canvas.getDimensions()),
              Arrays.toString(assembled.getDimensions())
          ));
      delta.get(placeholderLayer.getId(), canvas.getData()).set(assembled.getData());
      final StateSet<UUID> weights = new StateSet<>();
      weights.get(placeholderLayer.getId(), canvas.getData()).set(canvas.getData());
      final double sum = Arrays.stream(results).mapToDouble(x -> x.sum).average().getAsDouble();
      final double rate = Arrays.stream(results).mapToDouble(x -> x.rate).average().getAsDouble();
      final int count = Arrays.stream(results).mapToInt(x -> x.count).sum();
      return new PointSample(delta, weights, sum, rate, count);
    }
  }

  @Override
  public Layer getLayer() {
    return network;
  }

  public boolean isVerbose() {
    return verbose;
  }

  public TiledTrainable setVerbose(boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  @Override
  protected void _free() {
    this.network.addRef();
    super._free();
  }
}
