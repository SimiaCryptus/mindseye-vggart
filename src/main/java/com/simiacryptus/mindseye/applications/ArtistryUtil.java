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

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.simiacryptus.lang.Tuple2;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.AvgReducerLayer;
import com.simiacryptus.mindseye.layers.java.ImgTileSubnetLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.test.PCAUtil;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.notebook.FileHTTPD;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.util.FastRandom;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.data.DoubleStatistics;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;

import javax.annotation.Nonnull;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;
import java.util.function.IntUnaryOperator;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * The type Artistry util.
 */
public class ArtistryUtil {
  /**
   * Add layers handler.
   *
   * @param painterNetwork the painter network
   * @param server         the server
   */
  public static void addLayersHandler(final DAGNetwork painterNetwork, final FileHTTPD server) {
    if (null != server) server.addGET("layers.json", "application/json", out -> {
      try {
        JsonUtil.getMapper().writer().writeValue(out, TestUtil.samplePerformance(painterNetwork));
        out.close();
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
  }

  /**
   * Gram pipeline network.
   *
   * @param network      the network
   * @param mean         the mean
   * @param pcaTransform the pca transform
   * @return the pipeline network
   */
  @Nonnull
  public static PipelineNetwork gram(final PipelineNetwork network, Tensor mean, Tensor pcaTransform) {
    int[] dimensions = pcaTransform.getDimensions();
    int inputBands = mean.getDimensions()[2];
    int pcaBands = dimensions[2];
    int outputBands = pcaBands / inputBands;
    int width = dimensions[0];
    int height = dimensions[1];
    network.wrap(new ImgBandBiasLayer(mean.scale(-1))).freeRef();
    network.wrap(new ConvolutionLayer(width, height, inputBands, outputBands).set(pcaTransform)).freeRef();
    network.wrap(new GramianLayer()).freeRef();
    return network;
  }

  /**
   * Square avg pipeline network.
   *
   * @param network      the network
   * @param mean         the mean
   * @param pcaTransform the pca transform
   * @return the pipeline network
   */
  @Nonnull
  public static PipelineNetwork squareAvg(final PipelineNetwork network, Tensor mean, Tensor pcaTransform) {
    int[] dimensions = pcaTransform.getDimensions();
    int inputBands = mean.getDimensions()[2];
    int pcaBands = dimensions[2];
    int outputBands = pcaBands / inputBands;
    int width = dimensions[0];
    int height = dimensions[1];
    network.wrap(new ImgBandBiasLayer(mean.scale(-1))).freeRef();
    network.wrap(new ConvolutionLayer(width, height, inputBands, outputBands).set(pcaTransform)).freeRef();
    network.wrap(new SquareActivationLayer()).freeRef();
    network.wrap(new BandAvgReducerLayer()).freeRef();
    return network;
  }

  /**
   * Paint low res.
   *
   * @param canvas the canvas
   * @param scale  the scale
   */
  public static void paint_LowRes(final Tensor canvas, final int scale) {
    BufferedImage originalImage = canvas.toImage();
    canvas.set(Tensor.fromRGB(TestUtil.resize(
        TestUtil.resize(originalImage, originalImage.getWidth() / scale, true),
        originalImage.getWidth(), originalImage.getHeight()
    )));
  }

  /**
   * Paint lines.
   *
   * @param canvas the canvas
   */
  public static void paint_Lines(final Tensor canvas) {
    BufferedImage originalImage = canvas.toImage();
    BufferedImage newImage = new BufferedImage(originalImage.getWidth(), originalImage.getHeight(), BufferedImage.TYPE_INT_ARGB);
    Graphics2D graphics = (Graphics2D) newImage.getGraphics();
    IntStream.range(0, 100).forEach(i -> {
      Random random = new Random();
      graphics.setColor(new Color(random.nextInt(255), random.nextInt(255), random.nextInt(255)));
      graphics.drawLine(
          random.nextInt(originalImage.getWidth()),
          random.nextInt(originalImage.getHeight()),
          random.nextInt(originalImage.getWidth()),
          random.nextInt(originalImage.getHeight())
      );
    });
    canvas.set(Tensor.fromRGB(newImage));
  }

  /**
   * Paint circles.
   *
   * @param canvas the canvas
   * @param scale  the scale
   */
  public static void paint_Circles(final Tensor canvas, final int scale) {
    BufferedImage originalImage = canvas.toImage();
    BufferedImage newImage = new BufferedImage(originalImage.getWidth(), originalImage.getHeight(), BufferedImage.TYPE_INT_ARGB);
    Graphics2D graphics = (Graphics2D) newImage.getGraphics();
    IntStream.range(0, 10000).forEach(i -> {
      Random random = new Random();
      int positionX = random.nextInt(originalImage.getWidth());
      int positionY = random.nextInt(originalImage.getHeight());
      int width = 1 + random.nextInt(2 * scale);
      int height = 1 + random.nextInt(2 * scale);
      DoubleStatistics[] stats = {
          new DoubleStatistics(),
          new DoubleStatistics(),
          new DoubleStatistics()
      };
      canvas.coordStream(false).filter(c -> {
        int[] coords = c.getCoords();
        int x = coords[0];
        int y = coords[1];
        double relX = Math.pow(1 - 2 * ((double) (x - positionX) / width), 2);
        double relY = Math.pow(1 - 2 * ((double) (y - positionY) / height), 2);
        return relX + relY < 1.0;
      }).forEach(c -> stats[c.getCoords()[2]].accept(canvas.get(c)));
      graphics.setStroke(new Stroke() {
        @Override
        public Shape createStrokedShape(final Shape p) {
          return null;
        }
      });
      graphics.setColor(new Color(
          (int) stats[0].getAverage(),
          (int) stats[1].getAverage(),
          (int) stats[2].getAverage()
      ));
      graphics.fillOval(
          positionX,
          positionY,
          width,
          height
      );
    });
    canvas.set(Tensor.fromRGB(newImage));
  }

  /**
   * Paint plasma tensor.
   *
   * @param bands          the bands
   * @param noiseAmplitude the noise amplitude
   * @param noisePower     the noise power
   * @param size           the size
   * @return the tensor
   */
  public static Tensor paint_Plasma(int bands, final double noiseAmplitude, final double noisePower, final int size) {
    return expandPlasma(initSquare(bands), noiseAmplitude, noisePower, size, size);
  }

  public static Tensor paint_Plasma(int bands, final double noiseAmplitude, final double noisePower, final int width, final int height) {
    return expandPlasma(initSquare(bands), noiseAmplitude, noisePower, width, height);
  }

  @Nonnull
  private static Tensor initSquare(final int bands) {
    Tensor baseColor = new Tensor(1, 1, bands).setByCoord(c -> 100 + 200 * (Math.random() - 0.5));
    return new Tensor(2, 2, bands).setByCoord(c -> baseColor.get(0, 0, c.getCoords()[2]));
  }

  /**
   * Expand plasma tensor.
   *
   * @param image          the png
   * @param noiseAmplitude the noise amplitude
   * @param noisePower     the noise power
   * @param width          the width
   * @return the tensor
   */
  @Nonnull
  public static Tensor expandPlasma(Tensor image, final double noiseAmplitude, final double noisePower, final int width, final int height) {
    image.addRef();
    while (image.getDimensions()[0] < Math.max(width, height)) {
      Tensor newImage = expandPlasma(image, Math.pow(noiseAmplitude / image.getDimensions()[0], noisePower));
      image.freeRef();
      image = newImage;
    }
    Tensor tensor = Tensor.fromRGB(TestUtil.resize(image.toImage(), width, height));
    image.freeRef();
    return tensor;
  }

  @Nonnull
  public static Tensor expandPlasma(Tensor image, final double noiseAmplitude, final double noisePower, final int size) {
    image.addRef();
    while (image.getDimensions()[0] < size) {
      Tensor newImage = expandPlasma(image, Math.pow(noiseAmplitude / image.getDimensions()[0], noisePower));
      image.freeRef();
      image = newImage;
    }
    Tensor tensor = Tensor.fromRGB(TestUtil.resize(image.toImage(), size));
    image.freeRef();
    return tensor;
  }

  /**
   * Expand plasma tensor.
   *
   * @param seed  the seed
   * @param noise the noise
   * @return the tensor
   */
  public static Tensor expandPlasma(final Tensor seed, double noise) {
    int bands = seed.getDimensions()[2];
    int width = seed.getDimensions()[0] * 2;
    int height = seed.getDimensions()[1] * 2;
    Tensor returnValue = new Tensor(width, height, bands);
    DoubleUnaryOperator fn1 = x -> Math.max(Math.min(x + noise * (Math.random() - 0.5), 255), 0);
    DoubleUnaryOperator fn2 = x -> Math.max(Math.min(x + Math.sqrt(2) * noise * (Math.random() - 0.5), 255), 0);
    IntUnaryOperator addrX = x -> {
      while (x >= width) x -= width;
      while (x < 0) x += width;
      return x;
    };
    IntUnaryOperator addrY = x -> {
      while (x >= height) x -= height;
      while (x < 0) x += height;
      return x;
    };
    for (int band = 0; band < bands; band++) {
      for (int x = 0; x < width; x += 2) {
        for (int y = 0; y < height; y += 2) {
          double value = seed.get(x / 2, y / 2, band);
          returnValue.set(x, y, band, value);
        }
      }
      for (int x = 1; x < width; x += 2) {
        for (int y = 1; y < height; y += 2) {
          double value = (returnValue.get(addrX.applyAsInt(x - 1), addrY.applyAsInt(y - 1), band)) +
              (returnValue.get(addrX.applyAsInt(x - 1), addrY.applyAsInt(y + 1), band)) +
              (returnValue.get(addrX.applyAsInt(x + 1), addrY.applyAsInt(y - 1), band)) +
              (returnValue.get(addrX.applyAsInt(x + 1), addrY.applyAsInt(y + 1), band));
          value = fn2.applyAsDouble(value / 4);
          returnValue.set(x, y, band, value);
        }
      }
      for (int x = 0; x < width; x += 2) {
        for (int y = 1; y < height; y += 2) {
          double value = (returnValue.get(addrX.applyAsInt(x - 1), addrY.applyAsInt(y), band)) +
              (returnValue.get(addrX.applyAsInt(x + 1), addrY.applyAsInt(y), band)) +
              (returnValue.get(addrX.applyAsInt(x), addrY.applyAsInt(y - 1), band)) +
              (returnValue.get(addrX.applyAsInt(x), addrY.applyAsInt(y + 1), band));
          value = fn1.applyAsDouble(value / 4);
          returnValue.set(x, y, band, value);
        }
      }
      for (int x = 1; x < width; x += 2) {
        for (int y = 0; y < height; y += 2) {
          double value = (returnValue.get(addrX.applyAsInt(x - 1), addrY.applyAsInt(y), band)) +
              (returnValue.get(addrX.applyAsInt(x + 1), addrY.applyAsInt(y), band)) +
              (returnValue.get(addrX.applyAsInt(x), addrY.applyAsInt(y - 1), band)) +
              (returnValue.get(addrX.applyAsInt(x), addrY.applyAsInt(y + 1), band));
          value = fn1.applyAsDouble(value / 4);
          returnValue.set(x, y, band, value);
        }
      }
    }
    return returnValue;
  }

  /**
   * Avg pipeline network.
   *
   * @param network the network
   * @return the pipeline network
   */
  @Nonnull
  public static PipelineNetwork avg(final PipelineNetwork network) {
    network.wrap(new BandReducerLayer().setMode(PoolingLayer.PoolingMode.Avg)).freeRef();
    return network;
  }

  /**
   * With clamp pipeline network.
   *
   * @param network1 the network 1
   * @return the pipeline network
   */
  @Nonnull
  public static PipelineNetwork withClamp(final PipelineNetwork network1) {
    PipelineNetwork network = new PipelineNetwork(1);
    network.wrap(getClamp(255)).freeRef();
    network.wrap(network1).freeRef();
    return network;
  }

  /**
   * Sets precision.
   *
   * @param network   the network
   * @param precision the precision
   */
  public static void setPrecision(final DAGNetwork network, final Precision precision) {
    network.visitLayers(layer -> {
      if (layer instanceof MultiPrecision) {
        ((MultiPrecision) layer).setPrecision(precision);
      }
    });
  }

  /**
   * Pca tensor.
   *
   * @param cov   the bandCovariance
   * @param power the power
   * @return the tensor
   */
  @Nonnull
  public static Tensor pca(final Tensor cov, final double power) {
    final int inputbands = (int) Math.sqrt(cov.getDimensions()[2]);
    final int outputbands = inputbands;
    Array2DRowRealMatrix realMatrix = new Array2DRowRealMatrix(inputbands, inputbands);
    cov.coordStream(false).forEach(c -> {
      double v = cov.get(c);
      int x = c.getIndex() % inputbands;
      int y = (c.getIndex() - x) / inputbands;
      realMatrix.setEntry(x, y, v);
    });
    Tensor[] features = PCAUtil.pcaFeatures(realMatrix, outputbands, new int[]{1, 1, inputbands}, power);
    Tensor kernel = new Tensor(1, 1, inputbands * outputbands);
    PCAUtil.populatePCAKernel_1(kernel, features);
    return kernel;
  }

  /**
   * Gets clamp.
   *
   * @param max the max
   * @return the clamp
   */
  @Nonnull
  public static PipelineNetwork getClamp(final int max) {
    @Nonnull PipelineNetwork clamp = new PipelineNetwork(1);
    clamp.wrap(new ActivationLayer(ActivationLayer.Mode.RELU)).freeRef();
    clamp.wrap(new LinearActivationLayer().setBias(max).setScale(-1).freeze()).freeRef();
    clamp.wrap(new ActivationLayer(ActivationLayer.Mode.RELU)).freeRef();
    clamp.wrap(new LinearActivationLayer().setBias(max).setScale(-1).freeze()).freeRef();
    return clamp;
  }

  /**
   * To json string.
   *
   * @param obj the style parameters
   * @return the string
   */
  public static CharSequence toJson(final Object obj) {
    String json;
    try {
      ObjectMapper mapper = new ObjectMapper();
      mapper.configure(SerializationFeature.INDENT_OUTPUT, true);
      json = mapper.writeValueAsString(obj);
    } catch (JsonProcessingException e) {
      throw new RuntimeException(e);
    }
    return json;
  }

  /**
   * Load buffered png.
   *
   * @param image the style
   * @return the buffered png
   */
  @Nonnull
  public static BufferedImage load(final CharSequence image) {
    return HadoopUtil.getImage(image);
  }

  /**
   * Load buffered png.
   *
   * @param image     the png
   * @param imageSize the png size
   * @return the buffered png
   */
  @Nonnull
  public static BufferedImage load(final CharSequence image, final int imageSize) {
    BufferedImage source = HadoopUtil.getImage(image);
    return imageSize <= 0 ? source : TestUtil.resize(source, imageSize, true);
  }

  /**
   * Load buffered png.
   *
   * @param image  the png
   * @param width  the width
   * @param height the height
   * @return the buffered png
   */
  @Nonnull
  public static BufferedImage load(final CharSequence image, final int width, final int height) {
    BufferedImage bufferedImage = HadoopUtil.getImage(image);
    bufferedImage = TestUtil.resize(bufferedImage, width, height);
    return bufferedImage;
  }

  /**
   * Gram pipeline network.
   *
   * @param network the network
   * @param mean    the mean
   * @return the pipeline network
   */
  @Nonnull
  public static PipelineNetwork gram(final Layer network, Tensor mean) {
    if (!(network instanceof PipelineNetwork)) {
      PipelineNetwork pipelineNetwork = new PipelineNetwork();
      pipelineNetwork.wrap(network).freeRef();
      return gram(pipelineNetwork, mean);
    } else {
      PipelineNetwork pipelineNetwork = (PipelineNetwork) network;
      Tensor scale = mean.scale(-1);
      pipelineNetwork.wrap(new ImgBandBiasLayer(scale)).freeRef();
      scale.freeRef();
      pipelineNetwork.wrap(new GramianLayer()).freeRef();
      return pipelineNetwork;
    }
  }

  /**
   * Gram pipeline network.
   *
   * @param network the network
   * @return the pipeline network
   */
  @Nonnull
  public static PipelineNetwork gram(final Layer network) {
    if (!(network instanceof PipelineNetwork)) {
      PipelineNetwork pipelineNetwork = new PipelineNetwork();
      pipelineNetwork.wrap(pipelineNetwork).freeRef();
      return gram(pipelineNetwork);
    } else {
      PipelineNetwork pipelineNetwork = (PipelineNetwork) network;
      pipelineNetwork.wrap(new GramianLayer()).freeRef();
      return pipelineNetwork;
    }
  }

  /**
   * Randomize buffered png.
   *
   * @param contentImage the content png
   * @return the buffered png
   */
  @Nonnull
  public static BufferedImage randomize(final BufferedImage contentImage) {
    return randomize(contentImage, x -> FastRandom.INSTANCE.random());
  }

  /**
   * Randomize buffered png.
   *
   * @param contentImage the content png
   * @param f            the f
   * @return the buffered png
   */
  @Nonnull
  public static BufferedImage randomize(final BufferedImage contentImage, final DoubleUnaryOperator f) {
    return Tensor.fromRGB(contentImage).map(f).toRgbImage();
  }

  /**
   * Paint noise.
   *
   * @param canvas the canvas
   */
  public static void paint_noise(final Tensor canvas) {
    canvas.setByCoord(c -> FastRandom.INSTANCE.random());
  }

  /**
   * Wrap avg key.
   *
   * @param subnet the subnet
   * @return the key
   */
  protected static PipelineNetwork wrapAvg(final Layer subnet) {
    PipelineNetwork network = new PipelineNetwork(1);
    network.wrap(subnet).freeRef();
    network.wrap(new BandAvgReducerLayer()).freeRef();
    return network;
  }

  /**
   * Wrap tiled avg key.
   *
   * @param subnet the subnet
   * @param size   the size
   * @return the key
   */
  protected static PipelineNetwork wrapTiledAvg(final Layer subnet, final int size) {
    ImgTileSubnetLayer tileSubnetLayer = new ImgTileSubnetLayer(subnet, size, size, size, size);
    subnet.freeRef();
    return wrapAvg(tileSubnetLayer);
  }

  /**
   * Log exception with default t.
   *
   * @param <T>          the type parameter
   * @param log          the log
   * @param fn           the fn
   * @param defaultValue the default value
   * @return the t
   */
  public static <T> T logExceptionWithDefault(@Nonnull final NotebookOutput log, Supplier<T> fn, T defaultValue) {
    try {
      return fn.get();
    } catch (Throwable throwable) {
      try {
        log.eval(() -> {
          return throwable;
        });
      } catch (Throwable e2) {
      }
      return defaultValue;
    }
  }

  /**
   * Reduce.
   *
   * @param network               the network
   * @param functions             the functions
   * @param parallelLossFunctions the parallel loss functions
   */
  public static void reduce(final DAGNetwork network, final List<Tuple2<Double, DAGNode>> functions, final boolean parallelLossFunctions) {
    functions.stream().filter(x -> x._1 != 0).reduce((a, b) -> {
      return new Tuple2<>(1.0, network.wrap(new BinarySumLayer(a._1, b._1), a._2, b._2).setParallel(parallelLossFunctions));
    }).get();
  }

  /**
   * Gets files.
   *
   * @param file the file
   * @return the files
   */
  public static List<CharSequence> getHadoopFiles(CharSequence file) {
    return HadoopUtil.getFiles(file);
  }

  /**
   * Gets local files.
   *
   * @param file the file
   * @return the local files
   */
  public static List<CharSequence> getLocalFiles(CharSequence file) {
    File[] array = new File(file.toString()).listFiles();
    if (null == array) throw new IllegalArgumentException("Not Found: " + file);
    return Arrays.stream(array).map(File::getAbsolutePath).sorted(Comparator.naturalOrder()).collect(Collectors.toList());
  }


  /**
   * Tile cycle pipeline network.
   *
   * @param network the network
   * @param splits
   * @return the pipeline network
   */
  public static PipelineNetwork tileCycle(final PipelineNetwork network, final int splits) {
    PipelineNetwork netNet = new PipelineNetwork(1);
    netNet.wrap(
        new AvgReducerLayer(),
        Stream.concat(
            Stream.of(netNet.wrap(network, netNet.getInput(0))),
            IntStream.range(1, splits).mapToObj(i -> netNet.wrap(
                network,
                netNet.wrap(
                    new ImgTileCycleLayer().setXPos((double) i / splits).setYPos((double) i / splits),
                    netNet.getInput(0)
                )
            ))
        ).toArray(i -> new DAGNode[i])
    ).freeRef();
    return netNet;
  }

  public static Tensor loadTensor(final CharSequence contentSource, final int width, final int height) {
    return Tensor.fromRGB(load(contentSource, width, height));
  }

  public static Tensor loadTensor(final CharSequence contentSource, final int res) {
    return Tensor.fromRGB(load(contentSource, res));
  }
}