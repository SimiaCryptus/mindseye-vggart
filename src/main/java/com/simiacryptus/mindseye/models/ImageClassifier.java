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

import com.google.common.collect.Lists;
import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.Explodable;
import com.simiacryptus.mindseye.layers.cudnn.ActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.conv.FullyConnectedLayer;
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.BiasLayer;
import com.simiacryptus.mindseye.layers.java.EntropyLossLayer;
import com.simiacryptus.mindseye.layers.java.LinearActivationLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.TrainingMonitor;
import com.simiacryptus.mindseye.opt.line.ArmijoWolfeSearch;
import com.simiacryptus.mindseye.opt.orient.QQN;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.notebook.MarkdownNotebookOutput;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.Closeable;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public abstract class ImageClassifier implements NetworkFactory {

  protected static final Logger logger = LoggerFactory.getLogger(ImageClassifier.class);
  protected volatile Layer cachedLayer;
  @Nullable
  protected
  Tensor prototype = new Tensor(224, 224, 3);
  protected int cnt = 1;
  @Nonnull
  protected
  Precision precision = Precision.Float;
  private int batchSize;

  public static List<LinkedHashMap<CharSequence, Double>> predict(
      @Nonnull Layer network,
      int count,
      @Nonnull List<CharSequence> categories,
      int batchSize,
      Tensor... data
  ) {
    return predict(network, count, categories, batchSize, true, false, data);
  }

  public static List<LinkedHashMap<CharSequence, Double>> predict(
      @Nonnull Layer network,
      int count,
      @Nonnull List<CharSequence> categories,
      int batchSize,
      boolean asyncGC,
      boolean nullGC,
      Tensor[] data
  ) {
    try {
      return Lists.partition(Arrays.asList(data), 1).stream().flatMap(batch -> {
        Tensor[][] input = {
            batch.stream().toArray(i -> new Tensor[i])
        };
        Result[] inputs = ConstantResult.singleResultArray(input);
        @Nullable Result result = network.eval(inputs);
        result.freeRef();
        TensorList resultData = result.getData();
        //Arrays.stream(input).flatMap(Arrays::stream).forEach(ReferenceCounting::freeRef);
        //Arrays.stream(inputs).forEach(ReferenceCounting::freeRef);
        //Arrays.stream(inputs).buildMap(Result::getData).forEach(ReferenceCounting::freeRef);

        List<LinkedHashMap<CharSequence, Double>> maps = resultData.stream().map(tensor -> {
          @Nullable double[] predictionSignal = tensor.getData();
          int[] order = IntStream.range(0, 1000).mapToObj(x -> x)
              .sorted(Comparator.comparing(i -> -predictionSignal[i]))
              .mapToInt(x -> x).toArray();
          assert categories.size() == predictionSignal.length;
          @Nonnull LinkedHashMap<CharSequence, Double> topN = new LinkedHashMap<>();
          for (int i = 0; i < count; i++) {
            int index = order[i];
            topN.put(categories.get(index), predictionSignal[index]);
          }
          tensor.freeRef();
          return topN;
        }).collect(Collectors.toList());
        resultData.freeRef();
        return maps.stream();
      }).collect(Collectors.toList());
    } finally {
    }
  }

  @Nonnull
  public static TrainingMonitor getTrainingMonitor(@Nonnull ArrayList<StepRecord> history, final PipelineNetwork network) {
    return TestUtil.getMonitor(history);
  }

  @Nonnull
  protected static Layer add(@Nonnull Layer layer, @Nonnull PipelineNetwork model) {
    name(layer);
    if (layer instanceof Explodable) {
      Layer explode = ((Explodable) layer).explode();
      try {
        if (explode instanceof DAGNetwork) {
          ((DAGNetwork) explode).visitNodes(node -> name(node.getLayer()));
          logger.info(String.format(
              "Exploded %s to %s (%s nodes)",
              layer.getName(),
              explode.getClass().getSimpleName(),
              ((DAGNetwork) explode).getNodes().size()
          ));
        } else {
          logger.info(String.format("Exploded %s to %s (%s nodes)", layer.getName(), explode.getClass().getSimpleName(), explode.getName()));
        }
        return add(explode, model);
      } finally {
        layer.freeRef();
      }
    } else {
      model.wrap(layer).freeRef();
      return layer;
    }
  }

  @Nonnull
  protected static Tensor evaluatePrototype(@Nonnull final Layer layer, final Tensor prevPrototype, int cnt) {
    int numberOfParameters = layer.state().stream().mapToInt(x -> x.length).sum();
    @Nonnull int[] prev_dimensions = prevPrototype.getDimensions();
    TensorList newPrototype = layer.eval(prevPrototype).getDataAndFree();
    try {
      @Nonnull int[] new_dimensions = newPrototype.getDimensions();
      logger.info(String.format("Added key #%d: %s; %s params, dimensions %s (%s) -> %s (%s)", //
          cnt, layer, numberOfParameters, //
          Arrays.toString(prev_dimensions), Tensor.length(prev_dimensions), //
          Arrays.toString(new_dimensions), Tensor.length(new_dimensions)
      ));
      return newPrototype.get(0);
    } finally {
      newPrototype.freeRef();
    }
  }

  protected static void name(final Layer layer) {
    if (layer.getName().contains(layer.getId().toString())) {
      if (layer instanceof ConvolutionLayer) {
        layer.setName(layer.getClass().getSimpleName() + ((ConvolutionLayer) layer).getConvolutionParams());
      } else if (layer instanceof SimpleConvolutionLayer) {
        layer.setName(String.format("%s: %s", layer.getClass().getSimpleName(),
            Arrays.toString(((SimpleConvolutionLayer) layer).getKernelDimensions())
        ));
      } else if (layer instanceof FullyConnectedLayer) {
        layer.setName(String.format(
            "%s:%sx%s",
            layer.getClass().getSimpleName(),
            Arrays.toString(((FullyConnectedLayer) layer).inputDims),
            Arrays.toString(((FullyConnectedLayer) layer).outputDims)
        ));
      } else if (layer instanceof BiasLayer) {
        layer.setName(String.format(
            "%s:%s",
            layer.getClass().getSimpleName(),
            ((BiasLayer) layer).bias.length()
        ));
      }
    }
  }

  public static void setPrecision(DAGNetwork model, final Precision precision) {
    model.visitLayers(layer -> {
      if (layer instanceof MultiPrecision) {
        ((MultiPrecision) layer).setPrecision(precision);
      }
    });
  }

  public void deepDream(@Nonnull final NotebookOutput log, final Tensor image) {
    @Nonnull ArrayList<StepRecord> history = new ArrayList<>();
    String training_name = String.format("etc/training_%s.png", Long.toHexString(MarkdownNotebookOutput.random.nextLong()));
    log.p(String.format("<a href=\"%s\"><img src=\"%s\"></a>", training_name, training_name));
    try (Closeable closeable = log.getHttpd().addGET(training_name, "image/png", r -> {
      try {
        ImageIO.write(Util.toImage(TestUtil.plot(history)), "png", r);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    })) {
      log.eval(() -> {
        @Nonnull PipelineNetwork clamp = new PipelineNetwork(1);
        clamp.wrap(new ActivationLayer(ActivationLayer.Mode.RELU)).freeRef();
        clamp.wrap(new LinearActivationLayer().setBias(255).setScale(-1).freeze()).freeRef();
        clamp.wrap(new ActivationLayer(ActivationLayer.Mode.RELU)).freeRef();
        clamp.wrap(new LinearActivationLayer().setBias(255).setScale(-1).freeze()).freeRef();
        @Nonnull PipelineNetwork supervised = new PipelineNetwork(1);
        supervised.add(getNetwork().freeze(), supervised.wrap(clamp, supervised.getInput(0))).freeRef();
//      CudaTensorList gpuInput = CudnnHandle.apply(gpu -> {
//        Precision precision = Precision.Float;
//        return CudaTensorList.wrap(gpu.getPtr(TensorArray.wrap(png), precision, MemoryType.Managed), 1, png.getDimensions(), precision);
//      });
//      @Nonnull Trainable trainable = new TensorListTrainable(supervised, gpuInput).setVerbosity(1).setMask(true);
        @Nonnull Trainable trainable = new ArrayTrainable(supervised, 1).setVerbose(true).setMask(
            true,
            false
        ).setData(Arrays.<Tensor[]>asList(new Tensor[]{image}));
        new IterativeTrainer(trainable)
            .setMonitor(getTrainingMonitor(history, supervised))
            .setOrientation(new QQN())
            .setLineSearchFactory(name -> new ArmijoWolfeSearch())
            .setTimeout(60, TimeUnit.MINUTES)
            .runAndFree();
        try {
          BufferedImage toImage = Util.toImage(TestUtil.plot(history));
          if (null != toImage) ImageIO.write(toImage, "png", log.file(training_name));
        } catch (IOException e) {
          logger.warn("Error writing result images", e);
        }
        return TestUtil.plot(history);
      });
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  public List<LinkedHashMap<CharSequence, Double>> predict(
      @Nonnull Layer network,
      int count,
      @Nonnull List<CharSequence> categories,
      @Nonnull Tensor... data
  ) {
    return predict(network, count, categories, Math.max(data.length, getBatchSize()), data);
  }

  public abstract List<CharSequence> getCategories();

  public List<LinkedHashMap<CharSequence, Double>> predict(int count, Tensor... data) {
    return predict(getNetwork(), count, getCategories(), data);
  }

  public List<LinkedHashMap<CharSequence, Double>> predict(@Nonnull Layer network, int count, Tensor[] data) {
    return predict(network, count, getCategories(), data);
  }

  public int getBatchSize() {
    return batchSize;
  }

  @Nonnull
  public ImageClassifier setBatchSize(int batchSize) {
    this.batchSize = batchSize;
    return this;
  }

  public void deepDream(
      @Nonnull final NotebookOutput log,
      final Tensor image,
      final int targetCategoryIndex,
      final int totalCategories,
      Function<IterativeTrainer, IterativeTrainer> config
  ) {
    deepDream(log, image, targetCategoryIndex, totalCategories, config, getNetwork(), new EntropyLossLayer(), -1.0);
  }

  @Nonnull
  @Override
  public Layer getNetwork() {
    if (null == cachedLayer) {
      synchronized (this) {
        if (null == cachedLayer) {
          try {
            cachedLayer = buildNetwork();
            setPrecision((DAGNetwork) cachedLayer);
            if (null != prototype) prototype.freeRef();
            prototype = null;
            return cachedLayer;
          } catch (@Nonnull final RuntimeException e) {
            throw e;
          } catch (Exception e) {
            throw new RuntimeException(e);
          }
        }
      }
    }
    return cachedLayer;


  }

  protected abstract Layer buildNetwork();

  public void deepDream(
      @Nonnull final NotebookOutput log,
      final Tensor image,
      final int targetCategoryIndex,
      final int totalCategories,
      Function<IterativeTrainer, IterativeTrainer> config,
      final Layer network,
      final Layer lossLayer,
      final double targetValue
  ) {
    @Nonnull List<Tensor[]> data = Arrays.<Tensor[]>asList(new Tensor[]{
        image, new Tensor(1, 1, totalCategories).set(targetCategoryIndex, targetValue)
    });
    log.run(() -> {
      for (Tensor[] tensors : data) {
        ImageClassifier.logger.info(log.png(tensors[0].toImage(), "") + tensors[1]);
      }
    });
    @Nonnull ArrayList<StepRecord> history = new ArrayList<>();
    String training_name = String.format("etc/training_%s.png", Long.toHexString(MarkdownNotebookOutput.random.nextLong()));
    log.p(String.format("<a href=\"%s\"><img src=\"%s\"></a>", training_name, training_name));
    Closeable png = log.getHttpd().addGET(training_name, "image/png", r -> {
      try {
        ImageIO.write(Util.toImage(TestUtil.plot(history)), "png", r);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
    log.eval(() -> {
      @Nonnull PipelineNetwork clamp = new PipelineNetwork(1);
      clamp.wrap(new ActivationLayer(ActivationLayer.Mode.RELU)).freeRef();
      clamp.wrap(new LinearActivationLayer().setBias(255).setScale(-1).freeze()).freeRef();
      clamp.wrap(new ActivationLayer(ActivationLayer.Mode.RELU)).freeRef();
      clamp.wrap(new LinearActivationLayer().setBias(255).setScale(-1).freeze()).freeRef();
      @Nonnull PipelineNetwork supervised = new PipelineNetwork(2);
      supervised.wrap(
          lossLayer,
          supervised.add(
              network.freeze(),
              supervised.wrap(clamp, supervised.getInput(0))
          ),
          supervised.getInput(1)
      ).freeRef();
//      TensorList[] gpuInput = data.stream().buildMap(data1 -> {
//        return CudnnHandle.apply(gpu -> {
//          Precision precision = Precision.Float;
//          return CudaTensorList.wrap(gpu.getPtr(TensorArray.wrap(data1), precision, MemoryType.Managed), 1, png.getDimensions(), precision);
//        });
//      }).toArray(i -> new TensorList[i]);
//      @Nonnull Trainable trainable = new TensorListTrainable(supervised, gpuInput).setVerbosity(1).setMask(true);

      @Nonnull Trainable trainable = new ArrayTrainable(supervised, 1).setVerbose(true).setMask(true, false).setData(data);
      config.apply(new IterativeTrainer(trainable)
          .setMonitor(getTrainingMonitor(history, supervised))
          .setOrientation(new QQN())
          .setLineSearchFactory(name -> new ArmijoWolfeSearch())
          .setTimeout(60, TimeUnit.MINUTES))
          .setTerminateThreshold(Double.NEGATIVE_INFINITY)
          .runAndFree();
      try {
        png.close();
        BufferedImage image1 = Util.toImage(TestUtil.plot(history));
        if (null != image1) ImageIO.write(image1, "png", log.file(training_name));
      } catch (IOException e) {
        logger.warn("Error writing result images", e);
      }
      return TestUtil.plot(history);
    });
  }

  protected void setPrecision(DAGNetwork model) {
    setPrecision(model, precision);
  }

}
