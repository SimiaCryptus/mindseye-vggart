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

import com.simiacryptus.lang.Tuple2;
import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.models.CVPipe;
import com.simiacryptus.mindseye.models.CVPipe_VGG16;
import com.simiacryptus.mindseye.models.CVPipe_VGG19;
import com.simiacryptus.mindseye.models.LayerEnum;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.line.BisectionSearch;
import com.simiacryptus.mindseye.opt.orient.TrustRegionStrategy;
import com.simiacryptus.mindseye.opt.region.RangeConstraint;
import com.simiacryptus.mindseye.opt.region.TrustRegion;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.notebook.FileHTTPD;
import com.simiacryptus.notebook.MarkdownNotebookOutput;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.NullNotebookOutput;
import com.simiacryptus.util.Util;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.Closeable;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;

public abstract class DeepDream<T extends LayerEnum<T>, U extends CVPipe<T>> {
  private static final Logger logger = LoggerFactory.getLogger(DeepDream.class);
  private boolean tiled = false;

  @Nonnull
  public Tensor deepDream(final Tensor canvasImage, final StyleSetup<T> styleParameters, final int trainingMinutes) {
    return deepDream(null, new NullNotebookOutput(), canvasImage, styleParameters, trainingMinutes, 50, true);
  }

  @Nonnull
  public Tensor deepDream(
      final FileHTTPD server,
      @Nonnull final NotebookOutput log,
      final Tensor canvasImage,
      final StyleSetup<T> styleParameters,
      final int trainingMinutes,
      final int maxIterations,
      final boolean verbose
  ) {
    PipelineNetwork network = fitnessNetwork(processStats(styleParameters));
    log.p("Input Parameters:");
    log.eval(() -> {
      return ArtistryUtil.toJson(styleParameters);
    });
    Tensor result = train(
        server,
        verbose ? log : new NullNotebookOutput(),
        canvasImage,
        network,
        styleParameters.precision,
        trainingMinutes,
        maxIterations
    );
    log.p("Result:");
    log.p(log.png(result.toImage(), "Result"));
    return result;
  }

  @Nonnull
  public Tensor train(
      final FileHTTPD server,
      @Nonnull final NotebookOutput log,
      final Tensor canvasImage,
      PipelineNetwork network,
      final Precision precision,
      final int trainingMinutes,
      final int maxIterations
  ) {
    System.gc();
    String imageName = String.format("etc/image_%s.jpg", Long.toHexString(MarkdownNotebookOutput.random.nextLong()));
    log.p(String.format("<a href=\"%s\"><img src=\"%s\"></a>", imageName, imageName));
    Closeable closeable = log.getHttpd().addGET(imageName, imageName + "/jpeg", r -> {
      try {
        ImageIO.write(canvasImage.toImage(), "jpeg", r);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
    ImageUtil.monitorImage(canvasImage, false, false);
    network.setFrozen(true);
    MultiPrecision.setPrecision(network, precision);
    TestUtil.instrumentPerformance(network);
    if (null != server) ArtistryUtil.addLayersHandler(network, server);
    if (tiled) network = ArtistryUtil.tileCycle(network, 3);
    train(log, network, canvasImage, trainingMinutes, maxIterations);
    try {
      closeable.close();
      ImageIO.write(canvasImage.toImage(), "jpeg", log.file(imageName));
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return canvasImage;
  }

  public void train(
      @Nonnull final NotebookOutput log,
      final PipelineNetwork network,
      final Tensor canvas,
      final int trainingMinutes,
      final int maxIterations
  ) {
    @Nonnull Trainable trainable = getTrainable(network, canvas);
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
      new IterativeTrainer(trainable)
          .setMonitor(TestUtil.getMonitor(history))
          .setIterationsPerSample(100)
          .setOrientation(new TrustRegionStrategy() {
            @Override
            public TrustRegion getRegionPolicy(final Layer layer) {
              return new RangeConstraint();
            }
          })
          .setMaxIterations(maxIterations)
          .setLineSearchFactory(name -> new BisectionSearch().setSpanTol(1e-1).setCurrentRate(1e3))
//        .setLineSearchFactory(name -> new QuadraticSearch().setRelativeTolerance(1e-1))
//        .setLineSearchFactory(name -> new ArmijoWolfeSearch())
          .setTimeout(trainingMinutes, TimeUnit.MINUTES)
          .setTerminateThreshold(Double.NEGATIVE_INFINITY)
          .runAndFree();
      try {
        png.close();
        BufferedImage image = Util.toImage(TestUtil.plot(history));
        if (null != image) ImageIO.write(image, "png", log.file(training_name));
      } catch (IOException e) {
        logger.warn("Error writing result images", e);
      }
      return TestUtil.plot(history);
    });
  }

  @Nonnull
  public Trainable getTrainable(final PipelineNetwork network, final Tensor canvas) {
    return new ArrayTrainable(network, 1).setVerbose(true).setMask(true).setData(Arrays.asList(new Tensor[][]{{canvas}}));
  }

  public NeuralSetup<T> processStats(final StyleSetup<T> style) {
    NeuralSetup<T> self = new NeuralSetup<>(style);
    self.contentTarget = new ContentTarget<>();
    for (final T layerType : getLayerTypes()) {
      System.gc();
      final PipelineNetwork network = layerType.network();
      ContentCoefficients contentCoefficients = style.coefficients.get(layerType);
      if (null != contentCoefficients && 0 != contentCoefficients.rms) {
        self.contentTarget.content.put(layerType, network.eval(style.contentImage).getDataAndFree().getAndFree(0));
        logger.info(String.format("target_content_%s=%s", layerType.name(), self.contentTarget.content.get(layerType).prettyPrint()));
      }
    }
    return self;
  }

  @Nonnull
  public PipelineNetwork fitnessNetwork(NeuralSetup<T> setup) {
    PipelineNetwork pipelineNetwork = getInstance().getNetwork();
    Map<T, DAGNode> nodes = new HashMap<>();
    Map<T, UUID> ids = getInstance().getNodes();
    ids.forEach((l, id) -> nodes.put(l, pipelineNetwork.getChildNode(id)));
    PipelineNetwork network = processStats(setup, nodes, pipelineNetwork);
    //network = withClamp(network);
    MultiPrecision.setPrecision(network, setup.style.precision);
    return network;
  }

  @Nonnull
  public List<Tuple2<Double, DAGNode>> getFitnessComponents(NeuralSetup<T> setup, final Map<T, DAGNode> nodeMap) {
    List<Tuple2<Double, DAGNode>> functions = new ArrayList<>();
    functions.addAll(getContentComponents(setup, nodeMap));
    return functions;
  }

  @Nonnull
  public abstract T[] getLayerTypes();

  @Nonnull
  public ArrayList<Tuple2<Double, DAGNode>> getContentComponents(NeuralSetup<T> setup, final Map<T, DAGNode> nodeMap) {
    ArrayList<Tuple2<Double, DAGNode>> contentComponents = new ArrayList<>();
    for (final T layerType : getLayerTypes()) {
      final DAGNode node = nodeMap.get(layerType);
      if (setup.style.coefficients.containsKey(layerType)) {
        DAGNetwork network = node.getNetwork();
        final double coeff_content = setup.style.coefficients.get(layerType).rms;
        if (0 != coeff_content) {
          Tensor contentSignal = setup.contentTarget.content.get(layerType);
          if (contentSignal != null) {
            contentComponents.add(new Tuple2<>(coeff_content, network.wrap(new MeanSqLossLayer(),
                node, network.wrap(new ValueLayer(contentSignal))
            )));
          } else {
            logger.info("No content signal for " + layerType);
          }
        }
        final double coeff_gain = setup.style.coefficients.get(layerType).gain;
        if (0 != coeff_gain) {
          contentComponents.add(new Tuple2<>(-coeff_gain, network.wrap(
              new AvgReducerLayer(),
              network.wrap(new SquareActivationLayer(), node)
          )));
        }
      }
    }
    return contentComponents;
  }

  public PipelineNetwork processStats(NeuralSetup<T> setup, final Map<T, DAGNode> nodeMap, final PipelineNetwork network) {
    List<Tuple2<Double, DAGNode>> functions = getFitnessComponents(setup, nodeMap);
    functions.stream().filter(x -> x._1 != 0).reduce((a, b) -> new Tuple2<>(1.0, network.wrap(new BinarySumLayer(a._1, b._1), a._2, b._2))).get();
    return network;
  }

  public abstract U getInstance();

  public boolean isTiled() {
    return tiled;
  }

  public DeepDream<T, U> setTiled(boolean tiled) {
    this.tiled = tiled;
    return this;
  }

  public static class VGG16 extends DeepDream<CVPipe_VGG16.Layer, CVPipe_VGG16> {

    public CVPipe_VGG16 getInstance() {
      return CVPipe_VGG16.INSTANCE;
    }

    @Nonnull
    public CVPipe_VGG16.Layer[] getLayerTypes() {
      return CVPipe_VGG16.Layer.values();
    }

  }

  public static class VGG19 extends DeepDream<CVPipe_VGG19.Layer, CVPipe_VGG19> {

    public CVPipe_VGG19 getInstance() {
      return CVPipe_VGG19.INSTANCE;
    }

    @Nonnull
    public CVPipe_VGG19.Layer[] getLayerTypes() {
      return CVPipe_VGG19.Layer.values();
    }

  }

  public static class StyleSetup<T extends LayerEnum<T>> {
    public final Precision precision;
    public final transient Tensor contentImage;
    public final Map<T, ContentCoefficients> coefficients;


    public StyleSetup(final Precision precision, final Tensor contentImage, Map<T, ContentCoefficients> contentCoefficients) {
      this.precision = precision;
      this.contentImage = contentImage;
      this.coefficients = contentCoefficients;
    }

  }

  public static class ContentCoefficients {
    public final double rms;
    public final double gain;

    public ContentCoefficients(final double rms, final double gain) {
      this.rms = rms;
      this.gain = gain;
    }
  }

  public static class ContentTarget<T extends LayerEnum<T>> {
    public Map<T, Tensor> content = new HashMap<>();
  }

  public class NeuralSetup<T extends LayerEnum<T>> {

    public final StyleSetup<T> style;
    public ContentTarget<T> contentTarget = new ContentTarget<>();

    public NeuralSetup(final StyleSetup<T> style) {
      this.style = style;
    }
  }

}
