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
import com.simiacryptus.lang.ref.ReferenceCountingBase;
import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer;
import com.simiacryptus.mindseye.models.*;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.line.LineSearchCursor;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.orient.LBFGS;
import com.simiacryptus.mindseye.opt.orient.OrientationStrategy;
import com.simiacryptus.mindseye.opt.orient.TrustRegionStrategy;
import com.simiacryptus.mindseye.opt.region.OrthonormalConstraint;
import com.simiacryptus.mindseye.opt.region.TrustRegion;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.notebook.FileHTTPD;
import com.simiacryptus.notebook.MarkdownNotebookOutput;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.notebook.NullNotebookOutput;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.data.ScalarStatistics;
import org.apache.commons.math3.linear.BlockRealMatrix;
import org.apache.commons.math3.linear.LUDecomposition;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.Closeable;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public abstract class ColorTransfer<T extends LayerEnum<T>, U extends CVPipe<T>> {

  private static final Logger logger = LoggerFactory.getLogger(ColorTransfer.class);
  public boolean parallelLossFunctions = true;
  private SimpleConvolutionLayer colorForwardTransform;
  private boolean ortho = true;
  private boolean unit = true;

  @Nonnull
  public static SimpleConvolutionLayer invert(final SimpleConvolutionLayer colorForwardTransform) {
    try {
      colorForwardTransform.assertAlive();
      SimpleConvolutionLayer invConv = new SimpleConvolutionLayer(1, 1, 9);
      RealMatrix matrix = getMatrix(colorForwardTransform.kernel);
      RealMatrix inverse = inverse(matrix);
      setMatrix(invConv.kernel, inverse);
      return invConv;
    } catch (Throwable e1) {
      logger.info("Error inverting kernel", e1);
      return unitTransformer();
    }
  }

  public static RealMatrix inverse(final RealMatrix matrix) {
    try {
      return MatrixUtils.inverse(matrix);
    } catch (Throwable e1) {
      logger.info("Error inverting kernel", e1);
      return new LUDecomposition(matrix).getSolver().getInverse();
    }
  }

  public static void setMatrix(final Tensor tensor, final RealMatrix matrix) {
    tensor.setByCoord(c -> {
      int b = c.getCoords()[2];
      int i = b % 3;
      int o = b / 3;
      return matrix.getEntry(i, o);
    });
  }

  @Nonnull
  public static RealMatrix getMatrix(final Tensor tensor) {
    RealMatrix matrix = new BlockRealMatrix(3, 3);
    tensor.forEach((v, c) -> {
      int b = c.getCoords()[2];
      int i = b % 3;
      int o = b / 3;
      matrix.setEntry(i, o, v);
    }, false);
    return matrix;
  }

  @Nonnull
  public static SimpleConvolutionLayer unitTransformer() {
    return unitTransformer(3);
  }

  @Nonnull
  public static SimpleConvolutionLayer unitTransformer(final int bands) {
    SimpleConvolutionLayer colorForwardTransform = new SimpleConvolutionLayer(1, 1, bands * bands);
    colorForwardTransform.kernel.setByCoord(c -> {
      int band = c.getCoords()[2];
      int i = band % bands;
      int o = band / bands;
      return i == o ? 1.0 : 0.0;
    });
    return colorForwardTransform;
  }

  public static int[][] getIndexMap(final SimpleConvolutionLayer layer) {
    int[] kernelDimensions = layer.getKernelDimensions();
    double b = Math.sqrt(kernelDimensions[2]);
    int h = kernelDimensions[1];
    int w = kernelDimensions[0];
    int l = (int) (w * h * b);
    return IntStream.range(0, (int) b).mapToObj(i -> {
      return IntStream.range(0, l).map(j -> j + l * i).toArray();
    }).toArray(i -> new int[i][]);
  }

  public Tensor transfer(
      final Tensor canvasImage,
      final StyleSetup<T> styleParameters,
      final int trainingMinutes,
      final NeuralSetup measureStyle
  ) {
    return transfer(new NullNotebookOutput(), canvasImage, styleParameters, trainingMinutes, measureStyle, 50, true);
  }

  public Tensor transfer(
      @Nonnull final NotebookOutput log,
      final Tensor canvasImage,
      final StyleSetup<T> styleParameters,
      final int trainingMinutes,
      final NeuralSetup measureStyle,
      final int maxIterations,
      final boolean verbose
  ) {
    canvasImage.assertAlive();
    try {
      log.p("Input Content:");
      log.p(log.png(canvasImage.toImage(), "Input Content"));
      log.p("Style Content:");
      styleParameters.styleImages.forEach((file, styleImage) -> {
        log.p(log.png(styleImage.toImage(), file));
      });
      System.gc();
      if (verbose) {
        log.p("Input Parameters:");
        log.eval(() -> {
          return ArtistryUtil.toJson(styleParameters);
        });
      }
      this.setColorForwardTransform(train(log, styleParameters, trainingMinutes, measureStyle, maxIterations, verbose, canvasImage));
      Tensor result = forwardTransform(canvasImage);
      log.p("Result:");
      log.p(log.png(result.toImage(), "Output Canvas"));
      return canvasImage.set(result);
    } catch (Throwable e) {
      logger.warn("Error in color transfer", e);
      return canvasImage;
    }
  }

  @Nonnull
  public Tensor forwardTransform(final Tensor canvas) {
    Layer fwdTransform = getFwdTransform();
    if (null == fwdTransform) fwdTransform = unitTransformer();
    Tensor andFree = fwdTransform.eval(canvas).getDataAndFree().getAndFree(0);
    fwdTransform.freeRef();
    return andFree;
  }

  @Nonnull
  public Tensor inverseTransform(final Tensor canvas) {
    Layer invTransform = getInvTransform();
    Tensor andFree = invTransform.eval(canvas).getDataAndFree().getAndFree(0);
    invTransform.freeRef();
    return andFree;
  }

  @Nonnull
  public Layer getFwdTransform() {
    SimpleConvolutionLayer colorForwardTransform = getColorForwardTransform();
    if (null == colorForwardTransform) return unitTransformer();
    return PipelineNetwork.wrap(
        1,
        colorForwardTransform,
        ArtistryUtil.getClamp(255)
    );
  }

  @Nonnull
  public Layer getInvTransform() {
    PipelineNetwork network = new PipelineNetwork(1);
    SimpleConvolutionLayer colorForwardTransform = getColorForwardTransform();
    if (null == colorForwardTransform) return unitTransformer();
    network.wrap(
        ArtistryUtil.getClamp(255),
        network.wrap(
            invert(colorForwardTransform),
            network.getInput(0)
        )
    ).freeRef();
    return network;
  }

  @Nonnull
  public SimpleConvolutionLayer train(
      @Nonnull final NotebookOutput log,
      final StyleSetup<T> styleParameters,
      final int trainingMinutes,
      final NeuralSetup measureStyle,
      final int maxIterations,
      final boolean verbose,
      final Tensor canvas
  ) {
    NotebookOutput trainingLog = verbose ? log : new NullNotebookOutput();
    SimpleConvolutionLayer colorForwardTransform = unitTransformer();
    PipelineNetwork trainingAssembly = getNetwork(log, styleParameters, measureStyle, colorForwardTransform);
    Trainable trainable = getTrainable(canvas, trainingAssembly);
    trainingAssembly.freeRef();
    try {
      @Nonnull ArrayList<StepRecord> history = new ArrayList<>();
      String training_name = String.format("etc/training_%s.png", Long.toHexString(MarkdownNotebookOutput.random.nextLong()));
      log.p(String.format("<a href=\"%s\"><img src=\"%s\"></a>", training_name, training_name));
      Closeable png = log.getHttpd().addGET(training_name, "image/png", r -> {
        try {
          BufferedImage im = Util.toImage(TestUtil.plot(history));
          if (null != im) ImageIO.write(im, "png", r);
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
      try {
        trainingLog.eval(() -> {
          new IterativeTrainer(trainable)
              .setMonitor(TestUtil.getMonitor(history))
              .setOrientation(getOrientation())
              .setMaxIterations(maxIterations)
              .setIterationsPerSample(100)
              .setLineSearchFactory(name -> new QuadraticSearch().setRelativeTolerance(1e-1).setCurrentRate(1e0))
              .setTimeout(trainingMinutes, TimeUnit.MINUTES)
              .setTerminateThreshold(Double.NEGATIVE_INFINITY)
              .runAndFree();
          return TestUtil.plot(history);
        });
      } finally {
        try {
          png.close();
          BufferedImage image = Util.toImage(TestUtil.plot(history));
          if (null != image) ImageIO.write(image, "png", log.file(training_name));
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }
    } finally {
      trainable.freeRef();
    }
    return colorForwardTransform;
  }

  @Nonnull
  public final Trainable getTrainable(final Tensor canvas, final PipelineNetwork trainingAssembly) {
    return new ArrayTrainable(trainingAssembly, 1).setVerbose(true).setMask(false).setData(Arrays.asList(new Tensor[][]{{canvas}}));
  }

  @Nonnull
  public PipelineNetwork getNetwork(
      @Nonnull final NotebookOutput log,
      final StyleSetup<T> styleParameters,
      final NeuralSetup measureStyle,
      final SimpleConvolutionLayer colorForwardTransform
  ) {
    PipelineNetwork network = fitnessNetwork(measureStyle);
    network.setFrozen(true);
    TestUtil.instrumentPerformance(network);
    final FileHTTPD server = log.getHttpd();
    if (null != server) ArtistryUtil.addLayersHandler(network, server);
    PipelineNetwork trainingAssembly = new PipelineNetwork(1);
    trainingAssembly.wrap(
        network,
        trainingAssembly.wrap(
            ArtistryUtil.getClamp(255),
            trainingAssembly.add(
                colorForwardTransform,
                trainingAssembly.getInput(0)
            )
        )
    ).freeRef();
    MultiPrecision.setPrecision(trainingAssembly, styleParameters.precision);
    return trainingAssembly;
  }

  @Nonnull
  public OrientationStrategy<LineSearchCursor> getOrientation() {
    return new TrustRegionStrategy(new LBFGS()) {
      @Override
      public TrustRegion getRegionPolicy(final Layer layer) {
        if (layer instanceof SimpleConvolutionLayer) {
          return new OrthonormalConstraint(getIndexMap((SimpleConvolutionLayer) layer)).setOrtho(isOrtho()).setUnit(isUnit());
        }
        return null;
      }
    };
  }

  @Nonnull
  public ArrayList<Tuple2<Double, DAGNode>> getStyleComponents(
      final DAGNode node,
      final PipelineNetwork network,
      final LayerStyleParams styleParams,
      final Tensor mean,
      final Tensor covariance,
      final CenteringMode centeringMode
  ) {
    ArrayList<Tuple2<Double, DAGNode>> styleComponents = new ArrayList<>();
    if (null != styleParams && (styleParams.cov != 0 || styleParams.mean != 0 || styleParams.enhance != 0)) {
      InnerNode negTarget = null == mean ? null : network.wrap(new ValueLayer(mean.scale(-1)), new DAGNode[]{});
      node.addRef();
      InnerNode negAvg = network.wrap(new BandAvgReducerLayer().setAlpha(-1), node);
      if (styleParams.enhance != 0 || styleParams.cov != 0) {
        DAGNode recentered;
        switch (centeringMode) {
          case Origin:
            recentered = node;
            break;
          case Dynamic:
            negAvg.addRef();
            recentered = network.wrap(new GateBiasLayer(), node, negAvg);
            break;
          case Static:
            negTarget.addRef();
            recentered = network.wrap(new GateBiasLayer(), node, negTarget);
            break;
          default:
            throw new RuntimeException();
        }
        double covRms = null == covariance ? 1.0 : covariance.rms();
        if (styleParams.enhance != 0) {
          recentered.addRef();
          styleComponents.add(new Tuple2<>(
              -(0 == covRms ? styleParams.enhance : styleParams.enhance / covRms),
              network.wrap(
                  new AvgReducerLayer(),
                  network.wrap(
                      new SquareActivationLayer(),
                      recentered
                  )
              )
          ));
        }
        if (styleParams.cov != 0) {
          int[] covDim = covariance.getDimensions();
          assert 0 < covDim[2] : Arrays.toString(covDim);
          int inputBands = mean.getDimensions()[2];
          assert 0 < inputBands : Arrays.toString(mean.getDimensions());
          int outputBands = covDim[2] / inputBands;
          assert 0 < outputBands : Arrays.toString(covDim) + " / " + inputBands;
          double covScale = 0 == covRms ? 1 : 1.0 / covRms;
          recentered.addRef();
          styleComponents.add(new Tuple2<>(styleParams.cov, network.wrap(
              new MeanSqLossLayer().setAlpha(covScale),
              network.wrap(new ValueLayer(covariance), new DAGNode[]{}),
              network.wrap(new GramianLayer(), recentered)
          )
          ));
        }
        recentered.freeRef();
      } else {
        node.freeRef();
      }
      if (styleParams.mean != 0) {
        double meanRms = mean.rms();
        double meanScale = 0 == meanRms ? 1 : 1.0 / meanRms;
        styleComponents.add(new Tuple2<>(
            styleParams.mean,
            network.wrap(new MeanSqLossLayer().setAlpha(meanScale), negAvg, negTarget)
        ));
      } else {
        if (null != negTarget) negTarget.freeRef();
        if (null != negAvg) negAvg.freeRef();
      }
    } else {
      node.freeRef();
    }
    return styleComponents;
  }

  public NeuralSetup measureStyle(final StyleSetup<T> style) {
    NeuralSetup self = new NeuralSetup(style);
    List<CharSequence> keyList = style.styleImages.keySet().stream().collect(Collectors.toList());
    Tensor contentInput = style.contentImage;
    List<Tensor> styleInputs = keyList.stream().map(x -> style.styleImages.get(x)).collect(Collectors.toList());
    IntStream.range(0, keyList.size()).forEach(i -> {
      self.styleTargets.put(keyList.get(i), new StyleTarget<>());
    });
    self.contentTarget = new ContentTarget<>();
    for (final T layerType : getLayerTypes()) {
      System.gc();
      Layer network = layerType.network();
      try {
        MultiPrecision.setPrecision((DAGNetwork) network, style.precision);
        //network = new ImgTileSubnetLayer(network, 400,400,400,400);
        Tensor content = null == contentInput ? null : network.eval(contentInput).getDataAndFree().getAndFree(0);
        if (null != content) {
          self.contentTarget.content.put(layerType, content);
          logger.info(String.format("%s : target content = %s", layerType.name(), content.prettyPrint()));
          logger.info(String.format(
              "%s : content statistics = %s",
              layerType.name(),
              JsonUtil.toJson(new ScalarStatistics().add(content.getData()).getMetrics())
          ));
        }
        for (int i = 0; i < styleInputs.size(); i++) {
          Tensor styleInput = styleInputs.get(i);
          CharSequence key = keyList.get(i);
          StyleTarget<T> styleTarget = self.styleTargets.get(key);
          if (0 == self.style.styles.entrySet().stream().filter(e1 -> e1.getKey().contains(key)).map(x -> x.getValue().params.get(
              layerType)).filter(x -> null != x).filter(x -> x.mean != 0 || x.cov != 0).count())
            continue;
          System.gc();
          Layer wrapAvg = ArtistryUtil.wrapTiledAvg(network.copy(), 400);
          styleInput.assertAlive();
          Tensor mean = wrapAvg.eval(styleInput).getDataAndFree().getAndFree(0);
          wrapAvg.freeRef();
          styleTarget.mean.put(layerType, mean);
          logger.info(String.format("%s : style mean = %s", layerType.name(), mean.prettyPrint()));
          logger.info(String.format(
              "%s : mean statistics = %s",
              layerType.name(),
              JsonUtil.toJson(new ScalarStatistics().add(mean.getData()).getMetrics())
          ));
          if (0 == self.style.styles.entrySet().stream().filter(e1 -> e1.getKey().contains(key)).map(x -> x.getValue().params.get(
              layerType)).filter(x -> null != x).filter(x -> x.cov != 0).count())
            continue;
          System.gc();
          Layer gram = ArtistryUtil.wrapTiledAvg(ArtistryUtil.gram(network.copy()), 400);
          Tensor cov0 = gram.eval(styleInput).getDataAndFree().getAndFree(0);
          gram.freeRef();
          gram = ArtistryUtil.wrapTiledAvg(ArtistryUtil.gram(network.copy(), mean), 400);
          Tensor cov1 = gram.eval(styleInput).getDataAndFree().getAndFree(0);
          gram.freeRef();
          styleTarget.cov0.put(layerType, cov0);
          styleTarget.cov1.put(layerType, cov1);
          int featureBands = mean.getDimensions()[2];
          int covarianceElements = cov1.getDimensions()[2];
          int selectedBands = covarianceElements / featureBands;
          logger.info(String.format(
              "%s : target cov0 = %s",
              layerType.name(),
              cov0.reshapeCast(featureBands, selectedBands, 1).prettyPrintAndFree()
          ));
          logger.info(String.format(
              "%s : cov0 statistics = %s",
              layerType.name(),
              JsonUtil.toJson(new ScalarStatistics().add(cov0.getData()).getMetrics())
          ));
          logger.info(String.format(
              "%s : target cov1 = %s",
              layerType.name(),
              cov1.reshapeCast(featureBands, selectedBands, 1).prettyPrintAndFree()
          ));
          logger.info(String.format(
              "%s : cov1 statistics = %s",
              layerType.name(),
              JsonUtil.toJson(new ScalarStatistics().add(cov1.getData()).getMetrics())
          ));
        }
      } finally {
        network.freeRef();
      }
    }
    return self;
  }

  @Nonnull
  public List<Tuple2<Double, DAGNode>> getFitnessComponents(NeuralSetup setup, final Map<T, DAGNode> nodeMap) {
    List<Tuple2<Double, DAGNode>> functions = new ArrayList<>();
    functions.addAll(getContentComponents(setup, nodeMap));
    functions.addAll(getStyleComponents(setup, nodeMap));
    return functions;
  }

  @Nonnull
  public ArrayList<Tuple2<Double, DAGNode>> getStyleComponents(NeuralSetup setup, final Map<T, DAGNode> nodeMap) {
    ArrayList<Tuple2<Double, DAGNode>> styleComponents = new ArrayList<>();
    for (final List<CharSequence> keys : setup.style.styles.keySet()) {
      StyleTarget<T> styleTarget = keys.stream().map(x -> {
        StyleTarget<T> obj = setup.styleTargets.get(x);
        obj.addRef();
        return obj;
      }).reduce((a, b) -> {
        StyleTarget<T> r = a.add(b);
        a.freeRef();
        b.freeRef();
        return r;
      }).map(x -> {
        StyleTarget<T> r = x.scale(1.0 / keys.size());
        x.freeRef();
        return r;
      }).orElse(null);
      for (final T layerType : getLayerTypes()) {
        StyleCoefficients<T> styleCoefficients = setup.style.styles.get(keys);
        assert null != styleCoefficients;
        final DAGNode node = nodeMap.get(layerType);
        final PipelineNetwork network = (PipelineNetwork) node.getNetwork();
        LayerStyleParams styleParams = styleCoefficients.params.get(layerType);
        Tensor mean = null == styleTarget ? null : styleTarget.mean.get(layerType);
        Tensor covariance;
        switch (styleCoefficients.centeringMode) {
          case Origin:
            covariance = null == styleTarget ? null : styleTarget.cov0.get(layerType);
            break;
          case Dynamic:
          case Static:
            covariance = null == styleTarget ? null : styleTarget.cov1.get(layerType);
            break;
          default:
            throw new RuntimeException();
        }
        node.addRef();
        styleComponents.addAll(getStyleComponents(node, network, styleParams, mean, covariance, styleCoefficients.centeringMode));
      }
      if (null != styleTarget) styleTarget.freeRef();

    }
    return styleComponents;
  }

  @Nonnull
  public PipelineNetwork fitnessNetwork(NeuralSetup setup) {
    PipelineNetwork pipelineNetwork = getInstance().getNetwork();
    Map<T, DAGNode> nodes = new HashMap<>();
    Map<T, UUID> ids = getInstance().getNodes();
    ids.forEach((l, id) -> nodes.put(l, pipelineNetwork.getChildNode(id)));
    PipelineNetwork network = buildNetwork(setup, nodes, pipelineNetwork);
    MultiPrecision.setPrecision(network, setup.style.precision);
    return network;
  }

  @Nonnull
  public abstract T[] getLayerTypes();

  @Nonnull
  public ArrayList<Tuple2<Double, DAGNode>> getContentComponents(NeuralSetup setup, final Map<T, DAGNode> nodeMap) {
    ArrayList<Tuple2<Double, DAGNode>> contentComponents = new ArrayList<>();
    for (final T layerType : getLayerTypes()) {
      final DAGNode node = nodeMap.get(layerType);
      if (!setup.style.content.params.containsKey(layerType)) continue;
      final double coeff_content = setup.style.content.params.get(layerType);
      if (coeff_content != 0) {
        Tensor content = setup.contentTarget.content.get(layerType);
        if (content != null) {
          final PipelineNetwork network = (PipelineNetwork) node.getNetwork();
          assert network != null;
          InnerNode innerNode = network.wrap(new MeanSqLossLayer().setAlpha(1.0 / content.rms()),
              node, network.wrap(new ValueLayer(content), new DAGNode[]{})
          );
          contentComponents.add(new Tuple2<>(coeff_content, innerNode));
        }
      }
    }
    return contentComponents;
  }

  public abstract U getInstance();

  public PipelineNetwork buildNetwork(NeuralSetup setup, final Map<T, DAGNode> nodeMap, final PipelineNetwork network) {
    List<Tuple2<Double, DAGNode>> functions = getFitnessComponents(setup, nodeMap);
    ArtistryUtil.reduce(network, functions, parallelLossFunctions);
    return network;
  }

  public SimpleConvolutionLayer getColorForwardTransform() {
    SimpleConvolutionLayer colorForwardTransform = this.colorForwardTransform;
    if (null != colorForwardTransform) colorForwardTransform = (SimpleConvolutionLayer) colorForwardTransform.copy();
    return colorForwardTransform;
  }

  public synchronized void setColorForwardTransform(SimpleConvolutionLayer colorForwardTransform) {
    colorForwardTransform.assertAlive();
    if (null != this.colorForwardTransform) this.colorForwardTransform.freeRef();
    this.colorForwardTransform = colorForwardTransform;
    if (null != this.colorForwardTransform) this.colorForwardTransform.addRef();
  }

  public boolean isOrtho() {
    return ortho;
  }

  public ColorTransfer<T, U> setOrtho(boolean ortho) {
    this.ortho = ortho;
    return this;
  }

  public boolean isUnit() {
    return unit;
  }

  public ColorTransfer<T, U> setUnit(boolean unit) {
    this.unit = unit;
    return this;
  }

  public enum CenteringMode {
    Dynamic,
    Static,
    Origin
  }

  public static class VGG16 extends ColorTransfer<CVPipe_VGG16.Layer, CVPipe_VGG16> {

    public CVPipe_VGG16 getInstance() {
      return CVPipe_VGG16.INSTANCE;
    }

    @Nonnull
    public CVPipe_VGG16.Layer[] getLayerTypes() {
      return CVPipe_VGG16.Layer.values();
    }

  }

  public static class VGG19 extends ColorTransfer<CVPipe_VGG19.Layer, CVPipe_VGG19> {

    public CVPipe_VGG19 getInstance() {
      return CVPipe_VGG19.INSTANCE;
    }

    @Nonnull
    public CVPipe_VGG19.Layer[] getLayerTypes() {
      return CVPipe_VGG19.Layer.values();
    }

  }

  public static class Inception extends ColorTransfer<CVPipe_Inception.Strata, CVPipe_Inception> {

    public CVPipe_Inception getInstance() {
      return CVPipe_Inception.INSTANCE;
    }

    @Nonnull
    public CVPipe_Inception.Strata[] getLayerTypes() {
      return CVPipe_Inception.Strata.values();
    }

  }

  public static class ContentCoefficients<T extends LayerEnum<T>> {
    public final Map<T, Double> params = new HashMap<>();

    public ContentCoefficients set(final T l, final double v) {
      params.put(l, v);
      return this;
    }

  }

  public static class LayerStyleParams {
    public final double mean;
    public final double cov;
    private final double enhance;

    public LayerStyleParams(final double mean, final double cov, final double enhance) {
      this.mean = mean;
      this.cov = cov;
      this.enhance = enhance;
    }
  }

  public static class StyleSetup<T extends LayerEnum<T>> {
    public final Precision precision;
    public final transient Map<CharSequence, Tensor> styleImages;
    public final Map<List<CharSequence>, StyleCoefficients<T>> styles;
    public final ContentCoefficients<T> content;
    public transient Tensor contentImage;


    public StyleSetup(
        final Precision precision,
        final Tensor contentImage,
        ContentCoefficients<T> contentCoefficients,
        final Map<CharSequence, Tensor> styleImages,
        final Map<List<CharSequence>, StyleCoefficients<T>> styles
    ) {
      this.precision = precision;
      this.contentImage = contentImage;

      this.styleImages = styleImages;
      if (!styleImages.values().stream().allMatch(x -> x instanceof Tensor)) throw new RuntimeException();
      this.styles = styles;
      this.content = contentCoefficients;
    }

  }

  public static class StyleCoefficients<T extends LayerEnum<T>> {
    public final CenteringMode centeringMode;
    public final Map<T, LayerStyleParams> params = new HashMap<>();


    public StyleCoefficients(final CenteringMode centeringMode) {
      this.centeringMode = centeringMode;
    }

    public StyleCoefficients set(final T layerType, final double coeff_style_mean, final double coeff_style_cov) {
      return set(
          layerType,
          coeff_style_mean,
          coeff_style_cov,
          0.0
      );
    }

    public StyleCoefficients set(final T layerType, final double coeff_style_mean, final double coeff_style_cov, final double dream) {
      params.put(layerType, new LayerStyleParams(coeff_style_mean, coeff_style_cov, dream));
      return this;
    }

  }

  public static class ContentTarget<T extends LayerEnum<T>> {
    public Map<T, Tensor> content = new HashMap<>();
  }

  public class StyleTarget<T extends LayerEnum<T>> extends ReferenceCountingBase {
    public Map<T, Tensor> cov0 = new HashMap<>();
    public Map<T, Tensor> cov1 = new HashMap<>();
    public Map<T, Tensor> mean = new HashMap<>();

    @Override
    protected void _free() {
      super._free();
      if (null != cov0) cov0.values().forEach(ReferenceCountingBase::freeRef);
      if (null != cov1) cov1.values().forEach(ReferenceCountingBase::freeRef);
      if (null != mean) mean.values().forEach(ReferenceCountingBase::freeRef);
    }

    public StyleTarget<T> add(StyleTarget<T> right) {
      StyleTarget<T> newStyle = new StyleTarget<>();
      Stream.concat(mean.keySet().stream(), right.mean.keySet().stream()).distinct().forEach(layer -> {
        Tensor l = mean.get(layer);
        Tensor r = right.mean.get(layer);
        if (l != null && l != r) {
          Tensor add = l.add(r);
          newStyle.mean.put(layer, add);
        } else if (l != null) {
          l.addRef();
          newStyle.mean.put(layer, l);
        } else if (r != null) {
          r.addRef();
          newStyle.mean.put(layer, r);
        }
      });
      Stream.concat(cov0.keySet().stream(), right.cov0.keySet().stream()).distinct().forEach(layer -> {
        Tensor l = cov0.get(layer);
        Tensor r = right.cov0.get(layer);
        if (l != null && l != r) {
          Tensor add = l.add(r);
          newStyle.cov0.put(layer, add);
        } else if (l != null) {
          l.addRef();
          newStyle.cov0.put(layer, l);
        } else if (r != null) {
          r.addRef();
          newStyle.cov0.put(layer, r);
        }
      });
      Stream.concat(cov1.keySet().stream(), right.cov1.keySet().stream()).distinct().forEach(layer -> {
        Tensor l = cov1.get(layer);
        Tensor r = right.cov1.get(layer);
        if (l != null && l != r) {
          Tensor add = l.add(r);
          newStyle.cov1.put(layer, add);
        } else if (l != null) {
          l.addRef();
          newStyle.cov1.put(layer, l);
        } else if (r != null) {
          r.addRef();
          newStyle.cov1.put(layer, r);
        }
      });
      return newStyle;
    }

    public StyleTarget<T> scale(double value) {
      StyleTarget<T> newStyle = new StyleTarget<>();
      mean.keySet().stream().distinct().forEach(layer -> {
        newStyle.mean.put(layer, mean.get(layer).scale(value));
      });
      cov0.keySet().stream().distinct().forEach(layer -> {
        newStyle.cov0.put(layer, cov0.get(layer).scale(value));
      });
      cov1.keySet().stream().distinct().forEach(layer -> {
        newStyle.cov1.put(layer, cov1.get(layer).scale(value));
      });
      return newStyle;
    }

  }

  public class NeuralSetup {

    public final StyleSetup<T> style;
    public ContentTarget<T> contentTarget = new ContentTarget<>();
    public Map<CharSequence, StyleTarget<T>> styleTargets = new HashMap<>();


    public NeuralSetup(final StyleSetup<T> style) {
      this.style = style;
    }
  }

}
