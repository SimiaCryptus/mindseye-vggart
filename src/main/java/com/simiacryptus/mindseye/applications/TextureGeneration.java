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
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.models.CVPipe;
import com.simiacryptus.mindseye.models.CVPipe_VGG16;
import com.simiacryptus.mindseye.models.CVPipe_VGG19;
import com.simiacryptus.mindseye.models.LayerEnum;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.line.BisectionSearch;
import com.simiacryptus.mindseye.opt.orient.TrustRegionStrategy;
import com.simiacryptus.mindseye.opt.region.RangeConstraint;
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.Closeable;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * This notebook implements the Style Transfer protocol outlined in <a href="https://arxiv.org/abs/1508.06576">A Neural Algorithm of Artistic Style</a>
 *
 * @param <T> the type parameter
 * @param <U> the type parameter
 */
public abstract class TextureGeneration<T extends LayerEnum<T>, U extends CVPipe<T>> {

  private static final Logger logger = LoggerFactory.getLogger(TextureGeneration.class);
  /**
   * The Parallel loss functions.
   */
  public boolean parallelLossFunctions = true;
  private int tiling = 0;

  /**
   * Instantiates a new Texture generation.
   */
  public TextureGeneration() {
    tiling = 3;
  }

  /**
   * Generate buffered png.
   *
   * @param log             the log
   * @param styleTransfer   the style transfer
   * @param precision       the precision
   * @param imageSize       the png size
   * @param growthFactor    the growth factor
   * @param styles          the styles
   * @param trainingMinutes the training minutes
   * @param canvasImage     the canvas png
   * @param phases          the phases
   * @param maxIterations   the max iterations
   * @param styleSize       the style size
   * @return the buffered png
   */
  public static Tensor optimize(
      @Nonnull final NotebookOutput log,
      final VGG19 styleTransfer,
      final Precision precision,
      int imageSize,
      final double growthFactor,
      final Map<List<CharSequence>, StyleCoefficients<CVPipe_VGG19.Layer>> styles,
      final int trainingMinutes,
      Tensor canvasImage,
      final int phases,
      final int maxIterations,
      final int styleSize
  ) {
    Map<CharSequence, BufferedImage> styleImages = new HashMap<>();
    StyleSetup<CVPipe_VGG19.Layer> styleSetup;
    NeuralSetup measureStyle;

    styleImages.clear();
    if (0 < styleSize) {
      styleImages.putAll(styles.keySet().stream().flatMap(Collection::stream).collect(Collectors.toMap(
          x -> x,
          image -> ArtistryUtil.load(
              image,
              styleSize
          )
      )));
    } else {
      styleImages.putAll(styles.keySet().stream().flatMap(Collection::stream).collect(Collectors.toMap(x -> x, image -> ArtistryUtil.load(image))));
    }
    styleSetup = new StyleSetup<>(precision, styleImages, styles);
    measureStyle = styleTransfer.measureStyle(styleSetup);

    final AtomicReference<Double> currentStyleSize = new AtomicReference<Double>((double) styleSize);
    canvasImage = Tensor.fromRGB(TestUtil.resize(canvasImage.toImage(), imageSize, true));
    canvasImage = styleTransfer.optimize(log, measureStyle, canvasImage, trainingMinutes, maxIterations, true, styleSetup.precision);
    for (int i = 1; i < phases; i++) {
      imageSize *= growthFactor;
      currentStyleSize.updateAndGet(v -> v * growthFactor);

      styleImages.clear();
      if (0 < currentStyleSize.get()) {
        styleImages.putAll(styles.keySet().stream().flatMap(Collection::stream).collect(Collectors.toMap(
            x -> x,
            image -> ArtistryUtil.load(
                image,
                (int) (double) currentStyleSize.get()
            )
        )));
      } else {
        styleImages.putAll(styles.keySet().stream().flatMap(Collection::stream).collect(Collectors.toMap(x -> x, image -> ArtistryUtil.load(image))));
      }
      styleSetup = new StyleSetup<>(precision, styleImages, styles);
      measureStyle = styleTransfer.measureStyle(styleSetup);

      canvasImage = Tensor.fromRGB(TestUtil.resize(canvasImage.toImage(), imageSize, true));
      canvasImage = styleTransfer.optimize(log, measureStyle, canvasImage, trainingMinutes, maxIterations, true, styleSetup.precision);
    }
    return canvasImage;
  }

  /**
   * Init canvas buffered png.
   *
   * @param imageSize the png size
   * @return the buffered png
   */
  @Nonnull
  public static BufferedImage initCanvas(final AtomicInteger imageSize) {
    return ArtistryUtil.paint_Plasma(3, 100.0, 1.4, imageSize.get()).toImage();
  }

  public static Tensor optimize(
      @Nonnull final NotebookOutput log,
      final PipelineNetwork network,
      final Tensor workingImage,
      final int trainingMinutes,
      final int maxIterations,
      final boolean verbose,
      final Precision precision,
      final int tiling
  ) {
    return ArtistryUtil.logExceptionWithDefault(log, () -> {
      System.gc();
      TestUtil.monitorImage(workingImage, false, false);
      String imageName = String.format("etc/image_%s.jpg", Long.toHexString(MarkdownNotebookOutput.random.nextLong()));
      log.p(String.format("<a href=\"%s\"><img src=\"%s\"></a>", imageName, imageName));
      Closeable jpeg = log.getHttpd().addGET(imageName, "image/jpeg", outputStream -> {
        try {
          ImageIO.write(workingImage.toImage(), "jpeg", outputStream);
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
      PipelineNetwork frozen = (PipelineNetwork) network.setFrozen(true);
      TestUtil.instrumentPerformance(frozen);
      final FileHTTPD server = log.getHttpd();
      if (null != server) ArtistryUtil.addLayersHandler(frozen, server);
      if (tiling > 1) frozen = ArtistryUtil.tileCycle(frozen, tiling);
      try {
        PipelineNetwork finalFrozen = frozen;
        log.subreport("_training_" + UUID.randomUUID().toString(), sublog -> {
          train(sublog, workingImage, finalFrozen, trainingMinutes, maxIterations, precision);
          return null;
        });
      } finally {
        try {
          jpeg.close();
          ImageIO.write(workingImage.toImage(), "jpeg", log.file(imageName));
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }
      return workingImage;
    }, workingImage);
  }

  /**
   * Train.
   *  @param log             the log
   * @param canvas          the canvas
   * @param network         the network
   * @param trainingMinutes the training minutes
   * @param maxIterations   the max iterations
   * @param precision
   */
  public static void train(
      @Nonnull final NotebookOutput log,
      final Tensor canvas,
      final PipelineNetwork network,
      final int trainingMinutes,
      final int maxIterations,
      Precision precision) {
    ArtistryUtil.setPrecision(network, precision);
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
    log.run(() -> {
      Trainable trainable = getTrainable(network, canvas);
      new IterativeTrainer(trainable)
          .setMonitor(TestUtil.getMonitor(history))
          .setOrientation(new TrustRegionStrategy() {
            @Override
            public TrustRegion getRegionPolicy(final Layer layer) {
              return new RangeConstraint().setMin(1e-4).setMax(256);
            }
          })
          .setMaxIterations(maxIterations)
          .setIterationsPerSample(100)
          .setLineSearchFactory(name -> new BisectionSearch().setSpanTol(1e-1).setCurrentRate(1e6))
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
    });
  }

  /**
   * Gets trainable.
   *
   * @param network the network
   * @param canvas  the canvas
   * @return the trainable
   */
  @Nonnull
  public static Trainable getTrainable(final PipelineNetwork network, final Tensor canvas) {
    return new ArrayTrainable(network, 1).setVerbose(true).setMask(true).setData(Arrays.asList(new Tensor[][]{{canvas}}));
  }

  /**
   * Style transfer buffered png.
   *
   * @param canvasImage     the canvas png
   * @param styleParameters the style parameters
   * @param trainingMinutes the training minutes
   * @param measureStyle    the measureStyle style
   * @return the buffered png
   */
  public Tensor optimize(
      final Tensor canvasImage,
      final StyleSetup<T> styleParameters,
      final int trainingMinutes,
      final NeuralSetup measureStyle
  ) {
    final NotebookOutput log = new NullNotebookOutput();
    log.p("Input Parameters:");
    log.eval(() -> {
      return ArtistryUtil.toJson(styleParameters);
    });
    Precision precision = styleParameters.precision;
    return optimize(log, measureStyle, canvasImage, trainingMinutes, 50, true, precision);
  }

  public Tensor optimize(
      @Nonnull final NotebookOutput log,
      final NeuralSetup measureStyle, final Tensor canvasImage,
      final int trainingMinutes,
      final int maxIterations,
      final boolean verbose,
      final Precision precision
  ) {
    return optimize(log, fitnessNetwork(measureStyle), canvasImage, trainingMinutes, maxIterations, verbose, precision, tiling);
  }

  /**
   * Gets style components.
   *
   * @param node          the node
   * @param network       the network
   * @param styleParams   the style params
   * @param mean          the mean
   * @param covariance    the covariance
   * @param centeringMode the centering mode
   * @return the style components
   */
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
    if (null != styleParams && (styleParams.cov != 0 || styleParams.mean != 0)) {
      double meanRms = mean.rms();
      double meanScale = 0 == meanRms ? 1 : (1.0 / meanRms);
      InnerNode negTarget = network.wrap(new ValueLayer(mean.scale(-1)), new DAGNode[]{});
      InnerNode negAvg = network.wrap(new BandAvgReducerLayer().setAlpha(-1), node);
      if (styleParams.enhance != 0 || styleParams.cov != 0) {
        DAGNode recentered;
        switch (centeringMode) {
          case Origin:
            recentered = node;
            break;
          case Dynamic:
            recentered = network.wrap(new GateBiasLayer(), node, negAvg);
            break;
          case Static:
            recentered = network.wrap(new GateBiasLayer(), node, negTarget);
            break;
          default:
            throw new RuntimeException();
        }
        int[] covDim = covariance.getDimensions();
        double covRms = covariance.rms();
        if (styleParams.enhance != 0) {
          styleComponents.add(new Tuple2<>(-(0 == covRms ? styleParams.enhance : (styleParams.enhance / covRms)), network.wrap(
              new AvgReducerLayer(),
              network.wrap(new SquareActivationLayer(), recentered)
          )));
        }
        if (styleParams.cov != 0) {
          assert 0 < covDim[2] : Arrays.toString(covDim);
          int inputBands = mean.getDimensions()[2];
          assert 0 < inputBands : Arrays.toString(mean.getDimensions());
          int outputBands = covDim[2] / inputBands;
          assert 0 < outputBands : Arrays.toString(covDim) + " / " + inputBands;
          double covScale = 0 == covRms ? 1 : (1.0 / covRms);
          styleComponents.add(new Tuple2<>(styleParams.cov, network.wrap(
              new MeanSqLossLayer().setAlpha(covScale),
              network.wrap(new ValueLayer(covariance), new DAGNode[]{}),
              network.wrap(new GramianLayer(), recentered)
          )
          ));
        }
      }
      if (styleParams.mean != 0) {
        styleComponents.add(new Tuple2<>(
            styleParams.mean,
            network.wrap(new MeanSqLossLayer().setAlpha(meanScale), negAvg, negTarget)
        ));
      }
    }
    return styleComponents;
  }

  /**
   * Measure style neural setup.
   *
   * @param style the style
   * @return the neural setup
   */
  public NeuralSetup<T> measureStyle(final StyleSetup<T> style) {
    NeuralSetup<T> self = new NeuralSetup<>(style);
    List<CharSequence> keyList = style.styleImages.keySet().stream().collect(Collectors.toList());
    List<Tensor> styleInputs = keyList.stream().map(x -> style.styleImages.get(x)).map(img -> Tensor.fromRGB(img)).collect(Collectors.toList());
    IntStream.range(0, keyList.size()).forEach(i -> {
      self.styleTargets.put(keyList.get(i), new StyleTarget<>());
    });
    for (final T layerType : getLayerTypes()) {
      System.gc();
      final PipelineNetwork network = layerType.network();
      ArtistryUtil.setPrecision(network, style.precision);
      for (int i = 0; i < styleInputs.size(); i++) {
        Tensor styleInput = styleInputs.get(i);
        CharSequence key = keyList.get(i);
        if (0 == self.style.styles.entrySet().stream().filter(e1 -> e1.getKey().contains(key)).map(x -> x.getValue().params.get(
            layerType)).filter(x -> null != x).filter(x -> x.mean != 0 || x.cov != 0).count())
          continue;
        System.gc();

        Tensor mean;
        {
          PipelineNetwork wrapTiledAvg = ArtistryUtil.wrapTiledAvg(network.copy(), 600);
          ArtistryUtil.setPrecision(wrapTiledAvg, style.precision);
          mean = wrapTiledAvg.eval(styleInput).getDataAndFree().getAndFree(0);
          wrapTiledAvg.freeRef();
        }

        logger.info(String.format("%s : style mean = %s", layerType.name(), mean.prettyPrint()));
        logger.info(String.format(
            "%s : mean statistics = %s",
            layerType.name(),
            JsonUtil.toJson(new ScalarStatistics().add(mean.getData()).getMetrics())
        ));
        StyleTarget<T> styleTarget = self.styleTargets.get(key);
        styleTarget.mean.put(layerType, mean);

        if (0 == self.style.styles.entrySet().stream().filter(e1 -> e1.getKey().contains(key)).map(x -> x.getValue().params.get(
            layerType)).filter(x -> null != x).filter(x -> x.cov != 0).count())
          continue;

        System.gc();
        Tensor cov0;
        {
          PipelineNetwork wrapTiledAvg = ArtistryUtil.wrapTiledAvg(ArtistryUtil.gram(network.copy()), 600);
          ArtistryUtil.setPrecision(wrapTiledAvg, style.precision);
          cov0 = wrapTiledAvg.eval(styleInput).getDataAndFree().getAndFree(0);
          wrapTiledAvg.freeRef();
        }
        Tensor cov1;
        {
          PipelineNetwork wrapTiledAvg = ArtistryUtil.wrapTiledAvg(ArtistryUtil.gram(network.copy(), mean), 600);
          ArtistryUtil.setPrecision(wrapTiledAvg, style.precision);
          cov1 = wrapTiledAvg.eval(styleInput).getDataAndFree().getAndFree(0);
          wrapTiledAvg.freeRef();
        }

        int featureBands = mean.getDimensions()[2];
        int covarianceElements = cov1.getDimensions()[2];
        int selectedBands = covarianceElements / featureBands;
        logger.info(String.format("%s : target cov0 = %s", layerType.name(), cov0.reshapeCast(featureBands, selectedBands, 1).prettyPrint()));
        logger.info(String.format(
            "%s : cov0 statistics = %s",
            layerType.name(),
            JsonUtil.toJson(new ScalarStatistics().add(cov0.getData()).getMetrics())
        ));
        logger.info(String.format("%s : target cov1 = %s", layerType.name(), cov1.reshapeCast(featureBands, selectedBands, 1).prettyPrint()));
        logger.info(String.format(
            "%s : cov1 statistics = %s",
            layerType.name(),
            JsonUtil.toJson(new ScalarStatistics().add(cov1.getData()).getMetrics())
        ));
        styleTarget.cov0.put(layerType, cov0);
        styleTarget.cov1.put(layerType, cov1);
      }
    }
    return self;
  }

  /**
   * Gets fitness components.
   *
   * @param setup   the setup
   * @param nodeMap the node buildMap
   * @return the fitness components
   */
  @Nonnull
  public List<Tuple2<Double, DAGNode>> getFitnessComponents(NeuralSetup<T> setup, final Map<T, DAGNode> nodeMap) {
    List<Tuple2<Double, DAGNode>> functions = new ArrayList<>();
    functions.addAll(new ArrayList<>());
    functions.addAll(getStyleComponents(setup, nodeMap));
    return functions;
  }

  /**
   * Gets style components.
   *
   * @param setup   the setup
   * @param nodeMap the node buildMap
   * @return the style components
   */
  @Nonnull
  public ArrayList<Tuple2<Double, DAGNode>> getStyleComponents(NeuralSetup<T> setup, final Map<T, DAGNode> nodeMap) {
    ArrayList<Tuple2<Double, DAGNode>> styleComponents = new ArrayList<>();
    for (final T layerType : getLayerTypes())
      for (final List<CharSequence> keys : setup.style.styles.keySet()) {
        StyleTarget<T> styleTarget = keys.stream().map(x -> setup.styleTargets.get(x)).reduce((a, b) -> a.add(b)).map(x -> x.scale(1.0 / keys.size())).get();
        StyleCoefficients<T> styleCoefficients = setup.style.styles.get(keys);
        assert null != styleCoefficients;
        assert null != styleTarget;
        final DAGNode node = nodeMap.get(layerType);
        final PipelineNetwork network = (PipelineNetwork) node.getNetwork();
        LayerStyleParams styleParams = styleCoefficients.params.get(layerType);
        Tensor mean = styleTarget.mean.get(layerType);

        Tensor covariance;
        switch (styleCoefficients.centeringMode) {
          case Origin:
            covariance = styleTarget.cov0.get(layerType);
            break;
          case Dynamic:
          case Static:
            covariance = styleTarget.cov1.get(layerType);
            break;
          default:
            throw new RuntimeException();
        }
        styleComponents.addAll(getStyleComponents(node, network, styleParams, mean, covariance, styleCoefficients.centeringMode));
      }
    return styleComponents;
  }

  /**
   * Fitness function pipeline network.
   *
   * @param setup the setup
   * @return the pipeline network
   */
  @Nonnull
  public PipelineNetwork fitnessNetwork(NeuralSetup setup) {
    PipelineNetwork pipelineNetwork = getInstance().getNetwork();
    Map<T, DAGNode> nodes = new HashMap<>();
    Map<T, UUID> ids = getInstance().getNodes();
    ids.forEach((l, id) -> nodes.put(l, pipelineNetwork.getChildNode(id)));
    PipelineNetwork network = buildNetwork(setup, nodes, pipelineNetwork);
    //network = withClamp(network);
    ArtistryUtil.setPrecision(network, setup.style.precision);
    return network;
  }

  /**
   * Get key types t [ ].
   *
   * @return the t [ ]
   */
  @Nonnull
  public abstract T[] getLayerTypes();

  /**
   * Gets instance.
   *
   * @return the instance
   */
  public abstract U getInstance();

  /**
   * Measure style pipeline network.
   *
   * @param setup   the setup
   * @param nodeMap the node buildMap
   * @param network the network
   * @return the pipeline network
   */
  public PipelineNetwork buildNetwork(NeuralSetup<T> setup, final Map<T, DAGNode> nodeMap, final PipelineNetwork network) {
    List<Tuple2<Double, DAGNode>> functions = getFitnessComponents(setup, nodeMap);
    ArtistryUtil.reduce(network, functions, parallelLossFunctions);
    return network;
  }

  /**
   * Is tiled boolean.
   *
   * @return the boolean
   */
  public int getTiling() {
    return tiling;
  }

  /**
   * Sets tiled.
   *
   * @param tiling the tiled
   * @return the tiled
   */
  public TextureGeneration<T, U> setTiling(int tiling) {
    this.tiling = tiling;
    return this;
  }

  /**
   * The enum Centering mode.
   */
  public enum CenteringMode {
    /**
     * Dynamic centering mode.
     */
    Dynamic,
    /**
     * Static centering mode.
     */
    Static,
    /**
     * Origin centering mode.
     */
    Origin
  }

  /**
   * The type Vgg 16.
   */
  public static class VGG16 extends TextureGeneration<CVPipe_VGG16.Layer, CVPipe_VGG16> {

    public CVPipe_VGG16 getInstance() {
      return CVPipe_VGG16.INSTANCE;
    }

    @Nonnull
    public CVPipe_VGG16.Layer[] getLayerTypes() {
      return CVPipe_VGG16.Layer.values();
    }

  }

  /**
   * The type Vgg 19.
   */
  public static class VGG19 extends TextureGeneration<CVPipe_VGG19.Layer, CVPipe_VGG19> {

    public CVPipe_VGG19 getInstance() {
      return CVPipe_VGG19.INSTANCE;
    }

    @Nonnull
    public CVPipe_VGG19.Layer[] getLayerTypes() {
      return CVPipe_VGG19.Layer.values();
    }

  }

  /**
   * The type Layer style params.
   */
  public static class LayerStyleParams {
    /**
     * The Coeff style mean 0.
     */
    public final double mean;
    /**
     * The Coeff style bandCovariance 0.
     */
    public final double cov;
    /**
     * The Enhance.
     */
    public final double enhance;

    /**
     * Instantiates a new Layer style params.
     *
     * @param mean    the mean
     * @param cov     the bandCovariance
     * @param enhance the enhance
     */
    public LayerStyleParams(final double mean, final double cov, final double enhance) {
      this.mean = mean;
      this.cov = cov;
      this.enhance = enhance;
    }
  }

  /**
   * The type Style setup.
   *
   * @param <T> the type parameter
   */
  public static class StyleSetup<T extends LayerEnum<T>> {
    /**
     * The Precision.
     */
    public final Precision precision;
    /**
     * The Style png.
     */
    public final transient Map<CharSequence, BufferedImage> styleImages;
    /**
     * The Styles.
     */
    public final Map<List<CharSequence>, StyleCoefficients<T>> styles;


    /**
     * Instantiates a new Style setup.
     *
     * @param precision   the precision
     * @param styleImages the style png
     * @param styles      the styles
     */
    public StyleSetup(
        final Precision precision,
        final Map<CharSequence, BufferedImage> styleImages,
        final Map<List<CharSequence>, StyleCoefficients<T>> styles
    ) {
      this.precision = precision;
      this.styleImages = styleImages;
      this.styles = styles;
    }

  }

  /**
   * The type Style coefficients.
   *
   * @param <T> the type parameter
   */
  public static class StyleCoefficients<T extends LayerEnum<T>> {
    /**
     * The Dynamic center.
     */
    public final CenteringMode centeringMode;
    /**
     * The Params.
     */
    public final Map<T, LayerStyleParams> params = new HashMap<>();


    /**
     * Instantiates a new Style coefficients.
     *
     * @param centeringMode the dynamic center
     */
    public StyleCoefficients(final CenteringMode centeringMode) {
      this.centeringMode = centeringMode;
    }

    /**
     * Set style coefficients.
     *
     * @param layerType        the key type
     * @param coeff_style_mean the coeff style mean
     * @return the style coefficients
     */
    public StyleCoefficients set(final T layerType, final double coeff_style_mean) {
      return set(layerType, coeff_style_mean, 0);
    }

    /**
     * Set style coefficients.
     *
     * @param layerType        the key type
     * @param coeff_style_mean the coeff style mean
     * @param coeff_style_cov  the coeff style bandCovariance
     * @return the style coefficients
     */
    public StyleCoefficients<T> set(final T layerType, final double coeff_style_mean, final double coeff_style_cov) {
      return set(
          layerType,
          coeff_style_mean,
          coeff_style_cov,
          0
      );
    }

    /**
     * Set style coefficients.
     *
     * @param layerType        the key type
     * @param coeff_style_mean the coeff style mean
     * @param coeff_style_cov  the coeff style bandCovariance
     * @param enhance          the enhance
     * @return the style coefficients
     */
    public StyleCoefficients<T> set(final T layerType, final double coeff_style_mean, final double coeff_style_cov, final double enhance) {
      params.put(layerType, new LayerStyleParams(coeff_style_mean, coeff_style_cov, enhance));
      return this;
    }

  }

  /**
   * The type Content target.
   *
   * @param <T> the type parameter
   */
  public static class ContentTarget<T extends LayerEnum<T>> {
    /**
     * The Content.
     */
    public Map<T, Tensor> content = new HashMap<>();
  }

  /**
   * The type Style target.
   *
   * @param <T> the type parameter
   */
  public static class StyleTarget<T extends LayerEnum<T>> {
    /**
     * The Cov.
     */
    public Map<T, Tensor> cov0 = new HashMap<>();
    /**
     * The Cov.
     */
    public Map<T, Tensor> cov1 = new HashMap<>();
    /**
     * The Mean.
     */
    public Map<T, Tensor> mean = new HashMap<>();

    /**
     * Add style target.
     *
     * @param right the right
     * @return the style target
     */
    public StyleTarget<T> add(StyleTarget<T> right) {
      StyleTarget<T> newStyle = new StyleTarget<>();
      Stream.concat(mean.keySet().stream(), right.mean.keySet().stream()).distinct().forEach(layer -> {
        Tensor l = mean.get(layer);
        Tensor r = right.mean.get(layer);
        if (l != null && l != r) newStyle.mean.put(layer, l.add(r));
        else if (l != null) newStyle.mean.put(layer, l);
        else if (r != null) newStyle.mean.put(layer, r);
      });
      Stream.concat(cov0.keySet().stream(), right.cov0.keySet().stream()).distinct().forEach(layer -> {
        Tensor l = cov0.get(layer);
        Tensor r = right.cov0.get(layer);
        if (l != null && l != r) newStyle.cov0.put(layer, l.add(r));
        else if (l != null) newStyle.cov0.put(layer, l);
        else if (r != null) newStyle.cov0.put(layer, r);
      });
      Stream.concat(cov1.keySet().stream(), right.cov1.keySet().stream()).distinct().forEach(layer -> {
        Tensor l = cov1.get(layer);
        Tensor r = right.cov1.get(layer);
        if (l != null && l != r) newStyle.cov1.put(layer, l.add(r));
        else if (l != null) newStyle.cov1.put(layer, l);
        else if (r != null) newStyle.cov1.put(layer, r);
      });
      return newStyle;
    }

    /**
     * Scale style target.
     *
     * @param value the value
     * @return the style target
     */
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

  /**
   * The type Neural setup.
   *
   * @param <T> the type parameter
   */
  public static class NeuralSetup<T extends LayerEnum<T>> {

    /**
     * The Style parameters.
     */
    public final StyleSetup<T> style;

    /**
     * The Style targets.
     */
    public Map<CharSequence, StyleTarget<T>> styleTargets = new HashMap<>();


    /**
     * Instantiates a new Neural setup.
     *
     * @param style the style
     */
    public NeuralSetup(final StyleSetup<T> style) {
      this.style = style;
    }
  }

}
