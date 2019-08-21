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
import com.simiacryptus.mindseye.layers.java.ImgTileSelectLayer;
import com.simiacryptus.mindseye.models.*;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.orient.GradientDescent;
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
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public abstract class SegmentedStyleTransfer<T extends LayerEnum<T>, U extends CVPipe<T>> {

  private static final Logger logger = LoggerFactory.getLogger(SegmentedStyleTransfer.class);
  private final Map<MaskJob, Set<Tensor>> maskCache = new ConcurrentHashMap<>();
  public boolean parallelLossFunctions = true;
  private boolean tiled = false;
  private int content_masks = 3;
  private int content_colorClusters = 3;
  private int content_textureClusters = 3;
  private int style_masks = 3;
  private int stlye_colorClusters = 3;
  private int style_textureClusters = 3;

  public static List<Tensor> alphaList(final Tensor styleInput, final Set<Tensor> tensors) {
    return tensors.stream().map(x -> alpha(styleInput, x)).collect(Collectors.toList());
  }

  public static Tensor alpha(final Tensor content, final Tensor mask) {
    int xbands = mask.getDimensions()[2] - 1;
    return content.mapCoords(c -> {
      int[] coords = c.getCoords();
      return content.get(c) * mask.get(coords[0], coords[1], Math.min(coords[2], xbands));
    });
  }

  public static double alphaMaskSimilarity(final Tensor contentMask, final Tensor styleMask) {
    Tensor l = contentMask.sumChannels();
    Tensor r = styleMask.sumChannels();
    int[] dimensions = r.getDimensions();
    Tensor resize = Tensor.fromRGB(ImageUtil.resize(l.toImage(), dimensions[0], dimensions[1])).sumChannels();
    Tensor a = resize.unit();
    Tensor b = r.unit();
    double dot = a.dot(b);
    a.freeRef();
    b.freeRef();
    r.freeRef();
    l.freeRef();
    return dot;
  }

  public static Map<Tensor, Tensor> alphaMap(final Tensor styleInput, final Set<Tensor> tensors) {
    assert null != styleInput;
    assert null != tensors;
    assert tensors.stream().allMatch(x -> x != null);
    return tensors.stream().distinct().collect(Collectors.toMap(x -> x, x -> alpha(styleInput, x)));
  }

  @Nonnull
  public Trainable getTrainable(final Tensor canvas, final PipelineNetwork network) {
    return new ArrayTrainable(network, 1).setVerbose(true).setMask(true).setData(Arrays.asList(new Tensor[][]{{canvas}}));
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
    if (null != styleParams && null != mean && (styleParams.cov != 0 || styleParams.mean != 0)) {
      double meanRms = mean.rms();
      double meanScale = 0 == meanRms ? 1 : (1.0 / meanRms);
      InnerNode negTarget = network.wrap(new ValueLayer(mean.scale(-1)), new DAGNode[]{});
      node.addRef();
      InnerNode negAvg = network.wrap(new BandAvgReducerLayer().setAlpha(-1), node);
      if (styleParams.enhance != 0 || styleParams.cov != 0) {
        DAGNode recentered;
        switch (centeringMode) {
          case Origin:
            node.addRef();
            recentered = node;
            break;
          case Dynamic:
            negAvg.addRef();
            node.addRef();
            recentered = network.wrap(new GateBiasLayer(), node, negAvg);
            break;
          case Static:
            node.addRef();
            negTarget.addRef();
            recentered = network.wrap(new GateBiasLayer(), node, negTarget);
            break;
          default:
            throw new RuntimeException();
        }
        int[] covDim = covariance.getDimensions();
        double covRms = covariance.rms();
        if (styleParams.enhance != 0) {
          recentered.addRef();
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
          recentered.addRef();
          styleComponents.add(new Tuple2<>(styleParams.cov, network.wrap(
              new MeanSqLossLayer().setAlpha(covScale),
              network.wrap(new ValueLayer(covariance), new DAGNode[]{}),
              network.wrap(new GramianLayer(), recentered)
          )
          ));
        }
        recentered.freeRef();
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

  public Tensor transfer(
      @Nonnull final NotebookOutput log,
      final StyleSetup<T> styleParameters,
      final int trainingMinutes,
      final NeuralSetup measureStyle,
      final int maxIterations,
      final boolean verbose,
      final Tensor canvas
  ) {
//      log.p("Input Content:");
//      log.p(log.png(styleParameters.contentImage, "Content Image"));
//      log.p("Style Content:");
//      styleParameters.styleImages.forEach((file, styleImage) -> {
//        log.p(log.png(styleImage, file));
//      });
//      log.p("Input Canvas:");
//      log.p(log.png(canvasImage, "Input Canvas"));
    System.gc();
    NotebookOutput trainingLog = verbose ? log : new NullNotebookOutput();
    if (1 < getContent_masks()) log.h2("Content Partitioning");
    Set<Tensor> masks = getMasks(
        log,
        measureStyle.contentSource,
        new MaskJob(getContent_masks(), getContent_colorClusters(), getContent_textureClusters(), "content")
    );
    System.gc();
    if (1 < getContent_masks()) log.h2("Content Painting");
    if (verbose) {
      log.p("Input Parameters:");
      log.eval(() -> {
        return ArtistryUtil.toJson(styleParameters);
      });
    }
    final FileHTTPD server = log.getHttpd();
    ImageUtil.monitorImage(canvas, false, false);
    String imageName = String.format("etc/image_%s.jpg", Long.toHexString(MarkdownNotebookOutput.random.nextLong()));
    log.p(String.format("<a href=\"%s\"><img src=\"%s\"></a>", imageName, imageName));
    Closeable jpeg = server.addGET(imageName, "image/jpeg", r -> {
      try {
        ImageIO.write(canvas.toImage(), "jpeg", r);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
    Trainable trainable = trainingLog.eval(() -> {
      PipelineNetwork network = fitnessNetwork(measureStyle, masks);
      network.setFrozen(true);
      if (null != server) ArtistryUtil.addLayersHandler(network, server);
      if (tiled) network = ArtistryUtil.tileCycle(network, 3);
      MultiPrecision.setPrecision(network, styleParameters.precision);
      TestUtil.instrumentPerformance(network);
      Trainable trainable1 = getTrainable(canvas, network);
      network.freeRef();
      return trainable1;
    });
    masks.forEach(ReferenceCountingBase::freeRef);
    try {
      @Nonnull ArrayList<StepRecord> history = new ArrayList<>();
      String training_name = String.format("etc/training_plot_%s.png", Long.toHexString(MarkdownNotebookOutput.random.nextLong()));
      log.p(String.format("<a href=\"%s\"><img src=\"%s\"></a>", training_name, training_name));
      Closeable closeable = log.getHttpd().addGET(training_name, "image/png", r -> {
        try {
          BufferedImage image1 = Util.toImage(TestUtil.plot(history));
          if (null != image1) ImageIO.write(image1, "png", r);
        } catch (IOException e) {
          logger.warn("Error writing result images", e);
        }
      });
      trainingLog.run(() -> {
        new IterativeTrainer(trainable)
            .setMonitor(TestUtil.getMonitor(history))
            .setOrientation(new TrustRegionStrategy(new GradientDescent()) {
              @Override
              public TrustRegion getRegionPolicy(final Layer layer) {
                return new RangeConstraint().setMin(1e-2).setMax(256);
              }
            })
            .setMaxIterations(maxIterations)
            .setIterationsPerSample(100)
//            .setLineSearchFactory(name -> new BisectionSearch().setSpanTol(1e-1).setCurrentRate(1e6).s(1e6))
            .setLineSearchFactory(name -> new QuadraticSearch().setRelativeTolerance(1e-1).setCurrentRate(1e4).setMaxRate(5e5))
//            .setLineSearchFactory(name -> new ArmijoWolfeSearch().setAlpha(1e4).setMaxAlpha(1e6))
            .setTimeout(trainingMinutes, TimeUnit.MINUTES)
            .setTerminateThreshold(Double.NEGATIVE_INFINITY)
            .runAndFree();
      });
      try {
        closeable.close();
        BufferedImage image = Util.toImage(TestUtil.plot(history));
        if (null != image) ImageIO.write(image, "png", log.file(training_name));
      } catch (IOException e) {
        logger.warn("Error writing result images", e);
      }
      try {
        ImageIO.write(canvas.toImage(), "jpeg", log.file(imageName));
      } catch (IOException e) {
        logger.warn("Error writing result images", e);
      }
      return canvas;
    } finally {
      trainable.freeRef();
    }
  }

  public NeuralSetup measureStyle(final NotebookOutput log, final StyleSetup<T> style) {
    NeuralSetup self = new NeuralSetup(style);
    measureStyles(log, style, self);
    measureContent(log, style, self);
    return self;
  }

  public List<CharSequence> getStyleKeys(final StyleSetup<T> style) {
    return style.styleImages.keySet().stream().collect(Collectors.toList());
  }

  public void measureContent(
      final NotebookOutput log,
      final StyleSetup<T> style,
      final NeuralSetup self
  ) {
    self.contentTarget = new ContentTarget<>();
    self.contentSource = style.contentImage;
    for (final T layerType : getLayerTypes()) {
      System.gc();
      Layer network = layerType.network();
      try {
        MultiPrecision.setPrecision((DAGNetwork) network, style.precision);
        //network = new ImgTileSubnetLayer(network, 400,400,400,400);
        if (null != style.contentImage) {
          Tensor content = network.eval(style.contentImage).getDataAndFree().getAndFree(0);
          System.gc();
          self.contentTarget.content.put(layerType, content);
          logger.info(String.format("%s : target content = %s", layerType.name(), content.prettyPrint()));
          logger.info(String.format(
              "%s : content statistics = %s",
              layerType.name(),
              JsonUtil.toJson(new ScalarStatistics().add(content.getData()).getMetrics())
          ));
        }
      } finally {
        network.freeRef();
      }
    }
  }

  public Map<Tensor, Set<Tensor>> measureStyles(
      final NotebookOutput log,
      final StyleSetup<T> style,
      final NeuralSetup<T> self
  ) {
    final List<CharSequence> styleKeys = getStyleKeys(style);
    IntStream.range(0, styleKeys.size()).forEach(i -> {
      self.styleTargets.put(styleKeys.get(i), new SegmentedStyleTarget<>());
    });
    Map<CharSequence, Tensor> styleInputs = styleKeys.stream().collect(Collectors.toMap(x -> x, x -> {
      Tensor tensor = style.styleImages.get(x);
      tensor.assertAlive();
      return tensor;
    }));
    if (1 < getStyle_masks())
      log.h2(String.format("Style Partitioning (%d/%d/%d)", getStyle_masks(), getStlye_colorClusters(), getStyle_textureClusters()));
    Map<Tensor, Set<Tensor>> masks = styleInputs.entrySet().stream().collect(Collectors.toMap(x -> x.getValue(), (styleInput) -> {
      Set<Tensor> masks1 = getMasks(
          log,
          styleInput.getValue(),
          new MaskJob(getStyle_masks(), getStlye_colorClusters(), getStyle_textureClusters(), styleInput.getKey())
      );
      assert null != masks1;
      assert 0 != masks1.size();
      assert masks1.stream().allMatch(x -> x != null);
      assert masks1.stream().count() == masks1.stream().distinct().count();
      return masks1;
    }));
    for (final T layerType : getLayerTypes()) {
      System.gc();
      Layer network = layerType.network();
      try {
        MultiPrecision.setPrecision((DAGNetwork) network, style.precision);
        for (Map.Entry<CharSequence, Tensor> styleEntry : styleInputs.entrySet()) {
          CharSequence key = styleEntry.getKey();
          SegmentedStyleTarget<T> segmentedStyleTarget = self.styleTargets.get(key);
          if (0 == self.style.styles.entrySet().stream().filter(e1 -> e1.getKey().contains(key)).map(x -> x.getValue().params.get(
              layerType)).filter(x -> null != x).filter(x -> x.mean != 0 || x.cov != 0).count())
            continue;
          Tensor styleInput = styleEntry.getValue();
          alphaMap(styleInput, masks.get(styleInput)).forEach((mask, styleMask) -> {
            StyleTarget<T> styleTarget = segmentedStyleTarget.getSegment(mask);
            if (0 == self.style.styles.entrySet().stream().filter(e1 -> e1.getKey().contains(key)).map(x -> x.getValue().params.get(
                layerType)).filter(x -> null != x).filter(x -> x.cov != 0).count())
              return;
            measureStyle(network, styleTarget, layerType, styleMask, 800);
            logStyle(styleTarget, layerType);
          });
        }
      } finally {
        network.freeRef();
      }
    }
    masks.forEach((k, v) -> {
      //k.freeRef();
      v.forEach(ReferenceCountingBase::freeRef);
    });
    return masks;
  }

  public void logStyle(final StyleTarget<T> styleTarget, final T layerType) {
    Tensor cov0 = styleTarget.cov0.get(layerType);
    Tensor cov1 = styleTarget.cov1.get(layerType);
    Tensor mean = styleTarget.mean.get(layerType);
    if (null == mean) return;
    int featureBands = mean.getDimensions()[2];
    int covarianceElements = cov1.getDimensions()[2];
    int selectedBands = covarianceElements / featureBands;
    logger.info(String.format("%s : style mean = %s", layerType.name(), mean.prettyPrint()));
    logger.info(String.format(
        "%s : mean statistics = %s",
        layerType.name(),
        JsonUtil.toJson(new ScalarStatistics().add(mean.getData()).getMetrics())
    ));
    logger.info(String.format("%s : target cov0 = %s", layerType.name(), cov0.reshapeCast(featureBands, selectedBands, 1).prettyPrintAndFree()));
    logger.info(String.format(
        "%s : cov0 statistics = %s",
        layerType.name(),
        JsonUtil.toJson(new ScalarStatistics().add(cov0.getData()).getMetrics())
    ));
    logger.info(String.format("%s : target cov1 = %s", layerType.name(), cov1.reshapeCast(featureBands, selectedBands, 1).prettyPrintAndFree()));
    logger.info(String.format(
        "%s : cov1 statistics = %s",
        layerType.name(),
        JsonUtil.toJson(new ScalarStatistics().add(cov1.getData()).getMetrics())
    ));
  }

  public void measureStyle(final Layer network, final StyleTarget<T> styleTarget, final T layerType, final Tensor image, int tileSize) {
    int[] dimensions = image.getDimensions();
    int width = tileSize;
    int height = tileSize;
    int strideX = tileSize;
    int strideY = tileSize;
    int cols = (int) Math.max(1, (Math.ceil((dimensions[0] - width) * 1.0 / strideX) + 1));
    int rows = (int) Math.max(1, (Math.ceil((dimensions[1] - height) * 1.0 / strideY) + 1));
    if (cols == 1 && rows == 1) {
      measureStyle(network, styleTarget, layerType, image);
    } else {
      StyleTarget<T> tiledStyle = IntStream.range(0, rows).mapToObj(x -> x).flatMap(row -> {
        return IntStream.range(0, cols).mapToObj(col -> {
          StyleTarget<T> styleTarget1 = new StyleTarget<>();
          int positionX = col * strideX;
          int positionY = row * strideY;
          assert positionX >= 0;
          assert positionY >= 0;
          assert positionX < dimensions[0];
          assert positionY < dimensions[1];
          ImgTileSelectLayer tileSelectLayer = new com.simiacryptus.mindseye.layers.java.ImgTileSelectLayer(width, height, positionX, positionY);
          Tensor selectedTile = tileSelectLayer.eval(image).getDataAndFree().getAndFree(0);
          tileSelectLayer.freeRef();
          double factor = (double) selectedTile.length() / image.length();
          measureStyle(network, styleTarget1, layerType, selectedTile);
          StyleTarget<T> scale = styleTarget1.scale(factor);
          styleTarget1.freeRef();
          return scale;
        });
      }).reduce((a, b) -> {
        StyleTarget<T> add = a.add(b);
        a.freeRef();
        b.freeRef();
        return add;
      }).get();
      System.gc();
      put(tiledStyle, styleTarget);
      tiledStyle.freeRef();
    }
  }

  public void put(final StyleTarget<T> fromStyle, final StyleTarget<T> toStyle) {
    toStyle.mean.putAll(fromStyle.mean);
    fromStyle.mean.values().stream().forEach(ReferenceCountingBase::addRef);
    toStyle.cov0.putAll(fromStyle.cov0);
    fromStyle.cov0.values().stream().forEach(ReferenceCountingBase::addRef);
    toStyle.cov1.putAll(fromStyle.cov1);
    fromStyle.cov1.values().stream().forEach(ReferenceCountingBase::addRef);
  }

  public void measureStyle(final Layer network, final StyleTarget<T> styleTarget, final T layerType, final Tensor image) {
    try {
      if (image.length() <= 0) throw new IllegalArgumentException(Arrays.toString(image.getDimensions()));
      Layer wrapAvg = null;
      Tensor mean;
      try {
        wrapAvg = ArtistryUtil.wrapTiledAvg(network.copy(), 400);
        System.gc();
        MultiPrecision.setPrecision((DAGNetwork) wrapAvg, Precision.Float);
        mean = wrapAvg.eval(image).getDataAndFree().getAndFree(0);
        if (mean.length() > 0 && styleTarget.mean.put(layerType, mean) != null) throw new AssertionError();
      } finally {
        if (null != wrapAvg) wrapAvg.freeRef();
      }

      Layer gram = null;
      try {
        gram = ArtistryUtil.wrapTiledAvg(ArtistryUtil.gram(network.copy()), 400);
        System.gc();
        MultiPrecision.setPrecision((DAGNetwork) gram, Precision.Float);
        Tensor cov0 = gram.eval(image).getDataAndFree().getAndFree(0);
        if (cov0.length() > 0 && styleTarget.cov0.put(layerType, cov0) != null) throw new AssertionError();
      } finally {
        if (null != gram) gram.freeRef();
      }
      try {
        gram = ArtistryUtil.wrapTiledAvg(ArtistryUtil.gram(network.copy(), mean), 400);
        MultiPrecision.setPrecision((DAGNetwork) gram, Precision.Float);
        System.gc();
        Tensor cov1 = gram.eval(image).getDataAndFree().getAndFree(0);
        if (cov1.length() > 0 && styleTarget.cov1.put(layerType, cov1) != null) throw new AssertionError();
      } finally {
        if (null != gram) gram.freeRef();
      }
    } finally {
      image.freeRef();
      System.gc();
    }
  }

  public Set<Tensor> getMasks(final NotebookOutput log, final Tensor value, final MaskJob maskJob1) {
    int width = value.getDimensions()[0];
    int height = value.getDimensions()[1];
    return getMaskCache().computeIfAbsent(maskJob1, maskJob -> {
      Set<Tensor> tensors = ImageSegmenter.quickMasks(
          log,
          value,
          maskJob.getStyle_masks(),
          maskJob.getStlye_colorClusters(),
          maskJob.getStyle_textureClusters()
      )
          .stream().distinct().collect(Collectors.toSet());
      assert null != tensors;
      return tensors;
    }).stream().map(img -> {
      Tensor tensor = Tensor.fromRGB(ImageUtil.resize(img.toImage(), width, height));
      assert null != tensor;
      return tensor;
    }).collect(Collectors.toSet());
  }

  @Nonnull
  public ArrayList<Tuple2<Double, DAGNode>> getStyleComponents(
      NeuralSetup<T> setup,
      final Map<T, DAGNode> nodeMap,
      final Function<SegmentedStyleTarget<T>, StyleTarget<T>> selector
  ) {
    ArrayList<Tuple2<Double, DAGNode>> styleComponents = new ArrayList<>();
    for (final List<CharSequence> keys : setup.style.styles.keySet()) {
      StyleTarget<T> styleTarget = keys.stream().map(x -> {
        SegmentedStyleTarget<T> obj = setup.styleTargets.get(x);
        StyleTarget<T> choose = selector.apply(obj);
        choose.addRef();
        return choose;
      }).reduce((a, b) -> {
        StyleTarget<T> r = a.add(b);
        a.freeRef();
        b.freeRef();
        return r;
      }).map(x -> {
        StyleTarget<T> r = x.scale(1.0 / keys.size());
        x.freeRef();
        return r;
      }).get();
      assert null != styleTarget;
      for (final T layerType : getLayerTypes()) {
        final StyleCoefficients<T> styleCoefficients = setup.style.styles.get(keys);
        assert null != styleCoefficients;
        styleComponents.addAll(getStyleComponents(nodeMap, layerType, styleCoefficients, styleTarget));
      }
      styleTarget.freeRef();
    }
    return styleComponents;
  }

  @Nonnull
  public ArrayList<Tuple2<Double, DAGNode>> getStyleComponents(
      final Map<T, DAGNode> nodeMap,
      final T layerType,
      final StyleCoefficients<T> styleCoefficients,
      final StyleTarget<T> chooseStyleSegment
  ) {
    final DAGNode node = nodeMap.get(layerType);
    if (null == node) throw new RuntimeException("Not Found: " + layerType);
    final PipelineNetwork network = (PipelineNetwork) node.getNetwork();
    LayerStyleParams styleParams = styleCoefficients.params.get(layerType);
    Tensor mean = chooseStyleSegment.mean.get(layerType);
    Tensor covariance;
    switch (styleCoefficients.centeringMode) {
      case Origin:
        covariance = chooseStyleSegment.cov0.get(layerType);
        break;
      case Dynamic:
      case Static:
        covariance = chooseStyleSegment.cov1.get(layerType);
        break;
      default:
        throw new RuntimeException();
    }
    return getStyleComponents(node, network, styleParams, mean, covariance, styleCoefficients.centeringMode);
  }

  @Nonnull
  public PipelineNetwork fitnessNetwork(final NeuralSetup setup, final Set<Tensor> masks) {
    U networkModel = getNetworkModel();
    PipelineNetwork mainNetwork = networkModel.getNetwork();
    Map<T, UUID> modelNodes = networkModel.getNodes();
    List<Tuple2<Double, DAGNode>> mainFunctions = new ArrayList<>();
    Map<T, DAGNode> mainNodes = getNodes(modelNodes, mainNetwork, null);
    mainFunctions.addAll(getContentComponents(setup, mainNodes));
    masks.forEach((contentMask) -> {
      HashMap<String, String> idMap = new HashMap<>();
      DAGNetwork branchNetwork = mainNetwork.scrambleCopy(idMap);
      //logger.info("Branch Keys");
      //branchNetwork.logKeys();
      Map<T, DAGNode> nodeMap = getNodes(modelNodes, branchNetwork, idMap);
      List<Tuple2<Double, DAGNode>> branchFunctions = new ArrayList<>();
      branchFunctions.addAll(getStyleComponents(setup, nodeMap,
          x -> x.segments.entrySet().stream().max(Comparator.comparingDouble(e -> alphaMaskSimilarity(
              contentMask,
              e.getKey()
          ))).get().getValue()
      ));
      if (!branchFunctions.isEmpty()) ArtistryUtil.reduce(branchNetwork, branchFunctions, parallelLossFunctions);
      InnerNode importNode = mainNetwork.wrap(
          branchNetwork,
          mainNetwork.wrap(new ProductLayer(), mainNetwork.getInput(0), mainNetwork.constValue(contentMask))
      );
      mainFunctions.add(new Tuple2<>(1.0, importNode));
    });
    ArtistryUtil.reduce(mainNetwork, mainFunctions, parallelLossFunctions);
    MultiPrecision.setPrecision(mainNetwork, setup.style.precision);
    return mainNetwork;
  }

  @Nonnull
  public Map<T, DAGNode> getNodes(final Map<T, UUID> modelNodes, final DAGNetwork network, final HashMap<String, String> replacements) {
    Map<T, DAGNode> nodes = new HashMap<>();
    modelNodes.forEach((l, id) -> {
      UUID replaced = null == replacements ? id : replace(replacements, id);
      DAGNode childNode = network.getChildNode(replaced);
      if (null == childNode) {
        logger.warn(String.format("Could not find Node ID %s (replaced from %s) to represent %s", replaced, id, l));
      } else {
        nodes.put(l, childNode);
      }
    });
    return nodes;
  }

  @Nonnull
  public UUID replace(final HashMap<String, String> replacements, final UUID id) {
    return UUID.fromString(replacements.getOrDefault(id.toString(), id.toString()));
  }

  @Nonnull
  public abstract T[] getLayerTypes();

  @Nonnull
  public ArrayList<Tuple2<Double, DAGNode>> getContentComponents(NeuralSetup<T> setup, final Map<T, DAGNode> nodeMap) {
    ArrayList<Tuple2<Double, DAGNode>> contentComponents = new ArrayList<>();
    for (final T layerType : getLayerTypes()) {
      final DAGNode node = nodeMap.get(layerType);
      final double coeff_content = !setup.style.content.params.containsKey(layerType) ? 0 : setup.style.content.params.get(layerType);
      final PipelineNetwork network1 = (PipelineNetwork) node.getNetwork();
      if (coeff_content != 0) {
        Tensor content = setup.contentTarget.content.get(layerType);
        contentComponents.add(new Tuple2<>(coeff_content, network1.wrap(new MeanSqLossLayer().setAlpha(1.0 / content.rms()),
            node, network1.wrap(new ValueLayer(content), new DAGNode[]{})
        )));
      }
    }
    return contentComponents;
  }

  public abstract U getNetworkModel();

  public boolean isTiled() {
    return tiled;
  }

  public SegmentedStyleTransfer<T, U> setTiled(boolean tiled) {
    this.tiled = tiled;
    return this;
  }

  public int getContent_masks() {
    return content_masks;
  }

  public SegmentedStyleTransfer<T, U> setContent_masks(int content_masks) {
    this.content_masks = content_masks;
    return this;
  }

  public int getContent_colorClusters() {
    return content_colorClusters;
  }

  public SegmentedStyleTransfer<T, U> setContent_colorClusters(int content_colorClusters) {
    this.content_colorClusters = content_colorClusters;
    return this;
  }

  public int getContent_textureClusters() {
    return content_textureClusters;
  }

  public SegmentedStyleTransfer<T, U> setContent_textureClusters(int content_textureClusters) {
    this.content_textureClusters = content_textureClusters;
    return this;
  }

  public int getStyle_masks() {
    return style_masks;
  }

  public SegmentedStyleTransfer<T, U> setStyle_masks(int style_masks) {
    this.style_masks = style_masks;
    return this;
  }

  public int getStlye_colorClusters() {
    return stlye_colorClusters;
  }

  public SegmentedStyleTransfer<T, U> setStlye_colorClusters(int stlye_colorClusters) {
    this.stlye_colorClusters = stlye_colorClusters;
    return this;
  }

  public int getStyle_textureClusters() {
    return style_textureClusters;
  }

  public SegmentedStyleTransfer<T, U> setStyle_textureClusters(int style_textureClusters) {
    this.style_textureClusters = style_textureClusters;
    return this;
  }

  public Map<MaskJob, Set<Tensor>> getMaskCache() {
    return maskCache;
  }

  public enum CenteringMode {
    Dynamic,
    Static,
    Origin
  }

  public static class VGG16 extends SegmentedStyleTransfer<CVPipe_VGG16.Layer, CVPipe_VGG16> {

    public CVPipe_VGG16 getNetworkModel() {
      return CVPipe_VGG16.INSTANCE;
    }

    @Nonnull
    public CVPipe_VGG16.Layer[] getLayerTypes() {
      return CVPipe_VGG16.Layer.values();
    }

  }

  public static class VGG19 extends SegmentedStyleTransfer<CVPipe_VGG19.Layer, CVPipe_VGG19> {
    public VGG19() {
    }

    public CVPipe_VGG19 getNetworkModel() {
      return CVPipe_VGG19.INSTANCE;
    }

    @Nonnull
    public CVPipe_VGG19.Layer[] getLayerTypes() {
      return CVPipe_VGG19.Layer.values();
    }

  }

  public static class Inception extends SegmentedStyleTransfer<CVPipe_Inception.Strata, CVPipe_Inception> {
    public Inception() {
    }

    public CVPipe_Inception getNetworkModel() {
      return CVPipe_Inception.INSTANCE;
    }

    @Nonnull
    public CVPipe_Inception.Strata[] getLayerTypes() {
      return CVPipe_Inception.Strata.values();
    }

  }

  public static class ContentCoefficients<T extends LayerEnum<T>> {
    public final Map<T, Double> params = new HashMap<>();

    public ContentCoefficients<T> set(final T l, final double v) {
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
      if (!styleImages.values().stream().allMatch(x -> x instanceof Tensor)) throw new AssertionError();
      this.precision = precision;
      this.contentImage = contentImage;
      this.styleImages = styleImages;
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

    public StyleCoefficients<T> set(final T layerType, final double coeff_style_mean, final double coeff_style_cov) {
      return set(
          layerType,
          coeff_style_mean,
          coeff_style_cov,
          0.0
      );
    }

    public StyleCoefficients<T> set(final T layerType, final double coeff_style_mean, final double coeff_style_cov, final double dream) {
      params.put(layerType, new LayerStyleParams(coeff_style_mean, coeff_style_cov, dream));
      return this;
    }

  }

  public static class ContentTarget<T extends LayerEnum<T>> {
    public Map<T, Tensor> content = new HashMap<>();
  }

  public static class MaskJob {
    private final int style_masks;
    private final int stlye_colorClusters;
    private final int style_textureClusters;
    private final CharSequence key;

    private MaskJob(final int style_masks, final int stlye_colorClusters, final int style_textureClusters, final CharSequence key) {
      this.style_masks = style_masks;
      this.stlye_colorClusters = stlye_colorClusters;
      this.style_textureClusters = style_textureClusters;
      this.key = key;
    }

    public int getStyle_masks() {
      return style_masks;
    }

    public int getStlye_colorClusters() {
      return stlye_colorClusters;
    }

    public int getStyle_textureClusters() {
      return style_textureClusters;
    }

    public CharSequence getKey() {
      return key;
    }

    @Override
    public boolean equals(final Object o) {
      if (this == o) return true;
      if (!(o instanceof MaskJob)) return false;
      final MaskJob maskJob = (MaskJob) o;
      return style_masks == maskJob.style_masks &&
          stlye_colorClusters == maskJob.stlye_colorClusters &&
          style_textureClusters == maskJob.style_textureClusters &&
          Objects.equals(key, maskJob.key);
    }

    @Override
    public int hashCode() {
      return Objects.hash(style_masks, stlye_colorClusters, style_textureClusters, key);
    }
  }

  public static class SegmentedStyleTarget<T extends LayerEnum<T>> {
    private final Map<Tensor, StyleTarget<T>> segments = new HashMap<>();

    public StyleTarget<T> getSegment(final Tensor styleMask) {
      synchronized (segments) {
        StyleTarget<T> styleTarget = segments.computeIfAbsent(styleMask, x -> {
          StyleTarget<T> tStyleTarget = new StyleTarget<>();
          styleMask.addRef();
          return tStyleTarget;
        });
        styleTarget.addRef();
        return styleTarget;
      }
    }
  }

  public static class StyleTarget<T extends LayerEnum<T>> extends ReferenceCountingBase {
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

  public static class NeuralSetup<T extends LayerEnum<T>> {

    public final StyleSetup<T> style;
    public ContentTarget<T> contentTarget = new ContentTarget<>();
    public Map<CharSequence, SegmentedStyleTarget<T>> styleTargets = new HashMap<>();
    public Tensor contentSource;


    public NeuralSetup(final StyleSetup<T> style) {
      this.style = style;
    }
  }
}
