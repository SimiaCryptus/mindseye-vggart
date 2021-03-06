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
import com.simiacryptus.mindseye.models.*;
import com.simiacryptus.mindseye.network.DAGNetwork;
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
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public abstract class StyleTransfer<T extends LayerEnum<T>, U extends CVPipe<T>> {

  private static final Logger logger = LoggerFactory.getLogger(StyleTransfer.class);
  public boolean parallelLossFunctions = true;
  private boolean tiled = false;

  public Tensor transfer(final Tensor canvasImage, final StyleSetup<T> styleParameters, final int trainingMinutes, final NeuralSetup measureStyle) {
    return transfer(new NullNotebookOutput(), canvasImage, styleParameters, trainingMinutes, measureStyle, 50, true);
  }

  public Tensor transfer(
      @Nonnull final NotebookOutput log,
      final Tensor canvasData,
      final StyleSetup<T> styleParameters,
      final int trainingMinutes,
      final NeuralSetup measureStyle,
      final int maxIterations,
      final boolean verbose
  ) {
    try {
      transfer(log, styleParameters, trainingMinutes, measureStyle, maxIterations, verbose, canvasData);
      log.p("Result:");
      log.p(log.png(canvasData.toImage(), "Output Canvas"));
      return canvasData;
    } catch (Throwable e) {
      return canvasData;
    }
  }

  public void transfer(
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
    ImageUtil.monitorImage(canvas, false, false);
    String imageName = String.format("etc/image_%s.jpg", Long.toHexString(MarkdownNotebookOutput.random.nextLong()));
    log.p(String.format("<a href=\"%s\"><img src=\"%s\"></a>", imageName, imageName));
    Closeable jpeg = log.getHttpd().addGET(imageName, "image/jpeg", r -> {
      try {
        ImageIO.write(canvas.toImage(), "jpeg", r);
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    });
    if (verbose) {
      log.p("Input Parameters:");
      log.eval(() -> {
        return ArtistryUtil.toJson(styleParameters);
      });
    }
    NotebookOutput trainingLog = verbose ? log : new NullNotebookOutput();
    Trainable trainable = trainingLog.eval(() -> {
      PipelineNetwork network = fitnessNetwork(measureStyle);
      network.setFrozen(true);
      MultiPrecision.setPrecision(network, styleParameters.precision);
      TestUtil.instrumentPerformance(network);
      final FileHTTPD server = log.getHttpd();
      if (null != server) ArtistryUtil.addLayersHandler(network, server);
      if (tiled) network = ArtistryUtil.tileCycle(network, 3);
      Trainable trainable1 = getTrainable(canvas, network);
      network.freeRef();
      return trainable1;
    });
    try {
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
      trainingLog.eval(() -> {
        new IterativeTrainer(trainable)
            .setMonitor(TestUtil.getMonitor(history))
            .setOrientation(new TrustRegionStrategy() {
              @Override
              public TrustRegion getRegionPolicy(final Layer layer) {
                return new RangeConstraint().setMin(1e-2).setMax(256);
              }
            })
            .setMaxIterations(maxIterations)
            .setIterationsPerSample(100)
            .setLineSearchFactory(name -> new BisectionSearch().setSpanTol(1e-1).setCurrentRate(1e6))
            .setTimeout(trainingMinutes, TimeUnit.MINUTES)
            .setTerminateThreshold(Double.NEGATIVE_INFINITY)
            .runAndFree();
        return TestUtil.plot(history);
      });
      try {
        jpeg.close();
        ImageIO.write(canvas.toImage(), "jpeg", log.file(imageName));
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
      try {
        png.close();
        BufferedImage image = Util.toImage(TestUtil.plot(history));
        if (null != image) ImageIO.write(image, "png", log.file(training_name));
      } catch (IOException e) {
        logger.warn("Error writing result images", e);
      }
      log.p("Result:");
      log.p(log.png(canvas.toImage(), "Output Canvas"));
    } finally {
      trainable.freeRef();
    }
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

  public NeuralSetup measureStyle(final StyleSetup<T> style) {
    NeuralSetup self = new NeuralSetup(style);
    List<CharSequence> keyList = style.styleImages.keySet().stream().collect(Collectors.toList());
    List<Tensor> styleInputs = keyList.stream().map(x -> style.styleImages.get(x)).map(img -> Tensor.fromRGB(img)).collect(Collectors.toList());
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
        Tensor content = network.eval(style.contentImage).getDataAndFree().getAndFree(0);
        self.contentTarget.content.put(layerType, content);
        logger.info(String.format("%s : target content = %s", layerType.name(), content.prettyPrint()));
        logger.info(String.format(
            "%s : content statistics = %s",
            layerType.name(),
            JsonUtil.toJson(new ScalarStatistics().add(content.getData()).getMetrics())
        ));
        for (int i = 0; i < styleInputs.size(); i++) {
          Tensor styleInput = styleInputs.get(i);
          CharSequence key = keyList.get(i);
          StyleTarget<T> styleTarget = self.styleTargets.get(key);
          if (0 == self.style.styles.entrySet().stream().filter(e1 -> e1.getKey().contains(key)).map(x -> x.getValue().params.get(
              layerType)).filter(x -> null != x).filter(x -> x.mean != 0 || x.cov != 0).count())
            continue;
          System.gc();
          Layer wrapAvg = ArtistryUtil.wrapTiledAvg(network.copy(), 400);
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
        }
      } finally {
        network.freeRef();
      }
    }
    style.contentImage.freeRef();
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
      }).get();
      for (final T layerType : getLayerTypes()) {
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
      styleTarget.freeRef();

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
    //network = withClamp(network);
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

  public abstract U getInstance();

  public PipelineNetwork buildNetwork(NeuralSetup setup, final Map<T, DAGNode> nodeMap, final PipelineNetwork network) {
    List<Tuple2<Double, DAGNode>> functions = getFitnessComponents(setup, nodeMap);
    ArtistryUtil.reduce(network, functions, parallelLossFunctions);
    return network;
  }

  public boolean isTiled() {
    return tiled;
  }

  public StyleTransfer<T, U> setTiled(boolean tiled) {
    this.tiled = tiled;
    return this;
  }

  public enum CenteringMode {
    Dynamic,
    Static,
    Origin
  }

  public static class VGG16 extends StyleTransfer<CVPipe_VGG16.Layer, CVPipe_VGG16> {

    public CVPipe_VGG16 getInstance() {
      return CVPipe_VGG16.INSTANCE;
    }

    @Nonnull
    public CVPipe_VGG16.Layer[] getLayerTypes() {
      return CVPipe_VGG16.Layer.values();
    }

  }

  public static class VGG19 extends StyleTransfer<CVPipe_VGG19.Layer, CVPipe_VGG19> {

    public CVPipe_VGG19 getInstance() {
      return CVPipe_VGG19.INSTANCE;
    }

    @Nonnull
    public CVPipe_VGG19.Layer[] getLayerTypes() {
      return CVPipe_VGG19.Layer.values();
    }

  }

  public static class Inception extends StyleTransfer<CVPipe_Inception.Strata, CVPipe_Inception> {

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
    public final transient Map<CharSequence, BufferedImage> styleImages;
    public final Map<List<CharSequence>, StyleCoefficients<T>> styles;
    public final ContentCoefficients<T> content;
    public transient Tensor contentImage;


    public StyleSetup(
        final Precision precision,
        final Tensor contentImage,
        ContentCoefficients<T> contentCoefficients,
        final Map<CharSequence, BufferedImage> styleImages,
        final Map<List<CharSequence>, StyleCoefficients<T>> styles
    ) {
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
