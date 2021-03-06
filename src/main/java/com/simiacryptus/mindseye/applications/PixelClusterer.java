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

import com.simiacryptus.lang.ref.RecycleBin;
import com.simiacryptus.lang.ref.ReferenceCountingBase;
import com.simiacryptus.mindseye.eval.ArrayTrainable;
import com.simiacryptus.mindseye.eval.Trainable;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.*;
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.mindseye.layers.cudnn.conv.SimpleConvolutionLayer;
import com.simiacryptus.mindseye.layers.java.AutoEntropyLayer;
import com.simiacryptus.mindseye.layers.java.NthPowerActivationLayer;
import com.simiacryptus.mindseye.models.CVPipe;
import com.simiacryptus.mindseye.models.LayerEnum;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import com.simiacryptus.mindseye.opt.IterativeTrainer;
import com.simiacryptus.mindseye.opt.line.QuadraticSearch;
import com.simiacryptus.mindseye.opt.orient.TrustRegionStrategy;
import com.simiacryptus.mindseye.opt.region.RangeConstraint;
import com.simiacryptus.mindseye.opt.region.StaticConstraint;
import com.simiacryptus.mindseye.opt.region.TrustRegion;
import com.simiacryptus.mindseye.test.StepRecord;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.notebook.NotebookOutput;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.swing.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class PixelClusterer<T extends LayerEnum<T>, U extends CVPipe<T>> {
  private static final Logger logger = LoggerFactory.getLogger(PixelClusterer.class);
  private final boolean recenter;
  private final double globalBias;
  private final double globalGain;
  private final double[] entropyBias;
  private int clusters;
  private double seedPcaPower;
  private int orientation;
  private double globalDistributionEmphasis;
  private double selectionEntropyAdj;
  private int maxIterations;
  private int timeoutMinutes;
  private double seedMagnitude;
  private boolean rescale;

  public PixelClusterer(
      final int clusters,
      final int orientation,
      final double globalDistributionEmphasis,
      final double selectionEntropyAdj,
      final int maxIterations,
      final int timeoutMinutes,
      final double seedPcaPower,
      final double seedMagnitude,
      final boolean rescale,
      final boolean recenter,
      final double globalBias,
      final double globalGain,
      final double[] entropyBias
  ) {
    this.setClusters(clusters);
    this.setOrientation(orientation);
    this.setGlobalDistributionEmphasis(globalDistributionEmphasis);
    this.setSelectionEntropyAdj(selectionEntropyAdj);
    this.setMaxIterations(maxIterations);
    this.setTimeoutMinutes(timeoutMinutes);
    this.setSeedPcaPower(seedPcaPower);
    this.setSeedMagnitude(seedMagnitude);
    this.rescale = rescale;
    this.recenter = recenter;
    this.globalBias = globalBias;
    this.globalGain = globalGain;
    this.entropyBias = entropyBias;
  }

  public PixelClusterer(final int clusters) {
    this(
        clusters,
        -1,
        3,
        0,
        20,
        10,
        -0.5,
        1e1,
        true,
        true,
        0,
        1e0,
        new double[]{2e-1, 5e-2, 1e-3}
    );
  }

  public static double[] bandCovariance(final Stream<double[]> pixelStream, final int pixels, final double[] mean, final double[] rms) {
    return Arrays.stream(pixelStream.map(pixel -> {
      double[] crossproduct = RecycleBin.DOUBLES.obtain(pixel.length * pixel.length);
      int k = 0;
      for (int j = 0; j < pixel.length; j++) {
        for (int i = 0; i < pixel.length; i++) {
          crossproduct[k++] = ((pixel[i] - mean[i]) * rms[i]) * ((pixel[j] - mean[j]) * rms[j]);
        }
      }
      RecycleBin.DOUBLES.recycle(pixel, pixel.length);
      return crossproduct;
    }).reduce((a, b) -> {
      for (int i = 0; i < a.length; i++) {
        a[i] += b[i];
      }
      RecycleBin.DOUBLES.recycle(b, b.length);
      return a;
    }).get()).map(x -> x / pixels).toArray();
  }

  private static List<Tensor> pca(final double[] bandCovariance, final double eigenPower) {
    @Nonnull final EigenDecomposition decomposition = new EigenDecomposition(toMatrix(bandCovariance));
    return IntStream.range(0, (int) Math.sqrt(bandCovariance.length)).mapToObj(vectorIndex -> {
      double[] data = decomposition.getEigenvector(vectorIndex).toArray();
      return new Tensor(data, 1, 1, data.length).scaleInPlace(Math.pow(decomposition.getRealEigenvalue(vectorIndex), eigenPower));
    }).collect(Collectors.toList());
  }

  private static int countPixels(final Tensor featureImage) {
    int[] dimensions = featureImage.getDimensions();
    int width = dimensions[0];
    int height = dimensions[1];
    return width * height;
  }

  @Nonnull
  private static Array2DRowRealMatrix toMatrix(final double[] covariance) {
    final int bands = (int) Math.sqrt(covariance.length);
    Array2DRowRealMatrix matrix = new Array2DRowRealMatrix(bands, bands);
    int k = 0;
    for (int x = 0; x < bands; x++) {
      for (int y = 0; y < bands; y++) {
        matrix.setEntry(x, y, covariance[k++]);
      }
    }
    return matrix;
  }

  public PipelineNetwork analyze(final T layer, final NotebookOutput log, final Tensor metrics) {
    Layer model = modelingNetwork(layer, metrics);
    for (final double entropyBias : entropyBias) {
      log.eval(() -> {
        int[] dimensions = metrics.getDimensions();
        PipelineNetwork netEntropy = model.andThenWrap(entropyNetwork(dimensions[0] * dimensions[1], entropyBias));
        MultiPrecision.setPrecision(netEntropy, Precision.Float);
        Trainable trainable = null;
        try {
          trainable = getTrainable(metrics, netEntropy);
          return train(trainable);
        } finally {
          netEntropy.freeRef();
          if (null != trainable) trainable.freeRef();
        }
      });
    }
    return model.freeAndThenWrap(new SoftmaxActivationLayer().setMode(SoftmaxActivationLayer.SoftmaxMode.CHANNEL));
  }

  public Layer modelingNetwork(final T layer, final Tensor metrics) {
    return modelingNetwork(
        getGlobalBias(),
        getGlobalGain(),
        metrics,
        isRecenter(),
        isRescale(),
        getClusters(),
        getSeedMagnitude(),
        getSeedPcaPower()
    );
  }

  @Nonnull
  public Layer modelingNetwork(
      final double globalBias,
      final double globalGain,
      final Tensor metrics,
      final boolean recenter,
      final boolean rescale,
      final int clusters,
      final double seedMagnitude,
      final double seedPcaPower
  ) {
    int[] dimensions = metrics.getDimensions();
    int bands = dimensions[2];
    BandReducerLayer bandReducerLayer = new BandReducerLayer();
    Tensor meanTensor = bandReducerLayer.setMode(PoolingLayer.PoolingMode.Avg).eval(metrics).getDataAndFree().getAndFree(0);
    bandReducerLayer.freeRef();
    if (!recenter) Arrays.fill(meanTensor.getData(), 0);
    logger.info("Mean=" + Arrays.toString(meanTensor.getData()));
    Tensor bias = new Tensor(meanTensor.getData()).mapAndFree(v1 -> v1 * -1);
    Tensor _globalBias = new Tensor(meanTensor.getData()).mapAndFree(v1 -> globalBias);
    PipelineNetwork network = PipelineNetwork.wrap(
        1,
        new ImgBandBiasLayer(bands).set(bias),
        new SquareActivationLayer(),
        new BandReducerLayer().setMode(PoolingLayer.PoolingMode.Avg),
        new NthPowerActivationLayer().setPower(-0.5)
    );
    Tensor scaled;
    try {
      scaled = network.eval(metrics).getDataAndFree().getAndFree(0).mapAndFree(x -> x == 0.0 ? 1.0 : x);
    } finally {
      network.freeRef();
    }
    if (!rescale) Arrays.fill(scaled.getData(), 1);
    logger.info("Scaling=" + Arrays.toString(scaled.getData()));
    double[] bandCovariance = bandCovariance(metrics.getPixelStream(), countPixels(metrics), meanTensor.getData(), scaled.getData());
    meanTensor.freeRef();
    List<Tensor> seedVectors = pca(bandCovariance, seedPcaPower).stream().collect(Collectors.toList());
    String convolutionLayerName = "mix";
    ConvolutionLayer convolutionLayer = new ConvolutionLayer(1, 1, bands, clusters);
    convolutionLayer.getKernel().setByCoord(c -> {
      int band = c.getCoords()[2];
      int index1 = band / clusters;
      int index2 = band % clusters;
//      int index1 = band % bands;
//      int index2 = band / bands;
      double v = seedMagnitude * seedVectors.get(index2 % seedVectors.size()).get(index1) * ((index2 < seedVectors.size()) ? 1 : 2 * (Math.random() - 0.5));
      return Math.min(Math.max(-1, v), 1);
    });
    seedVectors.forEach(ReferenceCountingBase::freeRef);
    PipelineNetwork pipelineNetwork = new PipelineNetwork(1);
    pipelineNetwork.wrap(new ImgBandBiasLayer(bands).set(bias)).freeRef();
    pipelineNetwork.wrap(
        new ProductLayer(),
        pipelineNetwork.getHead(),
        pipelineNetwork.constValueWrap(new Tensor(scaled.getData(), 1, 1, scaled.getData().length))
    ).freeRef();
    pipelineNetwork.wrap(new ImgBandBiasLayer(bands).set(_globalBias).freeze()).freeRef();
    pipelineNetwork.wrap(convolutionLayer.explode().setName(convolutionLayerName)).freeRef();
    convolutionLayer.freeRef();
    pipelineNetwork.wrap(
        new ProductLayer(),
        pipelineNetwork.getHead(),
        pipelineNetwork.constValueWrap(new Tensor(new double[]{globalGain}, 1, 1, 1))
    ).freeRef();
    scaled.freeRef();
    bias.freeRef();
    _globalBias.freeRef();
    return pipelineNetwork;
  }

  @Nonnull
  public Trainable getTrainable(final Tensor metrics, final DAGNetwork netEntropy) {
    MultiPrecision.setPrecision(netEntropy, Precision.Float);
    return new ArrayTrainable(netEntropy, 1).setVerbose(true).setMask(false).setData(Arrays.asList(new Tensor[][]{{metrics}}));
  }

  @Nonnull
  public Layer entropyNetwork(final int pixels, final double entropyBias) {
    PipelineNetwork netEntropy = new PipelineNetwork(1);
    Tensor weights = new Tensor(Math.pow(2, getSelectionEntropyAdj()));
    netEntropy.wrap(
        new BinarySumLayer(getOrientation(), getOrientation() * -Math.pow(2, getGlobalDistributionEmphasis())),
        netEntropy.wrap(PipelineNetwork.wrap(
            1,
            new SoftmaxActivationLayer().setMode(SoftmaxActivationLayer.SoftmaxMode.CHANNEL).setAlgorithm(SoftmaxActivationLayer.SoftmaxAlgorithm.ACCURATE),
            new ImgBandBiasLayer(getClusters()).setWeights(i -> entropyBias),
            new AutoEntropyLayer()
        ), netEntropy.getInput(0)),
        netEntropy.wrap(PipelineNetwork.wrap(
            1,
            new ScaleLayer(weights),
            new SoftmaxActivationLayer().setMode(SoftmaxActivationLayer.SoftmaxMode.CHANNEL).setAlgorithm(SoftmaxActivationLayer.SoftmaxAlgorithm.ACCURATE),
            new BandAvgReducerLayer().setAlpha(pixels),
            new ImgBandBiasLayer(getClusters()).setWeights(i -> entropyBias),
            new AutoEntropyLayer()
        ), netEntropy.getInput(0))
    ).freeRef();
    weights.freeRef();
    return netEntropy;
  }

  public JPanel train(final Trainable trainable) {
    @Nonnull ArrayList<StepRecord> history = new ArrayList<>();
    try {
      new IterativeTrainer(trainable)
          .setMonitor(TestUtil.getMonitor(history))
          .setOrientation(new TrustRegionStrategy() {
            @Override
            public TrustRegion getRegionPolicy(final Layer layer) {
              if (layer instanceof SimpleConvolutionLayer) return new RangeConstraint(-1, 1);
              if (layer instanceof ProductLayer) return new StaticConstraint();
              //return new StaticConstraint();
              return null;
            }
          })
          .setMaxIterations(getMaxIterations())
          .setLineSearchFactory(name -> new QuadraticSearch().setRelativeTolerance(1e-1))
          //.setLineSearchFactory(name -> new BisectionSearch().setSpanTol(1e-1).setCurrentRate(1e-1))
          .setTimeout(getTimeoutMinutes(), TimeUnit.MINUTES)
          .setTerminateThreshold(Double.NEGATIVE_INFINITY)
          .runAndFree();
    } catch (Throwable e) {
      logger.warn("Error training", e);
    } finally {
      return TestUtil.plot(history);
    }
  }

  public int getClusters() {
    return clusters;
  }

  public PixelClusterer setClusters(int clusters) {
    this.clusters = clusters;
    return this;
  }

  public int getOrientation() {
    return orientation;
  }

  public PixelClusterer setOrientation(int orientation) {
    this.orientation = orientation;
    return this;
  }

  public double getGlobalDistributionEmphasis() {
    return globalDistributionEmphasis;
  }

  public PixelClusterer setGlobalDistributionEmphasis(double globalDistributionEmphasis) {
    this.globalDistributionEmphasis = globalDistributionEmphasis;
    return this;
  }

  public double getSelectionEntropyAdj() {
    return selectionEntropyAdj;
  }

  public PixelClusterer setSelectionEntropyAdj(double selectionEntropyAdj) {
    this.selectionEntropyAdj = selectionEntropyAdj;
    return this;
  }

  public int getMaxIterations() {
    return maxIterations;
  }

  public PixelClusterer setMaxIterations(int maxIterations) {
    this.maxIterations = maxIterations;
    return this;
  }

  public int getTimeoutMinutes() {
    return timeoutMinutes;
  }

  public PixelClusterer setTimeoutMinutes(int timeoutMinutes) {
    this.timeoutMinutes = timeoutMinutes;
    return this;
  }

  @Override
  public String toString() {
    return getClass().getSimpleName() + "{" +
        "clusters=" + getClusters() +
        ", seedPcaPower=" + getSeedPcaPower() +
        ", orientation=" + getOrientation() +
        ", globalDistributionEmphasis=" + getGlobalDistributionEmphasis() +
        ", selectionEntropyAdj=" + getSelectionEntropyAdj() +
        ", maxIterations=" + getMaxIterations() +
        ", timeoutMinutes=" + getTimeoutMinutes() +
        '}';
  }

  public double getSeedPcaPower() {
    return seedPcaPower;
  }

  public PixelClusterer setSeedPcaPower(double seedPcaPower) {
    this.seedPcaPower = seedPcaPower;
    return this;
  }

  public double getSeedMagnitude() {
    return seedMagnitude;
  }

  public PixelClusterer setSeedMagnitude(double seedMagnitude) {
    this.seedMagnitude = seedMagnitude;
    return this;
  }

  public boolean isRecenter() {
    return recenter;
  }

  public boolean isRescale() {
    return rescale;
  }

  public void setRescale(boolean rescale) {
    this.rescale = rescale;
  }

  public double getGlobalBias() {
    return globalBias;
  }

  public double getGlobalGain() {
    return globalGain;
  }
}
