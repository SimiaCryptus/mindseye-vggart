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

import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.mindseye.lang.cudnn.MultiPrecision;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.cudnn.BandReducerLayer;
import com.simiacryptus.mindseye.layers.cudnn.PoolingLayer;
import com.simiacryptus.mindseye.layers.cudnn.SoftmaxActivationLayer;
import com.simiacryptus.mindseye.layers.cudnn.conv.ConvolutionLayer;
import com.simiacryptus.mindseye.models.Hdf5Archive;
import com.simiacryptus.mindseye.models.ImageClassifier;
import com.simiacryptus.mindseye.models.VGG16_HDF5;
import com.simiacryptus.mindseye.models.VGG19_HDF5;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.util.Util;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.EigenDecomposition;
import org.apache.commons.math3.linear.RealVector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public abstract class PCAObjectLocation {

  private static final Logger logger = LoggerFactory.getLogger(PCAObjectLocation.class);

  public static Tensor renderAlpha(
      final double alphaPower,
      final Tensor img,
      final Result locationResult,
      final Tensor classification,
      final int category
  ) {
    TensorArray tensorArray = TensorArray.wrap(new Tensor(classification.getDimensions()).set(category, 1));
    DeltaSet<UUID> deltaSet = new DeltaSet<>();
    locationResult.accumulate(deltaSet, tensorArray);
    double[] rawDelta = deltaSet.getMap().entrySet().stream().filter(x -> x.getValue().target == img.getData()).findAny().get().getValue().getDelta();
    Tensor deltaColor = new Tensor(rawDelta, img.getDimensions()).mapAndFree(x -> Math.abs(x));
    Tensor delta1d = blur(deltaColor.sumChannels(), 3);
    return ImageUtil.normalizeBands(ImageUtil.normalizeBands(delta1d, 1).mapAndFree(x -> Math.pow(x, alphaPower)));
  }

  public static List<Tensor> blur(final List<Tensor> featureMasks, final int iterations) {
    if (0 >= iterations) return featureMasks;
    return featureMasks.stream().map(x -> blur(x, iterations)).collect(Collectors.toList());
  }

  @Nonnull
  public static Tensor blur(Tensor img, final int iterations) {
    int[] dimensions = img.getDimensions();
    int bands = dimensions[2];
    ConvolutionLayer blur = new ConvolutionLayer(3, 3, bands, bands);
    for (int i = 0; i < bands; i++) {
      blur.getKernel().set(0, 1, i * (bands + 1), 1.0);
      blur.getKernel().set(1, 1, i * (bands + 1), 1.0);
      blur.getKernel().set(1, 0, i * (bands + 1), 1.0);
      blur.getKernel().set(1, 2, i * (bands + 1), 1.0);
      blur.getKernel().set(2, 1, i * (bands + 1), 1.0);
    }
    blur.setPrecision(Precision.Float);
    final Layer blurExp = blur.explode();
    for (int i = 0; i < iterations; i++) {
      Tensor newImg = blurExp.eval(img).getDataAndFree().getAndFree(0);
      img.freeRef();
      img = newImg;
    }
    blur.freeRef();
    blurExp.freeRef();
    return img;
  }

  public abstract ImageClassifier getLocatorNetwork();

  public abstract ImageClassifier getClassifierNetwork();

  public void run(@Nonnull final NotebookOutput log) {
//    @Nonnull String logName = "cuda_" + log.getName() + ".log";
//    log.p(log.file((String) null, logName, "GPU Log"));
//    CudaSystem.addLog(new PrintStream(log.file(logName)));

    ImageClassifier classifier = getClassifierNetwork();
    Layer classifyNetwork = classifier.getNetwork();

    ImageClassifier locator = getLocatorNetwork();
    Layer locatorNetwork = locator.getNetwork();
    MultiPrecision.setPrecision((DAGNetwork) classifyNetwork, Precision.Float);
    MultiPrecision.setPrecision((DAGNetwork) locatorNetwork, Precision.Float);

    Tensor[][] inputData = loadImages_library();
//    Tensor[][] inputData = loadImage_Caltech101(log);
    double alphaPower = 0.8;

    final AtomicInteger index = new AtomicInteger(0);
    Arrays.stream(inputData).limit(10).forEach(row -> {
      log.h3("Image " + index.getAndIncrement());
      final Tensor img = row[0];
      log.p(log.png(img.toImage(), ""));
      Result classifyResult = classifyNetwork.eval(new MutableResult(row));
      Result locationResult = locatorNetwork.eval(new MutableResult(row));
      Tensor classification = classifyResult.getData().get(0);
      List<CharSequence> categories = classifier.getCategories();
      int[] sortedIndices = IntStream.range(0, categories.size()).mapToObj(x -> x)
          .sorted(Comparator.comparing(i -> -classification.get(i))).mapToInt(x -> x).limit(10).toArray();
      logger.info(Arrays.stream(sortedIndices)
          .mapToObj(i -> String.format("%s: %s = %s%%", i, categories.get(i), classification.get(i) * 100))
          .reduce((a, b) -> a + "\n" + b)
          .orElse(""));
      LinkedHashMap<CharSequence, Tensor> vectors = new LinkedHashMap<>();
      List<CharSequence> predictionList = Arrays.stream(sortedIndices).mapToObj(categories::get).collect(Collectors.toList());
      Arrays.stream(sortedIndices).limit(6).forEach(category -> {
        CharSequence name = categories.get(category);
        log.h3(name);
        Tensor alphaTensor = renderAlpha(alphaPower, img, locationResult, classification, category);
        log.p(log.png(img.toRgbImageAlphaMask(0, 1, 2, alphaTensor), ""));
        vectors.put(name, alphaTensor.unit());
      });

      Tensor avgDetection = vectors.values().stream().reduce((a, b) -> a.add(b)).get().scale(1.0 / vectors.size());
      Array2DRowRealMatrix covarianceMatrix = new Array2DRowRealMatrix(predictionList.size(), predictionList.size());
      for (int x = 0; x < predictionList.size(); x++) {
        for (int y = 0; y < predictionList.size(); y++) {
          Tensor l = vectors.get(predictionList.get(x));
          Tensor r = vectors.get(predictionList.get(y));
          covarianceMatrix.setEntry(x, y, null == l || null == r ? 0 : (l.minus(avgDetection)).dot(r.minus(avgDetection)));
        }
      }
      @Nonnull final EigenDecomposition decomposition = new EigenDecomposition(covarianceMatrix);


      for (int objectVector = 0; objectVector < 10; objectVector++) {
        log.h3("Eigenobject " + objectVector);
        double eigenvalue = decomposition.getRealEigenvalue(objectVector);
        RealVector eigenvector = decomposition.getEigenvector(objectVector);
        Tensor detectionRegion = IntStream.range(0, eigenvector.getDimension()).mapToObj(i -> {
          Tensor tensor = vectors.get(predictionList.get(i));
          return null == tensor ? null : tensor.scale(eigenvector.getEntry(i));
        }).filter(x -> null != x).reduce((a, b) -> a.add(b)).get();
        detectionRegion = detectionRegion.scale(255.0 / detectionRegion.rms());
        CharSequence categorization = IntStream.range(0, eigenvector.getDimension()).mapToObj(i -> {
          CharSequence category = predictionList.get(i);
          double component = eigenvector.getEntry(i);
          return String.format("<li>%s = %.4f</li>", category, component);
        }).reduce((a, b) -> a + "" + b).get();
        log.p(String.format("Object Detected: <ol>%s</ol>", categorization));
        log.p("Object Eigenvalue: " + eigenvalue);
        log.p("Object Region: " + log.png(img.toRgbImageAlphaMask(0, 1, 2, detectionRegion), ""));
        log.p("Object Region Compliment: " + log.png(img.toRgbImageAlphaMask(0, 1, 2, detectionRegion.scale(-1)), ""));
      }

//      final int[] orderedVectors = IntStream.range(0, 10).mapToObj(x -> x)
//        .sorted(Comparator.comparing(x -> -decomposition.getRealEigenvalue(x))).mapToInt(x -> x).toArray();
//      IntStream.range(0, orderedVectors.length)
//        .mapToObj(i -> {
//            //double realEigenvalue = decomposition.getRealEigenvalue(orderedVectors[i]);
//            return decomposition.getEigenvector(orderedVectors[i]).toArray();
//          }
//        ).toArray(i -> new double[i][]);

      log.p(String.format(
          "<table><tr><th>Cosine Distance</th>%s</tr>%s</table>",
          Arrays.stream(sortedIndices).limit(10).mapToObj(col -> "<th>" + categories.get(col) + "</th>").reduce((a, b) -> a + b).get(),
          Arrays.stream(sortedIndices).limit(10).mapToObj(r -> {
            return String.format("<tr><td>%s</td>%s</tr>", categories.get(r), Arrays.stream(sortedIndices).limit(10).mapToObj(col -> {
              Tensor l = vectors.get(categories.get(r));
              Tensor r2 = vectors.get(categories.get(col));
              return String.format("<td>%.4f</td>", (null == l || null == r2) ? 0 : Math.acos(l.dot(r2)));
            }).reduce((a, b) -> a + b).get());
          }).reduce((a, b) -> a + b).orElse("")
      ));
    });

    log.setFrontMatterProperty("status", "OK");
  }

  public Tensor[][] loadImages_library() {
    return Stream.of(
        "H:\\SimiaCryptus\\Artistry\\cat-and-dog.jpg"
    ).map(img -> {
      try {
        BufferedImage image = ImageIO.read(new File(img));
        image = ImageUtil.resize(image, 400, true);
        return new Tensor[]{Tensor.fromRGB(image)};
      } catch (IOException e) {
        throw new RuntimeException(e);
      }
    }).toArray(i -> new Tensor[i][]);
  }

  public <T> Comparator<T> getShuffleComparator() {
    final int seed = (int) ((System.nanoTime() >>> 8) % (Integer.MAX_VALUE - 84));
    return Comparator.comparingInt(a1 -> System.identityHashCode(a1) ^ seed);
  }

  public static class VGG16 extends PCAObjectLocation {

    @Override
    public ImageClassifier getLocatorNetwork() {
      ImageClassifier locator;
      try {
        locator = new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
          @Override
          protected void phase3b() {
            add(new BandReducerLayer().setMode(getFinalPoolingMode()));
          }
        }//.setSamples(5).setDensity(0.3)
            .setFinalPoolingMode(PoolingLayer.PoolingMode.Avg);
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
      return locator;
    }

    @Override
    public ImageClassifier getClassifierNetwork() {
      ImageClassifier classifier;
      try {
        classifier = new VGG16_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg16_weights.h5")))) {
          @Override
          protected void phase3b() {
            add(new SoftmaxActivationLayer()
                .setAlgorithm(SoftmaxActivationLayer.SoftmaxAlgorithm.ACCURATE)
                .setMode(SoftmaxActivationLayer.SoftmaxMode.CHANNEL));
            add(new BandReducerLayer().setMode(getFinalPoolingMode()));
          }
        }//.setSamples(5).setDensity(0.3)
            .setFinalPoolingMode(PoolingLayer.PoolingMode.Max);
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
      return classifier;
    }

  }

  public static class VGG19 extends PCAObjectLocation {

    @Override
    public ImageClassifier getLocatorNetwork() {
      ImageClassifier locator;
      try {
        locator = new VGG19_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg19_weights.h5")))) {
          @Override
          protected void phase3b() {
            add(new BandReducerLayer().setMode(getFinalPoolingMode()));
          }
        }//.setSamples(5).setDensity(0.3)
            .setFinalPoolingMode(PoolingLayer.PoolingMode.Avg);
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
      return locator;
    }

    @Override
    public ImageClassifier getClassifierNetwork() {
      ImageClassifier classifier;
      try {
        classifier = new VGG19_HDF5(new Hdf5Archive(Util.cacheFile(TestUtil.S3_ROOT.resolve("vgg19_weights.h5")))) {
          @Override
          protected void phase3b() {
            add(new SoftmaxActivationLayer()
                .setAlgorithm(SoftmaxActivationLayer.SoftmaxAlgorithm.ACCURATE)
                .setMode(SoftmaxActivationLayer.SoftmaxMode.CHANNEL));
            add(new BandReducerLayer().setMode(getFinalPoolingMode()));
          }
        }//.setSamples(5).setDensity(0.3)
            .setFinalPoolingMode(PoolingLayer.PoolingMode.Max);
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
      return classifier;
    }

  }
}
