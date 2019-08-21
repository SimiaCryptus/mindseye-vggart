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

package com.simiacryptus.mindseye.pyramid;

import com.simiacryptus.mindseye.applications.ArtistryUtil;
import com.simiacryptus.mindseye.applications.HadoopUtil;
import com.simiacryptus.mindseye.applications.ImageArtUtil;
import com.simiacryptus.mindseye.applications.SegmentedStyleTransfer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.models.CVPipe_Inception;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.mindseye.util.ImageUtil;
import com.simiacryptus.notebook.NotebookOutput;
import com.simiacryptus.util.JsonUtil;
import com.simiacryptus.util.Util;
import org.apache.commons.io.IOUtils;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

public class PyramidUtil {

  private static int viewerId = 0;

  @Nonnull
  public static BufferedImage reduce(final BufferedImage image) {
    BufferedImage result = new BufferedImage(image.getWidth() / 2, image.getHeight() / 2, image.getType());
    WritableRaster imageRaster = image.getRaster();
    WritableRaster resultRaster = result.getRaster();
    for (int x = 0; x < result.getWidth(); x++) {
      for (int y = 0; y < result.getHeight(); y++) {
        for (int b = 0; b < image.getRaster().getNumBands(); b++) {
          double v = 0;
          for (int xx = 0; xx < 2; xx++) {
            for (int yy = 0; yy < 2; yy++) {
              v += imageRaster.getSample(x * 2 + xx, y * 2 + yy, b);
            }
          }
          resultRaster.setSample(x, y, b, v / 4);
        }
      }
    }
    return result;
  }

  public static void writeViewer(@Nonnull final NotebookOutput log, final CharSequence name, final ImagePyramid... tilesource) {
    writeViewer(log, name, true, true, tilesource);
  }

  public static void writeViewer(
      @Nonnull final NotebookOutput log,
      final CharSequence name,
      final boolean wrapHorizontal,
      final boolean wrapVertical,
      final ImagePyramid... tilesource
  ) {
    String id = "openseadragon" + viewerId++;
    String html = "<html>\n" +
        "  <head>\n" +
        "\t<script src=\"../openseadragon/openseadragon.min.js\"></script>\n" +
        "</head>\n" +
        "<body>\n" +
        "  <div id=\"" + id + "\" style=\"width: 100%; height: 100%;\"></div>\n" +
        "<script src=\"openseadragon/openseadragon.min.js\"></script>\n" +
        "<script type=\"text/javascript\">\n" +
        "    var viewer = OpenSeadragon({\n" +
        "        id: \"" + id + "\",\n" +
        "        prefixUrl:     \"../openseadragon/images/\",\n" +
        "    navigatorSizeRatio: 0.5,\n" +
        "    wrapHorizontal:     " + wrapHorizontal + ",\n" +
        "    wrapVertical:     " + wrapVertical + ",\n" +
        "    sequenceMode: true,\n" +
        "    tileSources:   [" + Arrays.stream(tilesource).map(a -> a.tilespec(0)).reduce((a, b) -> a + "," + b).get() + "]\n" +
        "    });\n" +
        "</script>\n" +
        "</body>\n" +
        "</html>\n";
    log.eval(() -> {
      return JsonUtil.toJson(tilesource);
    });
    try {
      IOUtils.write(html, log.file("etc/" + name + ".view.html"), "UTF-8");
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    CharSequence name_without_etc;
    if (name.toString().startsWith("etc/")) {
      name_without_etc = name.toString().substring(4);
    } else {
      name_without_etc = name;
    }
    log.p("<a href=\"etc/" + name_without_etc + ".view.html\">Viewer</a>");
    log.p("<iframe src=\"etc/" + name_without_etc + ".view.html\" width=\"100%\" height=\"70%\"></iframe>");
  }

  public static ImagePyramid[] initImagePyramids(
      @Nonnull final NotebookOutput log,
      final CharSequence name,
      final int tileSize,
      final BufferedImage... rootImage
  ) {
    try {
      ImagePyramid[] levels = initImagePyramids(log, tileSize, 0, rootImage);
      writeViewer(log, name, levels);
      return levels;
    } catch (Throwable e) {
      log.p("Error creating pan-zoom", Util.toString(e::printStackTrace));
      return new ImagePyramid[0];
    }
  }

  public static ImagePyramid[] initImagePyramids(
      @Nonnull final NotebookOutput log,
      final int tileSize,
      final int minLevel,
      final BufferedImage[] rootImage
  ) {
    return IntStream.range(0, rootImage.length).mapToObj(imageNumber -> {
      final String prefix = String.format("tile_%d_", imageNumber);
      BufferedImage image = rootImage[imageNumber];
      return ImagePyramid.init(log, tileSize, minLevel, prefix, image);
    }).toArray(i -> new ImagePyramid[i]);
  }

  public static int getMaxLevel(final BufferedImage image, final double tileSize) {
    return (int) Math.ceil(Math.log(image.getWidth() / tileSize) / Math.log(2.0));
  }

  public static double getAspect(final double height, final int width) {
    return height / width;
  }

  @Nonnull
  public static BufferedImage toImage(final ValueSampler sample, final int width, final int height) {
    BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
    WritableRaster imageRaster = bufferedImage.getRaster();
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        for (int b = 0; b < imageRaster.getNumBands(); b++) {
          double xf = getAspect(x, width);
          double yf = getAspect(y, width);
          while (xf < 0) xf += 1;
          while (xf >= 1) xf -= 1;
          while (yf < 0) yf += 1;
          while (yf >= 1) yf -= 1;
          imageRaster.setSample(x, y, b, sample.getValue(xf, yf, b));
        }
      }
    }
    return bufferedImage;
  }

  public static void writeZip(
      final File root,
      final String file,
      final String zipName,
      Function<String, String> nameTransform
  ) throws IOException {
    File zipFile = File.createTempFile(zipName.replace("\\.zip$", ""), ".zip");
    try (OutputStream outputStream = new FileOutputStream(zipFile)) {
      IOUtils.copy(new ByteArrayInputStream(HadoopUtil.getData(file)), outputStream);
    }
    try (ZipFile zip = new ZipFile(zipFile)) {
      Enumeration<? extends ZipEntry> entries = zip.entries();
      while (entries.hasMoreElements()) {
        ZipEntry zipEntry = entries.nextElement();
        if (zipEntry.isDirectory()) continue;
        File entryFile = new File(root, nameTransform.apply(zipEntry.getName()));
        entryFile.getParentFile().mkdirs();
        FileOutputStream output = new FileOutputStream(entryFile);
        IOUtils.copy(zip.getInputStream(zipEntry), output);
        output.close();
      }
    }
    zipFile.delete();
  }

  public static double getAspect(final CharSequence styleSource) {
    return getAspect(ArtistryUtil.load(styleSource, -1));
  }

  public static double getAspect(final BufferedImage load) {
    if (null == load) return 1.0;
    return getAspect(load.getHeight(), load.getWidth());
  }

  public static void initJS(@Nonnull final NotebookOutput log) {
    try {
      writeZip(
          log.getResourceDir().getParentFile(),
          "https://s3-us-west-2.amazonaws.com/simiacryptus/openseadragon-bin-2.4.0.zip",
          "openseadragon-bin-2.4.0.zip",
          name -> name.replace("openseadragon-bin-2.4.0", "openseadragon")
      );
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Nonnull
  public static Function<BufferedImage, BufferedImage> getImageEnlargingFunction(
      final NotebookOutput log,
      final int width,
      final int height,
      final int trainingMinutes,
      final int maxIterations,
      final boolean verbose,
      final int magLevels,
      final int padding,
      final int style_resolution,
      final CharSequence... styleSources
  ) {
    final ImageArtUtil.TileLayout tileLayout = new ImageArtUtil.TileLayout(
        600,
        padding,
        0,
        0,
        new int[]{
            width,
            height}
    );
    SegmentedStyleTransfer<CVPipe_Inception.Strata, CVPipe_Inception> styleTransfer = getStyleTransfer();
    SegmentedStyleTransfer.StyleSetup<CVPipe_Inception.Strata> styleSetup = getStyleSetup(Precision.Float, style_resolution, styleSources);
    ImageArtUtil.StyleTransformer transformer = new ImageArtUtil.StyleTransformer(
        new ImageArtUtil.ImageArtOpParams(
            log,
            trainingMinutes,
            maxIterations,
            verbose
        ),
        styleTransfer,
        tileLayout,
        padding,
        0,
        0,
        styleSetup
    );
    return getImageEnlargingFunction(
        log,
        trainingMinutes,
        maxIterations,
        verbose,
        magLevels,
        padding,
        tileLayout,
        styleTransfer,
        styleSetup,
        transformer
    );
  }

  @Nonnull
  public static Function<BufferedImage, BufferedImage> getImageEnlargingFunction(
      final NotebookOutput log,
      final int trainingMinutes,
      final int maxIterations,
      final boolean verbose,
      final int magLevels,
      final int padding,
      final ImageArtUtil.TileLayout tileLayout,
      final SegmentedStyleTransfer<CVPipe_Inception.Strata, CVPipe_Inception> styleTransfer,
      final SegmentedStyleTransfer.StyleSetup<CVPipe_Inception.Strata> styleSetup,
      final ImageArtUtil.StyleTransformer transformer
  ) {
    final AtomicInteger tile = new AtomicInteger();
    final ImageArtUtil.ImageArtOpParams imageArtOpParams0 = new ImageArtUtil.ImageArtOpParams(
        log,
        trainingMinutes,
        maxIterations,
        verbose
    );
    SegmentedStyleTransfer.NeuralSetup measureStyle = styleTransfer.measureStyle(imageArtOpParams0.getLog(), styleSetup);
    return image -> {
      return log.subreport(sublog -> {
        final ImageArtUtil.ImageArtOpParams imageArtOpParams1 = new ImageArtUtil.ImageArtOpParams(
            sublog,
            trainingMinutes,
            maxIterations,
            verbose
        );
        BufferedImage canvas = ImageUtil.resize(
            image,
            (int) (image.getWidth() * Math.pow(2, magLevels)),
            true
        );
        sublog.p(sublog.jpg(image, "Input"));
        transformer.setLog(sublog).setImageArtOpParams(imageArtOpParams1);
        final HashMap<SegmentedStyleTransfer.MaskJob, Set<Tensor>> originalCache = new HashMap<>(styleTransfer.getMaskCache());
        final Tensor result;
        if (tileLayout.getCols() > 1 || tileLayout.getRows() > 1) {
          result = ImageArtUtil.tiledTransfer(
              imageArtOpParams1,
              Tensor.fromRGB(canvas),
              padding,
              0,
              0,
              tileLayout,
              transformer,
              Tensor.fromRGB(canvas)
          );
        } else {
          result = styleTransfer.transfer(
              imageArtOpParams1.getLog(),
              styleSetup,
              imageArtOpParams1.getMaxIterations(),
              measureStyle,
              imageArtOpParams1.getTrainingMinutes(),
              imageArtOpParams1.isVerbose(),
              Tensor.fromRGB(canvas)
          );
        }
        styleTransfer.getMaskCache().clear();
        styleTransfer.getMaskCache().putAll(originalCache);
        BufferedImage resultImage = result.toImage();
        sublog.p(sublog.jpg(resultImage, "Result Image"));
        return resultImage;
      }, log.getName() + "_" + String.format("Tile_%s", tile.getAndIncrement()));
    };
  }

  @Nonnull
  public static SegmentedStyleTransfer<CVPipe_Inception.Strata, CVPipe_Inception> getStyleTransfer() {
    return getStyleTransfer(1);
  }

  @Nonnull
  public static SegmentedStyleTransfer<CVPipe_Inception.Strata, CVPipe_Inception> getStyleTransfer(final int imageClusters) {
    SegmentedStyleTransfer<CVPipe_Inception.Strata, CVPipe_Inception> styleTransfer = new SegmentedStyleTransfer.Inception();
    styleTransfer.setStyle_masks(imageClusters);
    styleTransfer.setStyle_textureClusters(imageClusters);
    styleTransfer.setContent_colorClusters(imageClusters);
    styleTransfer.setContent_textureClusters(imageClusters);
    styleTransfer.setContent_masks(imageClusters);
    styleTransfer.parallelLossFunctions = true;
    styleTransfer.setTiled(false);
    return styleTransfer;
  }

  @Nonnull
  public static SegmentedStyleTransfer.StyleSetup<CVPipe_Inception.Strata> getStyleSetup(
      final Precision precision, final int style_resolution, final CharSequence... styleSources
  ) {
    SegmentedStyleTransfer.ContentCoefficients<CVPipe_Inception.Strata> contentCoefficients = new SegmentedStyleTransfer.ContentCoefficients<>();
    contentCoefficients.set(CVPipe_Inception.Strata.Layer_1, 1e-1);
    Map<CVPipe_Inception.Strata, Double> styleLayers = new HashMap<>();
    styleLayers.put(CVPipe_Inception.Strata.Layer_2, 1e0);
    styleLayers.put(CVPipe_Inception.Strata.Layer_3a, 1e0);
    return new SegmentedStyleTransfer.StyleSetup<>(
        precision,
        null,
        contentCoefficients,
        ImageArtUtil.getStyleImages(
            style_resolution, styleSources
        ),
        TestUtil.buildMap(x -> {
          x.put(
              Arrays.asList(styleSources),
              ImageArtUtil.getStyleCoefficients(
                  styleLayers,
                  1e0,
                  1e0,
                  5e-1
              )
          );
        })
    );
  }

  @Nonnull
  public static ValueSampler getValueSampler(final BufferedImage image) {
    return (xf, yf, band) -> {
      int x = (int) (xf * image.getWidth());
      int y = (int) (yf * image.getWidth());
      while (x < 0) x += image.getWidth();
      x %= image.getWidth();
      while (y < 0) y += image.getHeight();
      y %= image.getHeight();
      return image.getRaster().getSampleDouble(x, y, band);
    };
  }

}
