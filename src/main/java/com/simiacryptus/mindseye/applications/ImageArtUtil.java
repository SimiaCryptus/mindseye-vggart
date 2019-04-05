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

import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.java.ImgTileAssemblyLayer;
import com.simiacryptus.mindseye.layers.java.ImgTileSelectLayer;
import com.simiacryptus.mindseye.models.CVPipe_Inception;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;
import java.awt.*;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.util.*;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class ImageArtUtil {

  public static Tensor degrade(
      final int lowResolution, final int size, Tensor image
  ) {
    BufferedImage bufferedImage = image.toImage();
    image.freeRef();
    Tensor lowRes = Tensor.fromRGB(TestUtil.resize(TestUtil.resize(bufferedImage, size, true), lowResolution, true));
    image = ArtistryUtil.expandPlasma(lowRes, 1000.0, 1.1, size).scale(0.9);
    lowRes.freeRef();
    return image;
  }

  //  @Nonnull
//  public static ColorTransfer.StyleSetup<CVPipe_Inception.Strata> getColorAnalogSetup(
//      final List<CharSequence> styleKeys,
//      final Precision precision,
//      final Tensor canvasBufferedImage,
//      final Map<CharSequence, Tensor> styleImages,
//      final CVPipe_Inception.Strata layer
//  ) {
//    return new ColorTransfer.StyleSetup<CVPipe_Inception.Strata>(
//        precision,
//        canvasBufferedImage,
//        new ColorTransfer.ContentCoefficients<>(),
//        styleImages,
//        TestUtil.buildMap(map -> {
//          ColorTransfer.StyleCoefficients<CVPipe_Inception.Strata> styleCoefficients =
//              new ColorTransfer.StyleCoefficients<>(ColorTransfer.CenteringMode.Origin);
//          styleCoefficients.set(
//              layer,
//              1e0,
//              1e0,
//              (double) 0
//          );
//          map.put(styleKeys, styleCoefficients);
//        })
//    );
//  }
  @Nonnull
  public static ColorTransfer.StyleSetup<CVPipe_Inception.Strata> getColorAnalogSetup(
      final List<CharSequence> styleKeys,
      final Precision precision,
      final Tensor canvasBufferedImage,
      final Map<CharSequence, Tensor> styleImages,
      final CVPipe_Inception.Strata layer
  ) {
    return new ColorTransfer.StyleSetup<CVPipe_Inception.Strata>(
        precision,
        canvasBufferedImage,
        new ColorTransfer.ContentCoefficients<>(),
        styleImages,
        TestUtil.buildMap(map -> {
          ColorTransfer.StyleCoefficients<CVPipe_Inception.Strata> styleCoefficients =
              new ColorTransfer.StyleCoefficients<>(ColorTransfer.CenteringMode.Origin);
          styleCoefficients.set(
              layer,
              1e0,
              1e0,
              (double) 0
          );
          map.put(styleKeys, styleCoefficients);
        })
    );
  }

  @Nonnull
  public static Map<CharSequence, Tensor> getStyleImages(
      final int resolution,
      final CharSequence... styleSources
  ) {
    return getStyleImages(new HashMap<>(), resolution, styleSources);
  }

  @Nonnull
  public static Map<CharSequence, Tensor> getStyleImages(
      final Map<CharSequence, ColorTransfer<CVPipe_Inception.Strata, CVPipe_Inception>> styleColorTransforms,
      final int resolution,
      final CharSequence... styleSources
  ) {
    return TestUtil.buildMap(y -> y.putAll(Arrays.stream(styleSources).collect(Collectors.toMap(x -> x, file -> {
      ColorTransfer colorTransfer = styleColorTransforms.get(file);
      Tensor tensor = ArtistryUtil.loadTensor(file, resolution);
      return colorTransfer == null ? tensor : colorTransfer.forwardTransform(tensor);
    }))));
  }

  @Nonnull
  public static Map<CharSequence, Tensor> getStyleImages(
      final CharSequence[] styleSources,
      final Map<CharSequence, ColorTransfer<CVPipe_Inception.Strata, CVPipe_Inception>> styleColorTransforms,
      final int resolutionX, final int resolutionY
  ) {
    return TestUtil.buildMap(y -> y.putAll(Arrays.stream(styleSources).collect(Collectors.toMap(x -> x, file -> {
      ColorTransfer colorTransfer = styleColorTransforms.get(file);
      Tensor tensor = ArtistryUtil.loadTensor(file, resolutionX, resolutionY);
      return colorTransfer == null ? tensor : colorTransfer.forwardTransform(tensor);
    }))));
  }

  @Nonnull
  public static Map<CharSequence, Tensor> getStyleImages2(
      final CharSequence[] styleSources,
      final Map<CharSequence, ColorTransfer<CVPipe_Inception.Strata, CVPipe_Inception>> styleColorTransforms,
      final int resolutionX, final int resolutionY
  ) {
    return TestUtil.buildMap(y -> y.putAll(Arrays.stream(styleSources).collect(Collectors.toMap(x -> x, file -> {
      ColorTransfer colorTransfer = styleColorTransforms.get(file);
      Tensor tensor = ArtistryUtil.loadTensor(file, resolutionX, resolutionY);
      return colorTransfer == null ? tensor : colorTransfer.forwardTransform(tensor);
    }))));
  }

  @Nonnull
  public static Tensor tiledTransfer(
      final ImageArtOpParams imageArtOpParams,
      final Tensor canvas,
      final int padding,
      final int torroidalOffsetX,
      final int torroidalOffsetY,
      final TileLayout tileLayout,
      final TileTransformer transformer,
      final Tensor content
  ) {
    imageArtOpParams.getLog().p(String.format(
        "Using Tile Size %s x %s to partition %s x %s png into %s x %s tiles",
        tileLayout.getTileSizeX(),
        tileLayout.getTileSizeY(),
        tileLayout.getWidth(),
        tileLayout.getHeight(),
        tileLayout.getCols(),
        tileLayout.getRows()
    ));
    final Tensor[] contentTiles = ImgTileSelectLayer.toTiles(imageArtOpParams.getLog(), content,
        tileLayout.getTileSizeX(), tileLayout.getTileSizeY(),
        tileLayout.getTileSizeX() - padding, tileLayout.getTileSizeY() - padding,
        torroidalOffsetX, torroidalOffsetY
    );
    final Tensor[] canvasTiles = ImgTileSelectLayer.toTiles(imageArtOpParams.getLog(), canvas,
        tileLayout.getTileSizeX(), tileLayout.getTileSizeY(),
        tileLayout.getTileSizeX() - padding, tileLayout.getTileSizeY() - padding,
        torroidalOffsetX, torroidalOffsetY
    );
    if (contentTiles.length != canvasTiles.length) throw new AssertionError(contentTiles.length + " != " + canvasTiles.length);
    Stream<Tensor> tensorStream = IntStream.range(0, contentTiles.length).mapToObj(i -> {
      return transformer.apply(contentTiles[i], canvasTiles[i], i);
    });
    Tensor[] resultTiles = tensorStream.toArray(i -> new Tensor[i]);
    ImgTileAssemblyLayer assemblyLayer = new ImgTileAssemblyLayer(tileLayout.getCols(), tileLayout.getRows())
        .setPaddingX(padding)
        .setPaddingY(padding)
        .setOffsetX(torroidalOffsetX)
        .setOffsetY(torroidalOffsetY);
    final Tensor resultImage = assemblyLayer.eval(resultTiles).getDataAndFree().getAndFree(0);
    assemblyLayer.freeRef();
    imageArtOpParams.getLog().p("Assembled Result:");
    imageArtOpParams.getLog().p(imageArtOpParams.getLog().png(resultImage.toImage(), "Assembled Canvas"));
    return resultImage;
  }

  public static Tensor colorTransfer(
      final ImageArtOpParams opParams,
      final ColorTransfer<CVPipe_Inception.Strata, CVPipe_Inception> colorTransfer,
      final ColorTransfer.StyleSetup<CVPipe_Inception.Strata> styleSetup,
      final CharSequence contentSource,
      final int tileSize,
      final Tensor canvasImage
  ) {
    canvasImage.assertAlive();
    int padding = 10;
    int width = canvasImage.getDimensions()[0];
    int height = canvasImage.getDimensions()[1];
    final Tensor contentImage = ArtistryUtil.loadTensor(contentSource, width, height);
    int cols = (int) Math.max(1, (Math.ceil((width - tileSize) * 1.0 / (tileSize - padding)) + 1));
    int rows = (int) Math.max(1, (Math.ceil((height - tileSize) * 1.0 / (tileSize - padding)) + 1));
    contentImage.assertAlive();
    canvasImage.assertAlive();
    if (cols > 1 || rows > 1) {
      int tileSizeX = (cols <= 1) ? width : (int) Math.ceil(((double) (width - padding) / cols) + padding);
      int tileSizeY = (rows <= 1) ? height : (int) Math.ceil(((double) (height - padding) / rows) + padding);
      opParams.getLog().p(String.format(
          "Using Tile Size %s x %s to partition %s x %s png into %s x %s tiles",
          tileSizeX,
          tileSizeY,
          width,
          height,
          cols,
          rows
      ));
      Tensor[] contentTiles = ImgTileSelectLayer.toTiles(
          opParams.getLog(),
          contentImage,
          tileSizeX,
          tileSizeY,
          tileSizeX - padding,
          tileSizeY - padding,
          0,
          0
      );
      Tensor[] canvasTiles = ImgTileSelectLayer.toTiles(
          opParams.getLog(),
          canvasImage,
          tileSizeX,
          tileSizeY,
          tileSizeX - padding,
          tileSizeY - padding,
          0,
          0
      );
      contentImage.assertAlive();
      canvasImage.assertAlive();
      if (contentTiles.length != canvasTiles.length)
        throw new AssertionError(contentTiles.length + " != " + canvasTiles.length);
      Tensor[] resultTiles = IntStream.range(0, contentTiles.length).mapToObj(i -> {
        return colorTransfer.transfer(
            opParams.getLog(),
            canvasTiles[i],
            new ColorTransfer.StyleSetup<>(
                styleSetup.precision,
                contentTiles[i],
                styleSetup.content,
                styleSetup.styleImages,
                styleSetup.styles
            ),
            opParams.getTrainingMinutes(),
            colorTransfer.measureStyle(new ColorTransfer.StyleSetup<>(
                styleSetup.precision,
                contentTiles[i],
                styleSetup.content,
                styleSetup.styleImages,
                styleSetup.styles
            )),
            opParams.getMaxIterations(),
            opParams.isVerbose()
        );
      }).toArray(i -> new Tensor[i]);
      Tensor resultImage = new ImgTileAssemblyLayer(cols, rows).setPaddingX(padding).setPaddingY(padding).eval(resultTiles).getData().get(0);
      opParams.getLog().p("Assembled Result:");
      opParams.getLog().p(opParams.getLog().png(resultImage.toImage(), "Assembled Canvas"));
      resultImage.assertAlive();
      return resultImage;
    } else {
      return colorTransfer.transfer(
          opParams.getLog(),
          canvasImage,
          styleSetup,
          opParams.getTrainingMinutes(),
          colorTransfer.measureStyle(styleSetup),
          opParams.getMaxIterations(),
          opParams.isVerbose()
      );
    }
  }

  @Nonnull
  public static SegmentedStyleTransfer.StyleSetup<CVPipe_Inception.Strata> setContentImage(
      final SegmentedStyleTransfer.StyleSetup<CVPipe_Inception.Strata> styleSetup,
      final Tensor tileImage
  ) {
    styleSetup.styleImages.values().stream().forEach(x -> x.assertAlive());
    return new SegmentedStyleTransfer.StyleSetup<>(
        styleSetup.precision,
        tileImage,
        styleSetup.content,
        styleSetup.styleImages,
        styleSetup.styles
    );
  }

  public static Map<SegmentedStyleTransfer.MaskJob, Set<Tensor>> getMasks(
      final Map<SegmentedStyleTransfer.MaskJob, Set<Tensor[]>> maskJobSetMap,
      final int i
  ) {
    return maskJobSetMap.entrySet().stream().collect(Collectors.toMap(
        Map.Entry::getKey,
        x -> x.getValue().stream().map(v -> v[i]).collect(Collectors.toSet())
    ));
  }

  @Nonnull
  public static SegmentedStyleTransfer.StyleCoefficients<CVPipe_Inception.Strata> getStyleCoefficients(
      final Map<CVPipe_Inception.Strata, Double> styleLayers,
      final double coeff_style_mean,
      final double coeff_style_cov,
      final double dreamCoeff
  ) {
    SegmentedStyleTransfer.StyleCoefficients<CVPipe_Inception.Strata> styleCoefficients =
        new SegmentedStyleTransfer.StyleCoefficients<>(SegmentedStyleTransfer.CenteringMode.Origin);
    styleLayers.forEach((layer, coeff) -> styleCoefficients.set(
        layer,
        coeff_style_mean * coeff,
        coeff_style_cov * coeff,
        dreamCoeff * coeff
    ));
    return styleCoefficients;
  }

  @Nonnull
  public static SegmentedStyleTransfer.ContentCoefficients<CVPipe_Inception.Strata> scale(
      final SegmentedStyleTransfer.ContentCoefficients<CVPipe_Inception.Strata> contentCoefficients,
      final double contentMixingCoeff
  ) {
    SegmentedStyleTransfer.ContentCoefficients<CVPipe_Inception.Strata> contentCoefficients_phase = new SegmentedStyleTransfer.ContentCoefficients<>();
    contentCoefficients.params.forEach((a, b) -> {
      contentCoefficients_phase.set(a, contentMixingCoeff * b);
    });
    return contentCoefficients_phase;
  }

  @Nonnull
  public static Map<CharSequence, ColorTransfer<CVPipe_Inception.Strata, CVPipe_Inception>> getColorStyleEnhance(
      final ImageArtOpParams imageArtOpParams,
      final Precision precision,
      final AtomicInteger resolution,
      final int minStyleWidth,
      final ColorTransfer.StyleCoefficients<CVPipe_Inception.Strata> coefficients,
      final CharSequence[] styleSources
  ) {
    Map<CharSequence, ColorTransfer<CVPipe_Inception.Strata, CVPipe_Inception>> styleColorTransforms = new HashMap<>();
    for (final CharSequence styleSource : styleSources) {
      Map<List<CharSequence>, ColorTransfer.StyleCoefficients<CVPipe_Inception.Strata>> coefficientsMap = TestUtil.buildMap(buildMap -> {
        buildMap.put(Arrays.asList(styleSource), coefficients);
      });
      ColorTransfer<CVPipe_Inception.Strata, CVPipe_Inception> colorTransfer = new ColorTransfer.Inception();
      Tensor styleImage = Tensor.fromRGB(ArtistryUtil.load(styleSource, resolution.get()));
      HashMap<CharSequence, Tensor> styleImages = new HashMap<>();
      styleImage.addRef();
      styleImages.put(styleSource, styleImage);
      Tensor colorEnhancedStyleImage = colorTransfer(
          new ImageArtOpParams(
              imageArtOpParams.getLog(),
              imageArtOpParams.getMaxIterations(),
              imageArtOpParams.getTrainingMinutes(),
              imageArtOpParams.isVerbose()
          ),
          colorTransfer,
          new ColorTransfer.StyleSetup<>(
              precision,
              null,
              new ColorTransfer.ContentCoefficients<>(),
              styleImages,
              coefficientsMap
          ),
          styleSource,
          minStyleWidth,
          styleImage
      );
      imageArtOpParams.getLog().p(imageArtOpParams.getLog().png(colorEnhancedStyleImage.toImage(), "Enhanced Style Image"));
      styleImage.freeRef();
      styleColorTransforms.put(styleSource, colorTransfer);
    }
    return styleColorTransforms;
  }

  @Nonnull
  public static Rectangle2D measure(final Font font, final String text) {
    final Rectangle2D bounds;
    Graphics2D graphics = (Graphics2D) new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB).getGraphics();
    graphics.setFont(font);
    String[] lines = text.split("\n");
    double width = Arrays.stream(lines).mapToInt(t -> (int) graphics.getFontMetrics().getStringBounds(t, graphics).getWidth()).max().getAsInt();
    int height = Arrays.stream(lines).mapToInt(t -> (int) graphics.getFontMetrics().getStringBounds(t, graphics).getHeight()).sum();
    double line1height = graphics.getFontMetrics().getStringBounds(lines[0], graphics).getHeight();
    bounds = new Rectangle2D.Double(0, line1height, width, height);
    return bounds;
  }

  @Nonnull
  public static Font fitSize(
      final String text,
      final int resolution,
      final int padding,
      final String fontName, final int style
  ) {
    final Font font;
    Graphics2D graphics = (Graphics2D) new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB).getGraphics();
    double width = 0;
    int size = 12;
    while (width < (resolution - 2 * padding) && size < 1000) {
      size += 2;
      graphics.setFont(new Font(fontName, style, size));
      width = Arrays.stream(text.split("\n")).mapToInt(t -> (int) graphics.getFontMetrics().getStringBounds(
          t,
          graphics
      ).getWidth()).max().getAsInt();
    }
    size -= 2;
    font = new Font(fontName, style, size);
    return font;
  }

  public interface TileTransformer {
    Tensor apply(Tensor contentTile, final Tensor canvasTile, int i);
  }

  public static class ImageArtOpParams {
    @Nonnull
    private final NotebookOutput log;
    private final int maxIterations;
    private final int trainingMinutes;
    private final boolean verbose;

    public ImageArtOpParams(@Nonnull final NotebookOutput log, final int maxIterations, final int trainingMinutes, final boolean verbose) {
      this.log = log;
      this.maxIterations = maxIterations;
      this.trainingMinutes = trainingMinutes;
      this.verbose = verbose;
    }

    public NotebookOutput getLog() {
      return log;
    }

    public int getMaxIterations() {
      return maxIterations;
    }

    public int getTrainingMinutes() {
      return trainingMinutes;
    }

    public boolean isVerbose() {
      return verbose;
    }
  }

  public static class TileData {
    private final Tensor canvasTile;
    private final Map<SegmentedStyleTransfer.MaskJob, Set<Tensor>> tileMasks;
    private final Tensor contentTile;

    public TileData(
        final Tensor contentTile1,
        final Tensor canvasTile1,
        final Map<SegmentedStyleTransfer.MaskJob, Set<Tensor>> masks
    ) {
      canvasTile = canvasTile1;
      contentTile = contentTile1;
      tileMasks = masks;
    }

    public Tensor getCanvasTile() {
      return canvasTile;
    }

    public Map<SegmentedStyleTransfer.MaskJob, Set<Tensor>> getTileMasks() {
      return tileMasks;
    }

    public Tensor getContentTile() {
      return contentTile;
    }

  }

  public static class TileLayout {
    private final int[] canvasDimensions;
    private final int width;
    private final int height;
    private final int cols;
    private final int rows;
    private final int tileSizeX;
    private final int tileSizeY;

    public TileLayout(
        final int tileSize,
        final Tensor canvas,
        final int padding,
        final int torroidalOffsetX,
        final int torroidalOffsetY
    ) {
      this(tileSize, padding, torroidalOffsetX, torroidalOffsetY, canvas.getDimensions());
    }

    public TileLayout(
        final int tileSize,
        final int padding,
        final int torroidalOffsetX,
        final int torroidalOffsetY,
        final int[] dimensions
    ) {
      canvasDimensions = dimensions;
      width = canvasDimensions[0];
      height = canvasDimensions[1];
      if (width != canvasDimensions[0])
        throw new AssertionError(width + " != " + canvasDimensions[0]);
      if (height != canvasDimensions[1])
        throw new AssertionError(height + " != " + canvasDimensions[1]);
      cols = (int) Math.max(1, (Math.ceil((width - tileSize - torroidalOffsetX) * 1.0 / (tileSize - padding)) + 1));
      rows = (int) Math.max(1, (Math.ceil((height - tileSize - torroidalOffsetY) * 1.0 / (tileSize - padding)) + 1));
      tileSizeX = (cols <= 1) ? width : (int) Math.ceil(((double) (width - padding - torroidalOffsetX) / cols) + padding);
      tileSizeY = (rows <= 1) ? height : (int) Math.ceil(((double) (height - padding - torroidalOffsetY) / rows) + padding);
    }

    public int[] getCanvasDimensions() {
      return canvasDimensions;
    }

    public int getWidth() {
      return width;
    }

    public int getHeight() {
      return height;
    }

    public int getCols() {
      return cols;
    }

    public int getRows() {
      return rows;
    }

    public int getTileSizeX() {
      return tileSizeX;
    }

    public int getTileSizeY() {
      return tileSizeY;
    }

  }

  public static class StyleTransformer implements TileTransformer {
    final HashMap<SegmentedStyleTransfer.MaskJob, Set<Tensor>> originalCache;
    private final SegmentedStyleTransfer<CVPipe_Inception.Strata, CVPipe_Inception> styleTransfer;
    private final SegmentedStyleTransfer.StyleSetup<CVPipe_Inception.Strata> styleSetup;
    Map<SegmentedStyleTransfer.MaskJob, Set<Tensor[]>> maskJobSetMap;
    SegmentedStyleTransfer.NeuralSetup measuredStyle;
    private NotebookOutput log;
    private ImageArtOpParams imageArtOpParams;

    public StyleTransformer(
        final ImageArtOpParams imageArtOpParams,
        final SegmentedStyleTransfer<CVPipe_Inception.Strata, CVPipe_Inception> styleTransfer,
        final TileLayout tileLayout,
        final int padding,
        final int torroidalOffsetX,
        final int torroidalOffsetY,
        final SegmentedStyleTransfer.StyleSetup<CVPipe_Inception.Strata> styleSetup
    ) {
      this.setImageArtOpParams(imageArtOpParams);
      this.styleTransfer = styleTransfer;
      this.styleSetup = styleSetup;
      setLog(imageArtOpParams.getLog());
      originalCache = new HashMap<>(styleTransfer.getMaskCache());
      maskJobSetMap = originalCache.entrySet().stream().collect(Collectors.toMap(
          x -> x.getKey(),
          x -> x.getValue().stream().map(v -> ImgTileSelectLayer.toTiles(
              getLog(),
              Tensor.fromRGB(TestUtil.resize(v.toImage(),
                  tileLayout.getWidth(),
                  tileLayout.getHeight())),
              tileLayout.getTileSizeX(),
              tileLayout.getTileSizeY(),
              tileLayout.getTileSizeX() - padding,
              tileLayout.getTileSizeY() - padding,
              torroidalOffsetX,
              torroidalOffsetY
          )).collect(Collectors.toSet())
      ));
      measuredStyle = getNeuralSetup();
    }

    @Override
    public Tensor apply(final Tensor contentTile, final Tensor canvasTile, final int i) {
      TileData tileData = new TileData(contentTile, canvasTile, getMasks(maskJobSetMap, i));
      getLog().p(String.format("Processing Tile %s with size %s", i, Arrays.toString(tileData.getCanvasTile().getDimensions())));
      styleTransfer.getMaskCache().clear();
      styleTransfer.getMaskCache().putAll(tileData.getTileMasks());
      SegmentedStyleTransfer.StyleSetup<CVPipe_Inception.Strata> tileSetup = setContentImage(styleSetup, tileData.getContentTile());
      styleTransfer.measureContent(getLog(), tileSetup, measuredStyle);
      return styleTransfer.transfer(
          getLog(),
          tileSetup,
          getImageArtOpParams().getMaxIterations(),
          measuredStyle,
          getImageArtOpParams().getTrainingMinutes(),
          getImageArtOpParams().isVerbose(),
          tileData.getCanvasTile()
      );
    }

    @Nonnull
    public SegmentedStyleTransfer.NeuralSetup getNeuralSetup() {
      SegmentedStyleTransfer.NeuralSetup<CVPipe_Inception.Strata> measureStyle = new SegmentedStyleTransfer.NeuralSetup<>(styleSetup);
      styleTransfer.measureStyles(getLog(), styleSetup, measureStyle);
      return measureStyle;
    }

    public NotebookOutput getLog() {
      return log;
    }

    public StyleTransformer setLog(NotebookOutput log) {
      this.log = log;
      return this;
    }

    public ImageArtOpParams getImageArtOpParams() {
      return imageArtOpParams;
    }

    public StyleTransformer setImageArtOpParams(ImageArtOpParams imageArtOpParams) {
      this.imageArtOpParams = imageArtOpParams;
      return this;
    }
  }
}
