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

import com.simiacryptus.lang.SerializableSupplier;
import com.simiacryptus.mindseye.applications.ArtistryUtil;
import com.simiacryptus.mindseye.applications.HadoopUtil;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.notebook.NotebookOutput;
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.jetbrains.annotations.NotNull;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public final class ImagePyramid implements Serializable {
  public transient final static Logger logger = LoggerFactory.getLogger(ImagePyramid.class);

  private int tileSize;
  private int level;
  private double aspect;
  private String prefix;

  public ImagePyramid() {
  }

  public ImagePyramid(final int tileSize, final int level, final double aspect, final String prefix) {
    this.tileSize = tileSize;
    this.level = level;
    this.aspect = aspect;
    this.prefix = prefix;
  }

  @Nonnull
  public static ImagePyramid init(
      @Nonnull final NotebookOutput log,
      final int tileSize,
      final int minLevel,
      final String prefix,
      final BufferedImage image
  ) {
    ImagePyramid imagePyramid = new ImagePyramid(tileSize, PyramidUtil.getMaxLevel(image, tileSize), PyramidUtil.getAspect(image), prefix);
    imagePyramid.write(log, minLevel, image);
    return imagePyramid;
  }

  public ImagePyramid write(
      @Nonnull final NotebookOutput log,
      final int minLevel,
      final BufferedImage image
  ) {
    ImagePyramid write = write(log, image);
    write.rebuild(minLevel);
    return write;
  }

  public void rebuild(final int minLevel) {
    for (int level = getLevel() - 1; level >= minLevel; level--) {
      new ImagePyramid(getTileSize(), level, getAspect(), getPrefix()).reduceImagePyramidLayer(PyramidUtil::reduce, true);
    }
  }

  @Nonnull
  public ImagePyramid write(@Nonnull final NotebookOutput log, final BufferedImage image) {
    return write(log, PyramidUtil.getValueSampler(image));
  }

  @Nonnull
  public ImagePyramid write(@Nonnull final NotebookOutput log, final ValueSampler valueSampler) {
    ImagePyramid withPrefix = logRelative(log);
    writeTiles(log, valueSampler);
    return withPrefix;
  }

  @Nonnull
  public ImagePyramid logRelative(@Nonnull final NotebookOutput log) {
    String destination = ("file:///" + log.getResourceDir().getAbsolutePath()) + "/" + getPrefix();
    return withPrefix(destination);
  }

  @Nonnull
  public ImagePyramid withPrefix(final String destination) {
    return new ImagePyramid(getTileSize(), level, getAspect(), destination);
  }

  @Nonnull
  public ImagePyramid zoomPyramid(
      @Nonnull final NotebookOutput log,
      final int incLevel,
      final String localPrefix,
      final Function<BufferedImage, BufferedImage> imageFunction
  ) {
    buildNewImagePyramidLayer(incLevel, 20,
        imageFunction,
        true
    );
    ImagePyramid pyramidLevel = new ImagePyramid(getTileSize(), getLevel() + incLevel, getAspect(), localPrefix);
    PyramidUtil.writeViewer(log, String.format("%s%d_%d", localPrefix, getLevel(), 2 + getLevel()), pyramidLevel);
    return pyramidLevel;
  }

  @Nonnull
  public BufferedImage renderTile(
      final int row,
      final int col,
      final ValueSampler sampler
  ) {
    return renderTile(row, col, sampler, getTileSize(), getLevel());
  }

  @NotNull
  public static BufferedImage renderTile(int row, int col, ValueSampler sampler, int tileSize, int level) {
    BufferedImage image = new BufferedImage(tileSize, tileSize, BufferedImage.TYPE_INT_RGB);
    WritableRaster raster = image.getRaster();
    double mag = Math.pow(2, level);
    for (int y = 0; y < tileSize; y++) {
      for (int x = 0; x < tileSize; x++) {
        for (int b = 0; b < 3; b++) {
          double xf = (((double) x / tileSize) + col) / mag;
          double yf = (((double) y / tileSize) + row) / mag;
          raster.setSample(x, y, b, sampler.getValue(xf, yf, b));
        }
      }
    }
    return image;
  }

  @Nonnull
  public BufferedImage tileImage(
      final int row,
      final int col,
      final ValueSampler sample,
      final int buffer
  ) {
    int width = getTileSize() + buffer * 2;
    int height = getTileSize() + buffer * 2;
    BufferedImage bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
    WritableRaster imageRaster = bufferedImage.getRaster();
    int errThrottle = 10;
    for (int x = 0; x < width; x++) {
      for (int y = 0; y < height; y++) {
        for (int b = 0; b < imageRaster.getNumBands(); b++) {
          double xf = (((double) (x - buffer) / getTileSize()) + col) / Math.pow(2, getLevel());
          double yf = (((double) (y - buffer) / getTileSize()) + row) / Math.pow(2, getLevel());
          while (xf < 0) xf += 1;
          while (xf >= 1) xf -= 1;
          while (yf < 0) yf += getAspect();
          while (yf >= getAspect()) yf -= getAspect();
          try {
            imageRaster.setSample(x, y, b, sample.getValue(xf, yf, b));
          } catch (Throwable e) {
            if (errThrottle-- > 0) logger.warn(String.format("Error with pixel %d,%d,%d aka %s,%s with size %s/%s", x, y, b, xf, yf, width, getAspect()), e);
          }
        }
      }
    }
    return bufferedImage;
  }

  public void writeTile(
      @Nonnull final NotebookOutput log,
      final ValueSampler sampler,
      final int row,
      final int col
  ) {
    String name = String.format(getPrefix() + "%d_%d_%d.jpg", getLevel(), row, col);
    log.jpgFile(this.renderTile(row, col, sampler), new File(log.getResourceDir(), name));
  }

  public void writeTiles(
      @Nonnull final NotebookOutput log,
      final ValueSampler sample
  ) {
    IntStream.range(0, (int) Math.pow(2, getLevel())).parallel().forEach(row -> {
      IntStream.range(0, (int) Math.pow(2, getLevel())).forEach(col -> {
        this.writeTile(log,
            sample, row, col
        );
      });
    });
  }

  @Nonnull
  public String tilespec() {
    return tilespec(0);
  }

  @Nonnull
  public String tilespec(
      final int minLevel
  ) {
    return tilespec(minLevel, getLevel());
  }

  @Nonnull
  public String tilespec(
      final int minLevel, final int maxLevel
  ) {
    double declaredSize = getTileSize() * Math.pow(2, maxLevel);
    return "{\n" +
        "        height: " + ((int) (declaredSize * getAspect())) + ",\n" +
        "        width:  " + ((int) declaredSize) + ",\n" +
        "        tileSize: " + getTileSize() + ",\n" +
        "        minLevel: " + minLevel + ",\n" +
        "        maxLevel: " + maxLevel + ",\n" +
        "        getTileUrl: function( level, x, y ){\n" +
        "            return \"" + getPrefix() + "\" + level + \"_\" + y + \"_\" + x + \".jpg\";\n" +
        "        }\n" +
        "    }";
  }


  public static final WritableRaster NULL_RASTER = new BufferedImage(1, 1, BufferedImage.TYPE_INT_RGB).getRaster();

  @Nonnull
  public ValueSampler newSampler() {
    Map<String, WritableRaster> cache = new HashMap<>();
    ValueSampler upperLevel = getLevel() > 0 ? new ImagePyramid(
        getTileSize(),
        getLevel() - 1,
        getAspect(),
        getPrefix()
    ).newSampler() : null;
    return (xf, yf, band) -> {
      while (xf < 0) xf += 1;
      while (yf < 0) yf += getAspect();
      yf %= getAspect();
      xf %= 1;
      int col = (int) Math.floor((xf * Math.pow(2, getLevel())));
      int row = (int) Math.floor((yf * Math.pow(2, getLevel())));
      int x = (int) ((xf * Math.pow(2, getLevel()) - col) * getTileSize());
      int y = (int) ((yf * Math.pow(2, getLevel()) - row) * getTileSize());
      String tilename = getPrefix() + String.format("%d_%d_%d.jpg", getLevel(), row, col);
      WritableRaster raster;
      synchronized (cache) {
        raster = cache.computeIfAbsent(tilename, s -> {
          try {
            return ArtistryUtil.load(s, -1).getRaster();
          } catch (Throwable e) {
            logger.warn("Error getting " + s, e);
            return NULL_RASTER;
          }
        });
      }
      if (raster == null || raster.getWidth() == 1) {
        if (null == upperLevel) return 0;
        else return upperLevel.getValue(xf, yf, band);
      } else {
        while (x < 0) x += raster.getWidth();
        x %= raster.getWidth();
        while (y < 0) y += raster.getHeight();
        y %= raster.getHeight();
        return raster.getSampleDouble(x, y, band);
      }
    };
  }

  public void buildNewImagePyramidLayer(
      final int scaleJump,
      final int padding,
      final String baseDest,
      final Function<BufferedImage, BufferedImage> tileProcessor,
      final boolean parallel
  ) {
    getImageTiles(padding, parallel).stream().forEach(tile -> collect(scaleJump, padding, baseDest, new ImageTile(tile.row, tile.col, tileProcessor.apply(tile.getImage()))));
  }

  public List<ImageTile> getImageTiles(int padding, boolean parallel) {
    Stream<SerializableSupplier<ImageTile>> stream = getImageTileFns(padding).stream();
    if (parallel) stream = stream.parallel();
    return stream.map(Supplier::get).collect(Collectors.toList());
  }

  public List<SerializableSupplier<ImageTile>> getImageTileFns(int padding) {
    ValueSampler imagePyramidReader = this.newSampler();
    int width = (int) Math.pow(2, getLevel());
    int height = (int) Math.ceil(width * getAspect());
    return IntStream.range(0, height).mapToObj(x -> x).flatMap(row0 ->
        IntStream.range(0, width).mapToObj(col0 -> new ImageTileFn(row0, col0, imagePyramidReader, padding))).collect(Collectors.toList());
  }

  public void collect(int scaleJump, int padding, String baseDest, ImageTile imageTile) {
    double scaleFactor = Math.pow(2, scaleJump);
    double scaledPadding = scaleFactor * padding;
    BufferedImage image = imageTile.getImage();
    ValueSampler tileSampler = (xf, yf, band) -> {
      int x = (int) (xf * (image.getWidth() - 2 * scaledPadding) + scaledPadding);
      int y = (int) (yf * (image.getWidth() - 2 * scaledPadding) + scaledPadding);
      while (x < 0) x += image.getWidth();
      x %= image.getWidth();
      while (y < 0) y += image.getHeight();
      y %= image.getHeight();
      try {
        return image.getRaster().getSampleDouble(x, y, band);
      } catch (Throwable e) {
        throw new RuntimeException(String.format("Error getting (%d,%d) from (%f,%f) (aspect %f)", x, y, xf, yf, getAspect()), e);
      }
    };
    IntStream.range(1, scaleJump + 1).forEach(jumpIndex -> {
      IntStream.range(0, (int) Math.pow(2, jumpIndex)).parallel().forEach(row -> {
        IntStream.range(0, (int) Math.pow(2, jumpIndex)).forEach(col -> {
          String tilename = String.format(
              "%d_%d_%d.jpg",
              jumpIndex + getLevel(),
              (int) (row + Math.pow(2, jumpIndex) * imageTile.getRow()),
              (int) (col + Math.pow(2, jumpIndex) * imageTile.getCol())
          );
          logger.info("Writing " + baseDest + tilename);
          FSDataOutputStream write = HadoopUtil.write(baseDest + tilename);
          try {
            ImageIO.write(renderTile(row, col, tileSampler, getTileSize(), jumpIndex), "jpg", write);
            write.close();
          } catch (IOException e) {
            throw new RuntimeException(e);
          }
        });
      });
    });
  }

  public void buildNewImagePyramidLayer(
      final int scaleJump,
      final int padding,
      final Function<BufferedImage, BufferedImage> tileProcessor,
      final boolean parallel
  ) {
    this.buildNewImagePyramidLayer(
        scaleJump, padding,
        getPrefix(), tileProcessor, parallel
    );
  }

  public void reduceImagePyramidLayer(
      final Function<BufferedImage, BufferedImage> tileProcessor,
      final boolean parallel
  ) {
    ValueSampler imagePyramidReader = new ImagePyramid(
        getTileSize(),
        getLevel() + 1,
        getAspect(),
        getPrefix()
    ).newSampler();
    IntStream range = IntStream.range(0, (int) Math.ceil(Math.pow(2, getLevel()) * getAspect()));
    if (parallel) range = range.parallel();
    range.forEach(row0 -> {
      IntStream.range(0, (int) Math.pow(2, getLevel())).forEach(col0 -> {
        BufferedImage readTile = new ImagePyramid(getTileSize() * 2, getLevel(), getAspect(), getPrefix()).tileImage(
            row0, col0, imagePyramidReader, 0
        );
        BufferedImage resize = tileProcessor.apply(readTile);
        String tilename = String.format(
            "%d_%d_%d.jpg",
            getLevel(),
            row0,
            col0
        );
        logger.info("Wrote " + getPrefix() + tilename);
        FSDataOutputStream write = HadoopUtil.write(getPrefix() + tilename);
        try {
          ImageIO.write(resize, "jpg", write);
          write.close();
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
    });
  }

  public BufferedImage assemble(
      final int width
  ) {
    return PyramidUtil.toImage(this.newSampler(), width, (int) (getAspect() * width));
  }

  public void copyImagePyramidLayer(
      final String dest,
      final boolean parallel
  ) {
    int gridSize = (int) Math.pow(2, getLevel());
    int expectedMaxRow = (int) Math.ceil(gridSize * getAspect());
    IntStream range = IntStream.range(0, expectedMaxRow);
    if (parallel) range = range.parallel();
    range.forEach(row0 -> {
      IntStream.range(0, gridSize).forEach(col0 -> {
        String tilename = String.format(
            "%d_%d_%d.jpg",
            getLevel(),
            row0,
            col0
        );
        String destName = dest + tilename;
        String srcName = getPrefix() + tilename;
        try {
          FSDataOutputStream write = HadoopUtil.write(destName);
          IOUtils.write(HadoopUtil.getData(srcName), write);
          write.close();
        } catch (Throwable e) {
          e.printStackTrace();
        }
        System.out.printf("Wrote %s to %s%n", srcName, destName);
      });
    });
  }

  public ImagePyramid writeViewer(
      @Nonnull final NotebookOutput log
  ) {
    PyramidUtil.writeViewer(log, getPrefix(), this);
    return this;
  }

  public ImagePyramid copyReducePyramid(
      final String destination
  ) {
    if (getPrefix() != destination) new ImagePyramid(
        getTileSize(),
        getLevel(),
        getAspect(),
        getPrefix()
    ).copyImagePyramidLayer(destination, true);
    ImagePyramid imagePyramid = withPrefix(destination);
    imagePyramid.rebuild(0);
    return imagePyramid;
  }

  public int getTileSize() {
    return tileSize;
  }

  public int getLevel() {
    return level;
  }

  public double getAspect() {
    return aspect;
  }

  public String getPrefix() {
    return prefix;
  }

  public static class ImageTile implements Serializable {
    private final int row;
    private final int col;
    private final Tensor image;

    public ImageTile(int row, int col, BufferedImage image) {
      this.row = row;
      this.col = col;
      this.image = Tensor.fromRGB(image);
    }

    public int getRow() {
      return row;
    }

    public int getCol() {
      return col;
    }

    public BufferedImage getImage() {
      return image.toImage();
    }
  }

  public class ImageTileFn implements SerializableSupplier<ImageTile> {
    private final Integer row0;
    private final int col0;
    private final ValueSampler imagePyramidReader;
    private final int padding;

    public ImageTileFn(Integer row0, int col0, ValueSampler imagePyramidReader, int padding) {
      this.row0 = row0;
      this.col0 = col0;
      this.imagePyramidReader = imagePyramidReader;
      this.padding = padding;
    }

    @Override
    public ImageTile get() {
      return new ImageTile(row0, col0, ImagePyramid.this.tileImage(row0, col0, imagePyramidReader, padding
      ));
    }
  }
}
