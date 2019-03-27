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

package com.simiacryptus.mindseye.texture_generation;

import com.simiacryptus.mindseye.ImageScript;
import com.simiacryptus.mindseye.applications.ArtistryUtil;
import com.simiacryptus.mindseye.applications.ColorTransfer;
import com.simiacryptus.mindseye.applications.ImageArtUtil;
import com.simiacryptus.mindseye.applications.TextureGeneration;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.models.CVPipe_Inception;
import com.simiacryptus.mindseye.models.CVPipe_Inception;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;
import java.awt.*;
import java.awt.geom.Rectangle2D;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

/**
 * The type Hi def.
 */
public abstract class StyleSurvey extends ImageScript {

  public final double coeff_style_mean;
  /**
   * The Coeff style bandCovariance.
   */
  public final double coeff_style_cov;
  /**
   * The Style sources.
   */
  public final String[] styleSources;
  private final double plasma_intensity;
  private final int style_resolution;
  private final String[] text;
  private final int padding;
  private final Color color;
  private final String fontName;
  private final int style;
  /**
   * The Dream coeff.
   */
  public double dreamCoeff;
  double aspect_ratio;
  Precision precision = Precision.Float;

  public StyleSurvey(
      final double coeff_style_mean,
      final double coeff_style_cov,
      final double dreamCoeff,
      final int style_resolution,
      final int padding,
      final String fontName,
      final int style,
      final Color fontColor,
      String[] text,
      final double aspect_ratio,
      final double plasma_intensity,
      final String... styleSources
  ) {
    this.coeff_style_mean = coeff_style_mean;
    this.coeff_style_cov = coeff_style_cov;
    this.dreamCoeff = dreamCoeff;
    this.style_resolution = style_resolution;
    this.styleSources = styleSources;
    this.text = text;
    this.padding = padding;
    color = fontColor;
    this.fontName = fontName;
    this.style = style;
    this.aspect_ratio = aspect_ratio;
    this.plasma_intensity = plasma_intensity;
  }

  public void accept(@Nonnull NotebookOutput log) {
    int index = 0;
    List<String> filteredSources = log.subreport("load_styles", subreport -> {
      return Arrays.stream(styleSources).filter(styleSource -> {
        try {
          subreport.p(subreport.png(ArtistryUtil.load(styleSource, style_resolution), "Style Image"));
          return true;
        } catch (Throwable e) {
          e.printStackTrace(System.out);
          return false;
        }
      }).collect(Collectors.toList());
    });
    for (final String styleSource : filteredSources) {
      for (final String txt : text) {
        int seedResolution = 200;
        if (!txt.isEmpty()) {
          fontSurvey(log, txt, seedResolution);
        }
        BufferedImage initialImage = getInitialImage(txt, padding, style, color, fontName, seedResolution);
        final AtomicReference<Tensor> canvas = new AtomicReference<>(Tensor.fromRGB(initialImage));
        canvas.set(log.subreport("Color_Space_Analog", subreport -> {
          ColorTransfer<CVPipe_Inception.Strata, CVPipe_Inception> contentColorTransform = new ColorTransfer.Inception() {
          }.setOrtho(false).setUnit(false);
          //colorSyncContentCoeffMap.set(CVPipe_Inception.Strata.Layer_1a, 1e-1);
          int colorSyncResolution = 600;
          Tensor resizedCanvas = Tensor.fromRGB(TestUtil.resize(
              canvas.get().toImage(),
              colorSyncResolution,
              (int) (aspect_ratio * colorSyncResolution)
          ));
          final ColorTransfer.StyleSetup<CVPipe_Inception.Strata> styleSetup = ImageArtUtil.getColorAnalogSetup(
              Arrays.asList(styleSource),
              precision,
              resizedCanvas,
              ImageArtUtil.getStyleImages(
                  new String[]{styleSource},
                  new HashMap<>(),
                  colorSyncResolution, (int) (aspect_ratio * colorSyncResolution)
              ),
              CVPipe_Inception.Strata.Layer_1
          );
          contentColorTransform.transfer(
              subreport,
              resizedCanvas,
              styleSetup,
              getTrainingMinutes(),
              contentColorTransform.measureStyle(styleSetup),
              getMaxIterations(),
              isVerbose()
          );
          return contentColorTransform.forwardTransform(canvas.get());
        }));
        Tensor subresult = log.subreport("Rendering_" + index++, subreport -> {
          subreport.p(subreport.png(ArtistryUtil.load(styleSource, style_resolution), "Style Image"));
          canvas.set(tiledTexturePaintingPhase(subreport, canvas.get().copy(), styleSource));
          return canvas.get();
        });
        log.p(log.png(subresult.toImage(), txt));
      }
    }
  }

  public Object fontSurvey(@Nonnull final NotebookOutput log, final String txt, final int textResolution) {
    return log.subreport("Fonts", subreport -> {
      Arrays.stream(GraphicsEnvironment.getLocalGraphicsEnvironment().getAvailableFontFamilyNames())
          .filter(x -> !x.equals("EmojiOne Color"))
          .forEach(fontname -> {
            subreport.p(fontname);
            subreport.p(subreport.png(getInitialImage(txt, padding, Font.PLAIN, color, fontname, textResolution), fontname));
            subreport.p(subreport.png(getInitialImage(txt, padding, Font.ITALIC, color, fontname, textResolution), fontname));
            subreport.p(subreport.png(getInitialImage(txt, padding, Font.BOLD, color, fontname, textResolution), fontname));
          });
      return null;
    });
  }

  public Tensor tiledTexturePaintingPhase(
      final NotebookOutput log,
      Tensor canvas,
      final String styleSource
  ) {
    setMaxIterations(20);
    int styleFactor = style_resolution / 600;

    this.dreamCoeff *= 1e1;
    canvas = tiledTexturePaintingPhase(log, canvas.copy(), 200, getStyleSetup(
        styleSource, 200 * styleFactor,
        CVPipe_Inception.Strata.Layer_1,
        CVPipe_Inception.Strata.Layer_4a,
        CVPipe_Inception.Strata.Layer_4c
    ));
    canvas = tiledTexturePaintingPhase(log, canvas.copy(), 250, getStyleSetup(
        styleSource, 250 * styleFactor,
        CVPipe_Inception.Strata.Layer_1,
        CVPipe_Inception.Strata.Layer_3b,
        CVPipe_Inception.Strata.Layer_4a
    ));
    canvas = tiledTexturePaintingPhase(log, canvas.copy(), 300, getStyleSetup(
        styleSource, 300 * styleFactor,
        CVPipe_Inception.Strata.Layer_4b,
        CVPipe_Inception.Strata.Layer_4c,
        CVPipe_Inception.Strata.Layer_4d
    ));
    canvas = tiledTexturePaintingPhase(log, canvas.copy(), 350, getStyleSetup(
        styleSource, 350 * styleFactor,
        CVPipe_Inception.Strata.Layer_4a,
        CVPipe_Inception.Strata.Layer_4b,
        CVPipe_Inception.Strata.Layer_4c
    ));
    canvas = tiledTexturePaintingPhase(log, canvas.copy(), 400, getStyleSetup(
        styleSource, 400 * styleFactor,
        CVPipe_Inception.Strata.Layer_1,
        CVPipe_Inception.Strata.Layer_3a,
        CVPipe_Inception.Strata.Layer_3b
    ));
    this.dreamCoeff /= 1e1;

    canvas = tiledTexturePaintingPhase(log, canvas.copy(), 500, getStyleSetup(
        styleSource, 500 * styleFactor,
        CVPipe_Inception.Strata.Layer_2,
        CVPipe_Inception.Strata.Layer_3a,
        CVPipe_Inception.Strata.Layer_3b
    ));
    canvas = tiledTexturePaintingPhase(log, canvas.copy(), 550, getStyleSetup(
        styleSource, 550 * styleFactor,
        CVPipe_Inception.Strata.Layer_2,
        CVPipe_Inception.Strata.Layer_3b,
        CVPipe_Inception.Strata.Layer_4a
    ));
    //this.dreamCoeff /= 1e1;
    canvas = tiledTexturePaintingPhase(log, canvas.copy(), 600, getStyleSetup(
        styleSource, 600 * styleFactor,
        CVPipe_Inception.Strata.Layer_1,
        CVPipe_Inception.Strata.Layer_2,
        CVPipe_Inception.Strata.Layer_3a,
        CVPipe_Inception.Strata.Layer_3b
    ));
    return canvas;
  }

  @Nonnull
  public TextureGeneration.StyleSetup<CVPipe_Inception.Strata> getStyleSetup(
      final String styleSource, final int style_resolution,
      final CVPipe_Inception.Strata... layers
  ) {
    final Map<List<CharSequence>, TextureGeneration.StyleCoefficients<CVPipe_Inception.Strata>> styles = TestUtil.buildMap(x -> {
      TextureGeneration.StyleCoefficients<CVPipe_Inception.Strata> styleCoefficients = new TextureGeneration.StyleCoefficients<>(
          TextureGeneration.CenteringMode.Origin);
      for (final CVPipe_Inception.Strata layer : layers) {
        styleCoefficients.set(
            layer,
            coeff_style_mean,
            coeff_style_cov,
            dreamCoeff
        );
      }
      x.put(
          Arrays.asList(styleSource),
          styleCoefficients
      );
    });
    return new TextureGeneration.StyleSetup<>(
        precision,
        TestUtil.buildMap(y -> y.putAll(
            styles.keySet().stream().flatMap(
                x -> x.stream())
                .collect(Collectors.toMap(
                    x -> x,
                    file -> ArtistryUtil.load(
                        file,
                        style_resolution
                    )
                )))),
        styles
    );
  }

  public Tensor tiledTexturePaintingPhase(
      final NotebookOutput log,
      Tensor canvas,
      final int width,
      final TextureGeneration.StyleSetup<CVPipe_Inception.Strata> styleSetup
  ) {
    TextureGeneration.Inception textureGeneration = new TextureGeneration.Inception();
    textureGeneration.parallelLossFunctions = true;
    int height = (int) (aspect_ratio * width);
    textureGeneration.setTiling((int) Math.max(Math.min((2.0 * Math.pow(600, 2)) / (width * height), 9), 2));
    canvas = (Tensor.fromRGB(TestUtil.resize(canvas.toImage(), width, height)));
    log.p("Input Parameters:");
    log.eval(() -> {
      return ArtistryUtil.toJson(styleSetup);
    });
    return (textureGeneration.optimize(
        log,
        textureGeneration.measureStyle(styleSetup), canvas,
        getTrainingMinutes(),
        getMaxIterations(), isVerbose(), styleSetup.precision
    ));
  }

  @Nonnull
  public BufferedImage getInitialImage(
      final String text,
      final int padding,
      final int style,
      final Color color,
      final String fontName, final int resolution
  ) {
    if (text.isEmpty()) {
      return ArtistryUtil.paint_Plasma(3, 1000.0, 1.1, resolution, (int) (aspect_ratio * resolution))
          .map(x -> x * plasma_intensity).toImage();
    }
    BufferedImage image = ArtistryUtil.paint_Plasma(3, 1000.0, 1.1, resolution).toImage();
    Font font = ImageArtUtil.fitSize(text, resolution, padding, fontName, style);
    Rectangle2D bounds = ImageArtUtil.measure(font, text);
    aspect_ratio = (2 * padding + bounds.getHeight()) / (2 * padding + bounds.getWidth());
    image = ArtistryUtil.paint_Plasma(3, 1000.0, 1.1, resolution, (int) (aspect_ratio * resolution))
        .map(x -> x * plasma_intensity).toImage();
    Graphics2D graphics = (Graphics2D) image.getGraphics();
    graphics.setColor(color);
    graphics.setFont(font);
    int y = (int) ((Rectangle2D.Double) bounds).y + padding;
    for (final String line : text.split("\n")) {
      Rectangle2D stringBounds = graphics.getFontMetrics().getStringBounds(line, graphics);
      double centeringOffset = (bounds.getWidth() - stringBounds.getWidth()) / 2;
      graphics.drawString(line, (int) (padding + centeringOffset), y);
      y += stringBounds.getHeight();
    }
    return image;
  }

}
