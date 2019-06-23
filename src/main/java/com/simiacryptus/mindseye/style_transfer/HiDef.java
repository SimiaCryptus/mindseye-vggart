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

package com.simiacryptus.mindseye.style_transfer;

import com.simiacryptus.aws.exe.EC2NotebookRunner;
import com.simiacryptus.aws.exe.LocalNotebookRunner;
import com.simiacryptus.mindseye.ImageScript;
import com.simiacryptus.mindseye.applications.ArtistryUtil;
import com.simiacryptus.mindseye.applications.HadoopUtil;
import com.simiacryptus.mindseye.applications.StyleTransfer;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.cudnn.Precision;
import com.simiacryptus.mindseye.layers.java.ImgTileAssemblyLayer;
import com.simiacryptus.mindseye.layers.java.ImgTileSelectLayer;
import com.simiacryptus.mindseye.models.CVPipe_Inception;
import com.simiacryptus.mindseye.test.TestUtil;
import com.simiacryptus.notebook.NotebookOutput;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import java.util.stream.Stream;

public class HiDef extends ImageScript {

  public final double contentMixingCoeff = 1e0;
  public final double dreamCoeff = 1e0;


  public int resolution = 1200;
  public double coeff_style_mean = 1e1;
  public double coeff_style_cov = 1e0;
  public String[] styleSources = {
      "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/800px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
  };
  public String[] contentSources = {
//    "https://lh3.googleusercontent.com/wYifEEStf4s7gQlNlCawQe5Nd2MIWbSt-hv4_qk7A5hZuJy93bCWH1J3MwfCG5kVVjy20LOQ1n0LmwwZUpK6dVrFSl6hymJSPhDDGw7rl1IFjZ20MfhnY8qK8uVmDf-KCgw6E63iDLriehw-RacEt37hl4gsGIa4obv9jM6l3zT86m2R_23UjNNIw3ueAjxO0UjNO-hNXVUTK7sXAnvPa4C-60fmFJgX2mArcNHfBRSEId1NNP57xcllkgXt_FVwE8wKRFqPKbZS_vQ0QB8tfji_vGXyIXgney0I0k4oIb5y_Jdz0W7eUKhW_CD_3sgUD68fHuZk_uSHsz0R8BQcXSXYAKm0rumvgMk6-tD82hxbYf8kBoq60Qoq-XDY90y7J0yp7VOgp6wevZ8Gfvn0P-1cmY9DKLjRh8JBj_299QlzTeecFqWRfUcI7pU7JoHOAtecbiV57BO3MGXpGKyHQOZjMaXHOpo4Za6kG4I4YzLMR2m01hRY3b-Vf_PnA7fCs4kF73wMPpz8h26G4w8H3jUqB_rhi5N3JIybgf2eLArqNh-me0LDnlqyVf4esT5UgchvwOd6qsN2cBzqycGVKqDrh1FPmEwJ7dE_dlz0d8hzK9Uiw2TyYEXZteW5ICdwtOPM0Qxa_IDfQYIlh7dVE77H0w-ffoF8=w489-h652-no",
      "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/The_Earth_seen_from_Apollo_17.jpg/1024px-The_Earth_seen_from_Apollo_17.jpg"
  };

  public String[] contentSeeds = {
//    "https://lh3.googleusercontent.com/wYifEEStf4s7gQlNlCawQe5Nd2MIWbSt-hv4_qk7A5hZuJy93bCWH1J3MwfCG5kVVjy20LOQ1n0LmwwZUpK6dVrFSl6hymJSPhDDGw7rl1IFjZ20MfhnY8qK8uVmDf-KCgw6E63iDLriehw-RacEt37hl4gsGIa4obv9jM6l3zT86m2R_23UjNNIw3ueAjxO0UjNO-hNXVUTK7sXAnvPa4C-60fmFJgX2mArcNHfBRSEId1NNP57xcllkgXt_FVwE8wKRFqPKbZS_vQ0QB8tfji_vGXyIXgney0I0k4oIb5y_Jdz0W7eUKhW_CD_3sgUD68fHuZk_uSHsz0R8BQcXSXYAKm0rumvgMk6-tD82hxbYf8kBoq60Qoq-XDY90y7J0yp7VOgp6wevZ8Gfvn0P-1cmY9DKLjRh8JBj_299QlzTeecFqWRfUcI7pU7JoHOAtecbiV57BO3MGXpGKyHQOZjMaXHOpo4Za6kG4I4YzLMR2m01hRY3b-Vf_PnA7fCs4kF73wMPpz8h26G4w8H3jUqB_rhi5N3JIybgf2eLArqNh-me0LDnlqyVf4esT5UgchvwOd6qsN2cBzqycGVKqDrh1FPmEwJ7dE_dlz0d8hzK9Uiw2TyYEXZteW5ICdwtOPM0Qxa_IDfQYIlh7dVE77H0w-ffoF8=w489-h652-no",
      "https://mindseye-art-7f168.s3.us-west-2.amazonaws.com/reports/20180421230041/etc/com.simiacryptus.mindseye.style_transfer.Enlarging.8.png"
  };

  public void accept(@Nonnull NotebookOutput log) {
    StyleTransfer.Inception styleTransfer = new StyleTransfer.Inception();
    Precision precision = Precision.Float;
    styleTransfer.parallelLossFunctions = true;
    styleTransfer.setTiled(false);
    for (int i = 0; i < contentSources.length; i++) {
      String contentSource = contentSources[i];
      String contentSeed = contentSeeds[i];
      log.p("Content Source:");
      log.p(log.png(ArtistryUtil.load(contentSource, resolution), "Content Image"));
      log.p("Style Source:");
      for (final CharSequence styleSource : styleSources) {
        log.p(log.png(ArtistryUtil.load(styleSource, resolution), "Style Image"));
      }
      final Map<List<CharSequence>, StyleTransfer.StyleCoefficients<CVPipe_Inception.Strata>> styles =
          TestUtil.buildMap(x -> x.put(
              Arrays.asList(styleSources),
              new StyleTransfer.StyleCoefficients<CVPipe_Inception.Strata>(
                  StyleTransfer.CenteringMode.Origin)
                  .set(
                      CVPipe_Inception.Strata.Layer_0,
                      coeff_style_mean,
                      coeff_style_cov,
                      dreamCoeff
                  )
                  .set(
                      CVPipe_Inception.Strata.Layer_2,
                      coeff_style_mean,
                      coeff_style_cov,
                      dreamCoeff
                  )
                  .set(
                      CVPipe_Inception.Strata.Layer_1b,
                      coeff_style_mean,
                      coeff_style_cov,
                      dreamCoeff
                  )
                  .set(
                      CVPipe_Inception.Strata.Layer_1c,
                      coeff_style_mean,
                      coeff_style_cov,
                      dreamCoeff
                  )
          ));
      Tensor canvasImage = ArtistryUtil.loadTensor(contentSource, resolution);
      Tensor canvasSeed = Tensor.fromRGB(TestUtil.resize(
          HadoopUtil.getImage(contentSeed),
          canvasImage.getDimensions()[0],
          canvasImage.getDimensions()[1]
      ));
      StyleTransfer.StyleSetup<CVPipe_Inception.Strata> styleSetup = new StyleTransfer.StyleSetup<CVPipe_Inception.Strata>(
          precision,
          null,
          new StyleTransfer.ContentCoefficients<CVPipe_Inception.Strata>()
              .set(CVPipe_Inception.Strata.Layer_2, contentMixingCoeff * 1e-1)
              .set(CVPipe_Inception.Strata.Layer_1c, contentMixingCoeff)
              .set(CVPipe_Inception.Strata.Layer_1d, contentMixingCoeff),
          TestUtil.buildMap(y -> y.putAll(styles.keySet().stream().flatMap(x1 -> x1.stream()).collect(
              Collectors.toMap(x1 -> x1, file -> ArtistryUtil.load(file, resolution))))),
          styles
      );
      styleTransfer(log, styleTransfer, canvasImage, canvasSeed, styleSetup, 600, 600, 600, 600);
    }
  }

  public Tensor styleTransfer(
      @Nonnull final NotebookOutput log,
      final StyleTransfer.Inception styleTransfer,
      final Tensor canvasImage,
      final Tensor canvasSeed,
      final StyleTransfer.StyleSetup<CVPipe_Inception.Strata> styleSetup,
      final int width,
      final int height,
      final int strideX,
      final int strideY
  ) {
    if (canvasImage.getDimensions()[0] != canvasSeed.getDimensions()[0])
      throw new AssertionError(canvasImage.getDimensions()[0] + " != " + canvasSeed.getDimensions()[0]);
    if (canvasImage.getDimensions()[1] != canvasSeed.getDimensions()[1])
      throw new AssertionError(canvasImage.getDimensions()[1] + " != " + canvasSeed.getDimensions()[1]);
    int cols = (int) (Math.ceil((canvasImage.getDimensions()[0] - width) * 1.0 / strideX) + 1);
    int rows = (int) (Math.ceil((canvasImage.getDimensions()[1] - height) * 1.0 / strideY) + 1);
    Tensor resultImage;
    if (cols > 1 || rows > 1) {
      Tensor[] canvasTiles = ImgTileSelectLayer.toTiles(log, canvasImage, width, height, strideX, strideY, 0, 0);
      Tensor[] canvasSeeds = ImgTileSelectLayer.toTiles(log, canvasSeed, width, height, strideX, strideY, 0, 0);
      if (canvasTiles.length != canvasSeeds.length)
        throw new AssertionError(canvasTiles.length + " != " + canvasSeeds.length);
      Stream<Tensor> tensorStream = IntStream.range(0, canvasTiles.length).mapToObj(i -> {
        StyleTransfer.StyleSetup<CVPipe_Inception.Strata> tileSetup = new StyleTransfer.StyleSetup<>(
            styleSetup.precision,
            canvasTiles[i],
            styleSetup.content,
            styleSetup.styleImages,
            styleSetup.styles
        );
        StyleTransfer<CVPipe_Inception.Strata, CVPipe_Inception>.NeuralSetup measureStyle = styleTransfer.measureStyle(tileSetup);
        Tensor transfer = styleTransfer.transfer(
            log,
            canvasSeeds[i],
            tileSetup,
            getTrainingMinutes(),
            measureStyle,
            getMaxIterations(),
            isVerbose()
        );
        return transfer;
      });
      Tensor[] resultTiles = tensorStream.toArray(i -> new Tensor[i]);
      resultImage = new ImgTileAssemblyLayer(cols, rows).eval(resultTiles).getData().get(0);
      log.p("Assembled Result:");
      log.p(log.png(resultImage.toImage(), "Assembled Canvas"));
      return resultImage;
    } else {
      return styleTransfer.transfer(log, canvasImage, styleSetup,
          getTrainingMinutes(), styleTransfer.measureStyle(styleSetup), getMaxIterations(), isVerbose()
      );
    }
  }

  public static class Local {
    public static void main(String... args) throws Exception {
      LocalNotebookRunner.run(LocalNotebookRunner.getTask(HiDef.class));
    }
  }

  public static class EC2 {
    public static void main(String... args) throws Exception {
      EC2NotebookRunner.run(LocalNotebookRunner.getTask(HiDef.class));
    }
  }

}
