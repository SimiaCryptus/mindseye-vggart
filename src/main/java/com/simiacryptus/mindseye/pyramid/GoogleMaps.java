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
import com.simiacryptus.util.binary.Bits;

import javax.annotation.Nonnull;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

public final class GoogleMaps {

  public enum Mars {
    Elevation("https://mw1.google.com/mw-planetary/mars/elevation/t"),
    Visible("https://mw1.google.com/mw-planetary/mars/visible/t"),
    Infrared("https://mw1.google.com/mw-planetary/mars/infrared/t");


    private final String prefix;

    Mars(final String prefix) {
      this.prefix = prefix;
    }

    @Nonnull
    public ValueSampler getValueSampler(final int level) {
      double aspect = 1.0;
      double tileSize = 256;
      Map<String, BufferedImage> cache = new HashMap<>();
      ValueSampler upperLevel = level > 1 ? getValueSampler(level - 1) : null;
      return (xf, yf, band) -> {
        while (xf < 0) xf += 1;
        while (yf < 0) yf += aspect;
        yf %= aspect;
        xf %= 1;
        long mag = (long) Math.pow(2, level);
        String xbits = Bits.divide((long) Math.floor(xf * mag), mag, level + 1).padRight(level + 1).range(1, level).toBitString();
        String ybits = Bits.divide((long) Math.floor(yf * mag), mag, level + 1).padRight(level + 1).range(1, level).toBitString();
        String code = IntStream.range(0, level).mapToObj(i -> {
          if (xbits.charAt(i) == '1') {
            if (ybits.charAt(i) == '1') {
              return "s";
            } else {
              return "r";
            }
          } else {
            if (ybits.charAt(i) == '1') {
              return "t";
            } else {
              return "q";
            }
          }
        }).reduce((a, b) -> a + b).get();
        int col = (int) Math.floor((xf * mag));
        int row = (int) Math.floor((yf * mag));
        int x = (int) ((xf * mag - col) * tileSize);
        int y = (int) ((yf * mag - row) * tileSize);
        String tilename = prefix + code + ".jpg";
        WritableRaster raster = cache.computeIfAbsent(tilename, s -> {
          try {
            return ArtistryUtil.load(s, -1);
          } catch (Throwable e) {
            e.printStackTrace();
            return null;
          }
        }).getRaster();
        if (raster == null) {
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

  }
}
