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

import com.amazonaws.auth.DefaultAWSCredentialsProviderChain;
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.fs.s3a.S3AFileSystem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.Iterator;
import java.util.List;
import java.util.Spliterator;
import java.util.Spliterators;
import java.util.stream.Collectors;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

/**
 * The type Hadoop util.
 */
public class HadoopUtil {

  private static final Logger logger = LoggerFactory.getLogger(HadoopUtil.class);

  /**
   * Gets files.
   *
   * @param file the file
   * @return the files
   */
  public static List<CharSequence> getFiles(CharSequence file) {
    try {
      FileSystem fileSystem = getFileSystem(file);
      Path path = new Path(file.toString());
      if (!fileSystem.exists(path)) throw new IllegalStateException(path + " does not exist");
      List<CharSequence> collect = toStream(fileSystem.listFiles(path, false))
          .map(FileStatus::getPath).map(Path::toString).collect(Collectors.toList());
      collect.stream().forEach(child -> {
        try {
          if (!fileSystem.exists(new Path(child.toString())))
            throw new IllegalStateException(child + " does not exist");
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      });
      return collect;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  /**
   * To stream stream.
   *
   * @param <T>            the type parameter
   * @param remoteIterator the remote iterator
   * @return the stream
   */
  @Nonnull
  public static <T> Stream<T> toStream(final RemoteIterator<T> remoteIterator) {
    return StreamSupport.stream(Spliterators.spliterator(new Iterator<T>() {
      @Override
      public boolean hasNext() {
        try {
          return remoteIterator.hasNext();
        } catch (Throwable e) {
          logger.warn("Error listing files", e);
          return false;
        }
      }

      @Override
      public T next() {
        try {
          return remoteIterator.next();
        } catch (IOException e) {
          throw new RuntimeException(e);
        }
      }
    }, -1, Spliterator.IMMUTABLE), true);
  }

  /**
   * Gets png.
   *
   * @param file the file
   * @return the png
   */
  @Nonnull
  public static BufferedImage getImage(final CharSequence file) {
    if (file.toString().startsWith("http")) {
      try {
        BufferedImage read = ImageIO.read(new URL(file.toString()));
        if (null == read) throw new IllegalArgumentException("Error reading " + file);
        return read;
      } catch (Throwable e) {
        throw new RuntimeException("Error reading " + file, e);
      }
    }
    FileSystem fileSystem = getFileSystem(file.toString());
    Path path = new Path(file.toString());
    try {
      if (!fileSystem.exists(path)) throw new IllegalArgumentException("Not Found: " + path);
      try (FSDataInputStream open = fileSystem.open(path)) {
        byte[] bytes = IOUtils.toByteArray(open);
        try (ByteArrayInputStream in = new ByteArrayInputStream(bytes)) {
          return ImageIO.read(in);
        }
      }
    } catch (Throwable e) {
      throw new RuntimeException("Error reading " + file, e);
    }
  }

  /**
   * Get data byte [ ].
   *
   * @param file the file
   * @return the byte [ ]
   */
  public static byte[] getData(final CharSequence file) {
    FileSystem fileSystem = getFileSystem(file);
    Path path = new Path(file.toString());
    try {
      if (!fileSystem.exists(path)) throw new IllegalArgumentException("Not Found: " + path);
      try (FSDataInputStream open = fileSystem.open(path)) {
        return IOUtils.toByteArray(open);
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public static FSDataOutputStream write(final CharSequence file) {
    FileSystem fileSystem = getFileSystem(file);
    Path path = new Path(file.toString());
    try {
      return fileSystem.create(path);
    } catch (IOException e) {
      throw new RuntimeException(String.format("Error writing %s", file),e);
    }
  }

  /**
   * Gets file system.
   *
   * @param file the file
   * @return the file system
   */
  public static FileSystem getFileSystem(final CharSequence file) {
    Configuration conf = getHadoopConfig();
    FileSystem fileSystem;
    try {
      fileSystem = FileSystem.get(new Path(file.toString()).toUri(), conf);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return fileSystem;
  }

  /**
   * Gets hadoop config.
   *
   * @return the hadoop config
   */
  @Nonnull
  public static Configuration getHadoopConfig() {
    Configuration configuration = new Configuration(false);

    File tempDir = new File("temp");
    tempDir.mkdirs();
    configuration.set("hadoop.tmp.dir", tempDir.getAbsolutePath());
//    configuration.set("fs.http.impl", org.apache.hadoop.fs.http.HttpFileSystem.class.getCanonicalName());
//    configuration.set("fs.https.impl", org.apache.hadoop.fs.http.HttpsFileSystem.class.getCanonicalName());
    configuration.set("fs.git.impl", com.simiacryptus.hadoop_jgit.GitFileSystem.class.getCanonicalName());
    configuration.set("fs.s3a.impl", S3AFileSystem.class.getCanonicalName());
    configuration.set("fs.s3.impl", S3AFileSystem.class.getCanonicalName());
    configuration.set("fs.s3a.aws.credentials.provider", DefaultAWSCredentialsProviderChain.class.getCanonicalName());
    return configuration;
  }
}
