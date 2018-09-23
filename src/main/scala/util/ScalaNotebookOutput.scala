/*
 * Copyright (c) 2018 by Andrew Charneski.
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

package util

import java.awt.image.BufferedImage
import java.awt.{Graphics2D, RenderingHints}

import com.simiacryptus.lang.UncheckedSupplier
import com.simiacryptus.notebook.NotebookOutput

/**
  * Created by Andrew Charneski on 5/14/2017.
  */
trait ScalaNotebookOutput extends NotebookOutput {

  private val default_max_log: Int = 64 * 1024

  def eval[T](fn: => T): T = {
    eval(new UncheckedSupplier[T] {
      override def get(): T = fn
    }, default_max_log, 4)
  }

  def code[T](fn: () => T): T = {
    eval(new UncheckedSupplier[T] {
      override def get(): T = fn()
    }, default_max_log, 4)
  }

  def draw[T](fn: (Graphics2D) ⇒ Unit, width: Int = 600, height: Int = 400): BufferedImage = {
    eval(new UncheckedSupplier[BufferedImage] {
      override def get(): BufferedImage = {
        val image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
        val graphics = image.getGraphics.asInstanceOf[Graphics2D]
        graphics.asInstanceOf[Graphics2D].setRenderingHints(new RenderingHints(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BICUBIC))
        fn(graphics)
        image
      }
    }, default_max_log, 4)
  }

}
