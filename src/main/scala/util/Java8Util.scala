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

import java.util.function.{BiFunction, BinaryOperator, Consumer, DoubleSupplier, DoubleUnaryOperator, Function, IntToDoubleFunction, Supplier, ToDoubleBiFunction, ToDoubleFunction}

import com.simiacryptus.lang.UncheckedSupplier

object Java8Util {

  implicit def cvt(fn: Int ⇒ Double): IntToDoubleFunction = {
    new IntToDoubleFunction {
      override def applyAsDouble(v: Int): Double = fn(v)
    }
  }

  implicit def cvtUnchecked[T <: AnyRef](fn: () ⇒ T): UncheckedSupplier[T] = {
    new UncheckedSupplier[T] {
      override def get(): T = fn.apply()
    }
  }

  implicit def cvt[T <: AnyRef](fn: () ⇒ T): Supplier[T] = {
    new Supplier[T] {
      override def get(): T = fn.apply()
    }
  }

  implicit def cvt(fn: () ⇒ Unit): Runnable = {
    new Runnable {
      override def run(): Unit = fn.apply()
    }
  }

  implicit def cvt[T <: AnyRef, U <: AnyRef, R <: AnyRef](fn: (T, U) ⇒ R): BiFunction[T, U, R] = {
    new BiFunction[T, U, R] {
      override def apply(t: T, u: U) = fn.apply(t, u)
    }
  }

  implicit def cvt[T <: AnyRef](fn: (T, T) ⇒ T): BinaryOperator[T] = {
    new BinaryOperator[T] {
      override def apply(t: T, u: T) = fn.apply(t, u)
    }
  }

  implicit def cvt[T <: AnyRef](fn: () ⇒ Double): DoubleSupplier = {
    new DoubleSupplier {
      override def getAsDouble: Double = fn.apply()
    }
  }

  implicit def cvt[T](fn: T ⇒ Double): ToDoubleFunction[T] = {
    new ToDoubleFunction[T] {
      override def applyAsDouble(v: T): Double = fn(v)
    }
  }

  implicit def cvt(fn: Double ⇒ Double): DoubleUnaryOperator = {
    new DoubleUnaryOperator {
      override def applyAsDouble(v: Double): Double = fn(v)
    }
  }

  implicit def cvt[T](fn: T ⇒ Unit): Consumer[T] = {
    new Consumer[T] {
      override def accept(t: T): Unit = fn(t)
    }
  }

  implicit def cvt[T, U](fn: T ⇒ U): Function[T, U] = {
    new Function[T, U] {
      override def apply(v1: T): U = fn(v1)
    }
  }

  implicit def cvt[T, U](fn: (T, U) ⇒ Double): ToDoubleBiFunction[T, U] = {
    new ToDoubleBiFunction[T, U] {
      override def applyAsDouble(v: T, u: U): Double = fn(v, u)
    }
  }

}
