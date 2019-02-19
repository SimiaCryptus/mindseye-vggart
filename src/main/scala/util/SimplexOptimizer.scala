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

package interactive.superres

import org.apache.commons.math3.analysis.MultivariateFunction
import org.apache.commons.math3.optim._
import org.apache.commons.math3.optim.nonlinear.scalar._
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv._

import scala.reflect.runtime.universe
import scala.reflect.runtime.universe._

trait SimplexOptimization {
  def optimize(fn: this.type ⇒ Double)(implicit ev1: scala.reflect.ClassTag[this.type]): this.type = {
    SimplexOptimizer.apply[this.type](this, fn)
  }
}

object SimplexOptimizer {

  def apply[T: TypeTag](initial: T,
                        fitnessFunction: T ⇒ Double,
                        maxIterations: Int = 1000,
                        relativeTolerance: Double = 1e-2,
                        absoluteTolerance: Double = 1
                       )(implicit ev1: scala.reflect.ClassTag[T]): T = {
    val (factory, toArray) = buildAdapters[T]
    val optimizer = new SimplexOptimizer(relativeTolerance, absoluteTolerance)
    optimizer.getConvergenceChecker
    val dimensions = toArray(initial).length
    val optimalMetaparameters = factory(optimizer.optimize(
      new ObjectiveFunction(new MultivariateFunction {
        override def value(doubles: Array[Double]): Double = {
          fitnessFunction(factory(doubles))
        }
      }),
      new InitialGuess(toArray(initial)),
      GoalType.MINIMIZE, new MaxIter(maxIterations), new MaxEval(maxIterations),
      new MultiDirectionalSimplex(dimensions)
      //      new SimpleBounds(
      //        (0 until dimensions).mapCoords(d ⇒ min).toArray,
      //        (0 until dimensions).mapCoords(d ⇒ max).toArray
      //      )
    ).getPoint)
    optimalMetaparameters
  }

  def buildAdapters[T: TypeTag](implicit ev1: scala.reflect.ClassTag[T]) = {
    val classSymbol = typeOf[T].typeSymbol.asClass
    val classMirror = scala.reflect.runtime.currentMirror.reflectClass(classSymbol)
    val primaryConstructor = classSymbol.primaryConstructor.asMethod
    val constructorMirror = classMirror.reflectConstructor(primaryConstructor)
    val factory: Array[Double] ⇒ T = args ⇒ {
      constructorMirror(args: _*).asInstanceOf[T]
    }
    val toArray: T ⇒ Array[Double] = obj ⇒ {
      primaryConstructor.paramLists.head.map(arg ⇒ {
        val reflect: universe.InstanceMirror = scala.reflect.runtime.currentMirror.reflect(obj)
        reflect.reflectField(typeOf[T].decl(arg.name).asTerm).get.asInstanceOf[Double]
      }).toArray
    }
    (factory, toArray)
  }
}
