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

import interactive.superres.SimplexOptimizer
import org.apache.commons.math3.analysis.MultivariateFunction
import org.apache.commons.math3.optim._
import org.apache.commons.math3.optim.nonlinear.scalar.{noderiv, _}
import org.apache.commons.math3.random.{JDKRandomGenerator, RandomGenerator}

import scala.reflect.runtime.universe
import scala.reflect.runtime.universe._


trait CMAESOptimization {
  def optimize(fn: this.type ⇒ Double)(implicit ev1: scala.reflect.ClassTag[this.type]): this.type = {
    SimplexOptimizer.apply[this.type](this, fn)
  }
}

object CMAESOptimizer {

  def apply[T: TypeTag](initial: T,
                        fitnessFunction: T ⇒ Double,
                        maxIterations: Int = 1000,
                        stdDev: Double = 0.2,
                        min: Double = 0.0,
                        max: Double = 1.0,
                        stopFitness: Double = 0.0,
                        population: Int = 5
                       )(implicit ev1: scala.reflect.ClassTag[T]): T = {
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

    val isActiveCMA: Boolean = true
    val diagonalOnly: Int = 0
    val checkFeasableCount: Int = 1
    val random: RandomGenerator = new JDKRandomGenerator
    val generateStatistics: Boolean = false
    val dimensions: Int = toArray(initial).length

    val checker: ConvergenceChecker[PointValuePair] = new ConvergenceChecker[PointValuePair] {
      override def converged(iteration: Int, previous: PointValuePair, current: PointValuePair): Boolean = {
        false
      }
    }
    val optimizer = new noderiv.CMAESOptimizer(
      maxIterations, stopFitness, isActiveCMA, diagonalOnly, checkFeasableCount, random, generateStatistics, checker
    )
    val optimalMetaparameters = factory(optimizer.optimize(
      new ObjectiveFunction(new MultivariateFunction {
        override def value(doubles: Array[Double]): Double = {
          fitnessFunction(factory(doubles))
        }
      }),
      new InitialGuess(toArray(initial)),
      GoalType.MINIMIZE, new MaxIter(maxIterations), new MaxEval(maxIterations),
      new noderiv.CMAESOptimizer.Sigma((0 until dimensions).map(d ⇒ stdDev).toArray),
      new noderiv.CMAESOptimizer.PopulationSize(population),
      new SimpleBounds(
        (0 until dimensions).map(d ⇒ min).toArray,
        (0 until dimensions).map(d ⇒ max).toArray
      )
    ).getPoint)
    optimalMetaparameters
  }

}
