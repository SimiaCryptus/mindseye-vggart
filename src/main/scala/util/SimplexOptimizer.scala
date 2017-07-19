package interactive.superres

import org.apache.commons.math3.analysis.MultivariateFunction
import org.apache.commons.math3.optim._
import org.apache.commons.math3.optim.nonlinear.scalar._
import org.apache.commons.math3.optim.nonlinear.scalar.noderiv._

import scala.reflect.runtime.universe
import scala.reflect.runtime.universe._

trait SimplexOptimization {
  def optimize(fn : this.type ⇒ Double)(implicit ev1 : scala.reflect.ClassTag[this.type]) : this.type = {
    SimplexOptimizer.apply[this.type](this, fn)
  }
}

object SimplexOptimizer {

  def apply[T:TypeTag](initial : T, fitnessFunction: T ⇒ Double)(implicit ev1 : scala.reflect.ClassTag[T]): T = {
    val classSymbol = typeOf[T].typeSymbol.asClass
    val classMirror = scala.reflect.runtime.currentMirror.reflectClass(classSymbol)
    val primaryConstructor = classSymbol.primaryConstructor.asMethod
    val constructorMirror = classMirror.reflectConstructor(primaryConstructor)
    val factory : Array[Double] ⇒ T = args ⇒ {
      constructorMirror(args:_*).asInstanceOf[T]
    }
    val toArray : T ⇒ Array[Double] = obj ⇒ {
      primaryConstructor.paramLists.head.map(arg ⇒ {
        val reflect: universe.InstanceMirror = scala.reflect.runtime.currentMirror.reflect(obj)
        reflect.reflectField(typeOf[T].decl(arg.name).asTerm).get.asInstanceOf[Double]
      }).toArray
    }
    val optimizer = new SimplexOptimizer(1e-2, 1e-2)
    val dimensions = toArray(initial).length
    val optimalMetaparameters = factory(optimizer.optimize(
      new ObjectiveFunction(new MultivariateFunction {
        override def value(doubles: Array[Double]): Double = {
          fitnessFunction(factory(doubles))
        }
      }),
      new InitialGuess(toArray(initial)),
      GoalType.MINIMIZE, new MaxIter(1000), new MaxEval(1000),
      new MultiDirectionalSimplex(dimensions)
    ).getPoint)
    optimalMetaparameters
  }

}
