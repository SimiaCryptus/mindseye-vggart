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

import com.simiacryptus.mindseye.lang.LayerBase
import com.simiacryptus.mindseye.layers.java.MonitoringWrapperLayer
import com.simiacryptus.util.MonitoredObject

/**
  * Created by Andrew Charneski on 7/20/2017.
  */
object NNLayerUtil {
  implicit def cast(inner: LayerBase) = new NNLayerUtil(inner)
}

case class NNLayerUtil(inner: LayerBase) {
  def withMonitor = new MonitoringWrapperLayer(inner)
  def addTo(monitor: MonitoredObject) = withMonitor.addTo(monitor)
}