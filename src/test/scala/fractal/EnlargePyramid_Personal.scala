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

package fractal

object EnlargePyramid_Personal extends EC2Task[EnlargePyramid_Personal](classOf[EnlargePyramid_Personal]) {

}

class EnlargePyramid_Personal extends EnlargePyramid(
  aspect = .59353,
  localPrefix = "tile_1_",
  reportPath = "reports/20180812222258",
  bucket = "mindseye-art-7f168",
  styleSources = Array("s3a://simiacryptus/photos/shutterstock_1065730331.jpg")
)
