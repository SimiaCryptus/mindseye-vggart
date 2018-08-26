import com.simiacryptus.aws.exe.EC2NodeSettings
import com.simiacryptus.sparkbook._
import fractal._

val pyramidInit: InitialPyramid = new InitialPyramid(
  initialContent = "https://mindseye-art-7f168.s3.us-west-2.amazonaws.com/reports/20180824155832/etc/fractal.InitialPainting_Personal.6.png",
  styleSources = Array("s3a://simiacryptus/photos/shutterstock_1065730331.jpg")
) with EC2Runner with AWSNotebookRunner {
  override def nodeSettings = EC2NodeSettings.DeepLearningAMI
}.exe();






