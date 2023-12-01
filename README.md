# Train ResUNet-a D6 (Waldner and Diakogiannis, 2020) for crop field delineation on the AWS SageMaker


ResUNet-a D6 (<a href="https://doi.org/10.1016/j.rse.2020.111741">Waldner and Diakogiannis, 2020</a>) is the combination UNet, ResNet, and dilated convolution, which can perform skillful crop field delineation. In particular, it leverages the idea of multi-task training, which enhance the learning performance by letting the model to learn from multiple relevant tasks. In this case, the model learn to depict the field boundaries along with field extents, dstance to the boundaries, and HSV.


Here, I used AI4Boundaries dataset (<a href="https://doi.org/10.5194/essd-15-317-2023">d'Andrimont et al., 2023</a>) for the model training, validation, and test. A tutorial of how to train/tune a TensorFlow model on the AWS SageMaker can be seen <a href="https://aws.amazon.com/tutorials/train-tune-deep-learning-model-amazon-sagemaker/">Here</a>
