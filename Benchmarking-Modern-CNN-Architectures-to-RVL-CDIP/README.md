# Benchmarking-Modern-CNN-Architectures-to-RVL-CDIP

For questions please contact [Jonathan DeGange](mailto:jdegange85@gmail.com)

This is the associated repository with Medium article [Benchmarking Modern CNN Architectures to RVLCDIP](https://medium.com/@jdegange85/benchmarking-modern-cnn-architectures-to-rvl-cdip-9dd0b7ec2955)

### Select Experiments

| ModelName          | Batch Size | Test Accuracy | Steps(K) | Image Size (^2) | Optimizer | LR   | Cutout |
|--------------------|------------|---------------|----------|-----------------|-----------|------|--------|
| EfficientNetB4     | 64         | 0.9281        | 500      | 380             | SGD       | 0.01 | Y      |
| IncerptionResNetV2 | 16         | 0.9263        | 250      | 512             | SGD       | 0.1  | N      |
| EfficientNetB2     | 64         | 0.9157        | 500      | 260             | SGD       | 0.01 | Y      |
| EfficientNetB0     | 64         | 0.9053        | 500      | 224             | SGD       | 0.01 | Y      |
| EfficientNetB0     | 32         | 0.9036        | 247.5    | 224             | SGD       | 0.01 | Y      |
| EfficientNetB0     | 32         | 0.8983        | 14       | 224             | SGD       | 0.01 | Y      |
| EfficientNetB0     | 64         | 0.8951        | 100      | 224             | SGD       | 0.01 | Y      |
| EfficientNetB0     | 32         | 0.8921        | 192.5    | 224             | Adadelta  | 1    | Y      |

### Dataset: 
To download the RVL-CDIP dataset used in the study, please use the original author's provided [link](http://www.cs.cmu.edu/~aharley/rvl-cdip/).

## Data augmentation strategy

We used cutout and mixup augmentation strategies together with Python image generator producing the augmentation on the fly during training on CPU.

![alt text][logo]

[logo]: https://github.com/jdegange/Benchmarking-Modern-CNN-Architectures-to-RVL-CDIP/raw/master/cutout_examples.png "Logo Title Text 2"

### Reference
#To be updated
Several repositories were used to construct this repo. [Keras Applications][1], 
[classification_models][2], provided very helpful Keras Tensorflow implementations for many of the referenced architectures. Thank you. Stochastic weight averaging was performed using the helpful implementation using @kristpapadopoulos's constant learning rate implementation, some some light modification [repo][3].
