## Repository Introduction:

We have discovered the existence of unfairness in model quantization. This repository serves as an implementation method to uncover the phenomenon of quantization fairness, using the datasets UTKface, CIFAR-10, and MNIST as examples. It demonstrates the performance differences of model quantization on unfair and fair datasets, as well as the variations in fairness degradation caused by two different quantization methods: QAT and PTQ.

### Quantization

Quantization methods：QAT(Quantization-Aware Training) and PTQ(Post-Training Quantization)

Quantization bits: 16bits、8bits、4bits

## Code Structure:

The main code is in quantization/code/.

The main-cifar10.py file performs quantization experimental tests on the CIFAR10 dataset. Since the CIFAR10 dataset is already available in PyTorch, it can be downloaded during runtime. 

The main-mnist.py file conducts experiments on the MNIST dataset. The dataset can be downloaded during runtime as well.

The main-utkface.py file also performs quantization experimental tests on the UTK-Face dataset, which requires downloading the UTK-Face dataset separately. The UTK-Face dataset is located in "/data/age_gender.gz" file. Before running the code, it needs to be extracted and converted into a CSV file.

```
unfairness-in-model-quantization
├── __init__.py
├── quantization
│   ├── __init__.py
│   └── code
│       ├── __init__.py
│       ├── bn_fuse
│       │   ├── bn_fuse.py
│       │   ├── bn_fused_model_test.py
│       │   └── models_save
│       │       └── models_save.txt
│       ├── main_class.py
│       ├── main_dataset.py
│       ├── main_dataset.py
│       ├── CustomUTK.py
│       ├── MultNN.py
│       ├── models_save
│       │   └── models_save.txt
│       └── quantize.py
│           
├── data
│   ├── cifar-10-batches-py
│   └── MNIST
└── models
    ├── __init__.py
    ├── nin.py
    ├── nin_gc.py
    └── resnet.py

```

## Environmental requirements

- python >= 3.5
- torch >= 1.1.0
- torchvison >= 0.3.0
- numpy
- onnx == 1.6.0
- tensorrt == 7.0.0.11

## Parameter introduction

Since the code part of the reference comes from(https://github.com/666DZY666/micronet/blob/master/README.md?plain=1)

--q_type, Quantization type (0-symmetry, 1-asymmetry)

--q_level, Weight quantification level(0-Channel level, 1-Level)

--weight_observer, weight_observer (0-MinMaxObserver, 1-MovingAverageMinMaxObserver)

--bn_fuse, Quantify the BN fusion logo

--bn_fuse_calib, Quantify the BN fusion calibration flag

--pretrained_model, Pre-trained floating-point models

--ptq, ptq_observer

--ptq_control, ptq_control

--ptq_batch, The number of batches for PTQ

--percentile, Proportion of PTQ calibration

## Test

*Weight Attribute Quantization bits*

--w_bits --a_bits, Weight w and Attribute a Quantization bits

- W16A16

```bash
python main.py --w_bits 16 --a_bits 16
```

- W8A8

```bash
python main.py --w_bits 8 --a_bits 8
```

- W4A4

```bash
python main.py --w_bits 4 --a_bits 4
```

QAT

- Default: symmetric, (weighted) channel-level quantization, BN not fused, weight_observer-MinMaxObserver, no loading of pre-trained floating-point models, w_bits 8 , a_bits 8, QAT

```bash
python main.py --q_type 0 --q_level 0 --weight_observer 0
```

- Combined with WA

```bash
python main.py --w_bits 16 --a_bits 16 --q_type 0 --q_level 0 --weight_observer 0
```

PTQ

--refine，load the parameters of the pre-trained floating-point model and quantize on it

- Default: symmetric, (weighted) channel-level quantization, BN not fused, weight_observer-MinMaxObserver, w_bits 8 , a_bits 8, ptq

```bash
python main.py --refine ./models_save/nin_gc.pth --q_level 0 --bn_fuse --pretrained_model --ptq_control --ptq --batch_size 32
```

- Combined with WA

```bash
python main.py --refine ./models_save/nin_gc.pth --q_level 0 --bn_fuse --pretrained_model --ptq_control --ptq --batch_size 32 --w_bits 16 --a_bits 16
```

