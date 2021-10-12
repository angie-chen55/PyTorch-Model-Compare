# PyTorch Model Compare (WIP)

A tiny package to compare two neural networks in PyTorch. There are many ways to compare two neural networks, but one robust and scalable way is using the **Centered Kernel Alignment** (CKA) metric, where the features of the networks are compared.

### Centered Kernel Alignment
Centered Kernel Alignment (CKA) is a representation similarity metric that is widely used for understanding the representations learned by neural networks. Specifically, CKA takes two feature maps / representations **X** and **Y** as input and computes their normalized similarity (in terms of the Hilbert-Schmidt Independence Criterion (HSIC)) as

<p style="text-align:center;">
<img src="assets/cka.png" alt="CKA original version" width="70%">
</p>

However, the above formula is not scalable against deep architectures and large datasets. Therefore, a minibatch version can be constructed that uses an unbiased estimator of the HSIC as

![alt text](assets/cka_mb.png "CKA minibatch version")

![alt text](assets/cka_hsic.png "CKA HSIC calculation")

The above form of CKA is from the 2021 ICLR paper by [Nguyen T., Raghu M, Kornblith S](https://arxiv.org/abs/2010.15327).

## Getting Started

### Installation
```
pip install torch_cka
```
### Usage
```python
from torch_cka import CKA
model1 = resnet18(pretrained=True)
model2 = resnet34(pretrained=True)

dataloader = DataLoader(your_dataset, 
                        batch_size=batch_size, 
                        shuffle=False)

cka = CKA(model1, model2,
          device='cuda')

cka.compare(dataloader)
```

## Examples
`torch_cka` can be used with any pytorch model (subclass of `nn.Module`) and can be used with pretrained models available from popular sources like torchHub, timm, huggingface etc. Some examples of where this package can come in handy are illustrated below.

### Comparing ResNets

A simple experiment is to analyse the features learned by two architectures of the same family - ResNets. Taking two ResNets - ResNet18 and ResNet34 - pre-trained on the Imagenet dataset, we can analyse how they produce their features on, say CIFAR10 for simplicity. This comparison is shown as a heatmap below. 

<p style="text-align:center;">
<img src="assets/resnet_compare.png" alt="Comparing ResNet18 and ResNet34" width="75%">
</p>

We see high degree of similarity between the two models in lower layers as they both learn similar representations from the data. However at higher layers, the similarity reduces as the deeper model (ResNet34) learn higher order features which the is elusive to the shallower model (ResNet18). Yet, they do indeed have certain similarity in their last fc layer which acts as the feature classifier.

### Comparing Two Similar Architectures
Another way of using CKA is in ablation studies. We can go further than those ablation studies that only focus on resultant performance and employ CKA to study the internal representations.

### Comparing a ResNet with Vision Transformer (ViT)
CNNs have been analysed a lot over the past decade since AlexNet. We somewhat know what sort of features they learn across their layers (through visualizations) and we have put them to good use. One interesting approach is to compare these understandable features with newer models that don't permit easy visualizations (like recent vision transformer architectures) and study them. This has indeed been a hot research topic (see [Raghu et.al 2021](https://arxiv.org/abs/2108.08810)).

### Comparing Dataset Drift 
Yet another application is to compare two datasets - preferably two versions of the data. This is especially useful in production where data drift is a known issue. If you have an updated version of a dataset, you can study how your model will perform on it by comparing the representations of the datasets. This can be more telling about actual performance than simply comparing the datasets directly.

## Tips
- If your model is large (lots of layers or large feature maps), try to extract from select layers. This is to avoid out of memory issues. 
- If you still want to compare the entire feature map, you can run it multiple times with few layers at each iteration and export your data using `cka.export()`. The exported data can then be concatenated to produce the full CKA matrix.
- Give proper model names to avoid confusion when interpreting the results. The code automatically extracts the model name for you by default, but it is good practice to label the models according to your use case.






