# HCGN

Hybrid CNN-GCN Network for Hyperspectral Image Classification

Abstract: In recent years, convolutional neural networks (CNNs) have been impressive due to their excellent feature representation abilities, but it is difficult to learn long-distance spatial structures information. Unlike CNN, graph convolutional networks (GCNs) can well handle the intrinsic manifold structures of hyperspectral images (HSIs). However, the existing GCN-based classification methods do 
not fully utilize the edge relationship, which makes their performance is limited. In addition, a small number of training samples is also a reason for hindering high-performance hyperspectral image classification. Therefore, this paper proposes a hybrid CNN-GCN network (HCGN) for hyperspectral image classification. Firstly, a graph edge enhanced module (GEEM) is designed to enhance the superpixel-level features of graph edge nodes and improve the spatial discrimination ability of ground objects. In particular, considering multiscale information is complementary, a multiscale graph edge enhanced module (MS-GEEM) based on GEEM is proposed to fully utilize texture structures of different sizes. Then, in order to enhance the pixel-level multi hierarchical fine feature representation of images, a multiscale cross fusion module (MS-CFM) based on the CNN framework is proposed. Finally, the extracted pixel-level features and superpixel-level features are cascaded. Through a series of experiments, it has been proved that compared with some state-of-the-art methods, HCGN combines the advantages of CNN and GCN frameworks, can provide superior classification performance under limited training samples, and demonstrates the advantages and great potential of HCGN.

Environment: 
Python 3.8
PyTorch 1.10

How to use it?
---------------------
This toolbox consists of two proposed branchs, i.e., Strategy for Extracting Spectral Features Based on MSSP and Strategy for Extracting Spatial Features, that can be plug-and-played into both pixel-wise and patch-wise hyperspectral image classification. For more details, please refer to the paper.

Here an example experiment is given by using **Indian Pines hyperspectral data**. Here is an example experiment using hyperspectral data of Indian pine trees. Set parameters in the **Main. py** function: **FLAG=1, current_ Train_ Ratio=0.01, Scale=50, Scale1=200, Scale2=300)**, run directly to generate the results. Please note that due to the randomness of parameter initialization, the experimental results may differ slightly from the results reported in the paper.

If you want to run the code in your own data, you can accordingly change the input (e.g., data, labels) and tune the parameters.

If you encounter the bugs while using this code, please do not hesitate to contact us.

If emergency, you can also add my email: liaodiling2020@163.com or QQ: 3097264896.

