# alexnet.py Notes
This file is to serve as general notes from my experience writing the code.

## CNN Notes

Convolution is a mathematical operation that merge two sets of information: input data (images) and a small matrix called a filter or kernel.

In the case of CNNs, convolution involves sliding this filter across the input data and computing a dot product between the filter and local regions of the input.

Resulting data is a new matrix called a feature/activation map.

Functionally:
- Filter (3x3 matrix) moves/slides across the input image
- At each position, the filter multiplies its values with the corresponding values in the image patch (Matrix-Matrix) and sums the result
- The sum is placed in the corresponding position of the feature map
- Filter continues to slide over the entire image repeating the process and produces a complete feature map.

Key Points:
- Filters/Kernels are small matrices that are learned during training to detect specific features
- Feature maps are the output of the convolution operation, indicating where certain features are present in the input
- Hierachical Feature Extraction where early layers detect simple features (edges/textures) while deeper layers combine these into more complex patterns like shapes and objects
- Convolution reduces the number of parameters and computations, making deeper networks feasible for large-scale data sets

## Alex Net Architecture Notes

Alex Net (AN) consists of five convolutional layers that apply filters (kernels) to input images to extract features.

A "filter operation" is the sliding of a kernel over the input image, computing the dot products at each position producing the feature maps.

Naturally this extends itself towards GPU parallelizability!
- We can compute output pixels in a feature map independently as we need only consider certain regions of the input
- We can compute multiple featuire maps simultaneously
- We can compute batches of images and each image processed in parallel
- Every pooling operation can be independent

The fully connected layers sit at 60M parameters and are the prime target for parallelization.

Max pooling after conv. layers to downsample feature maps which reduce spatial dimensions and preserve important features.

Local response normalization after first and second conv. layers to normalize neuron activities.

Dropout randomly sets the output of hidden neurons to zero with prob 0.5 to avoid overfitting.
