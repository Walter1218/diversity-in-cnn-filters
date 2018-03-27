# diversity-in-cnn-filters
Diversity promoting prior based on Determinantal Point Process over filters of a convolutional layer

### Similarity metric between filters
- Compute FT of kernels in all filters of a layer
- Take magnitude of FTs (because magnitude is translation invariant)
- Apply RBF kernel over resulting 3-tensors (assuming 3-tensor as a vector) to compute similarity between two filters
- Perform step 3 for all pairs to get symmetric positive semi-definite similarity matrix.


### Diversity promoting term
- With the similarity matrix `L` computed as above, we minimize `(log(det(L)+eps)-log(det(L+I)))^2`