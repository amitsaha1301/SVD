# SVD
# Singular Value Decomposition (SVD)

Singular Value Decomposition (SVD) is a mathematical technique widely used in linear algebra, data science, and machine learning. It is a method of factorizing a matrix into three other matrices, revealing its fundamental structure. SVD is particularly useful in tasks like dimensionality reduction, noise reduction, and understanding the latent structure of data.

## Theoretical Concept

For any real or complex matrix **A** of size \( m \times n \), SVD decomposes it into three matrices:

\[
A = U \Sigma V^T
\]

Where:
- **\( U \)**: An \( m \times m \) orthogonal (or unitary) matrix. The columns of \( U \) are called the **left singular vectors**.
- **\( \Sigma \)**: An \( m \times n \) diagonal matrix with non-negative real numbers on the diagonal. These values are called **singular values** and are ordered in descending order (\( \sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r \geq 0 \)).
- **\( V^T \)**: The transpose of an \( n \times n \) orthogonal (or unitary) matrix. The columns of \( V \) are called the **right singular vectors**.

## Properties of SVD

1. **Orthogonality**: The matrices \( U \) and \( V \) are orthogonal (or unitary in the complex case), meaning:
   \[
   U^T U = I \quad \text{and} \quad V^T V = I
   \]

2. **Singular Values**: The diagonal entries of \( \Sigma \) are the square roots of the eigenvalues of \( A^T A \) (or \( AA^T \)).

3. **Rank**: The number of non-zero singular values equals the rank of \( A \).

4. **Low-Rank Approximation**: By truncating the smaller singular values in \( \Sigma \), you can create a lower-rank approximation of \( A \), which is useful for dimensionality reduction.

## Geometric Interpretation

SVD provides a geometric interpretation of how a matrix transforms space. Specifically:
- \( U \) represents a rotation or reflection in the input space.
- \( \Sigma \) scales the transformed data along orthogonal axes.
- \( V^T \) represents a rotation or reflection in the output space.

This decomposition allows us to understand how data is stretched, compressed, or rotated when transformed by the matrix \( A \).

## Applications of SVD

1. **Dimensionality Reduction**:
   - Principal Component Analysis (PCA) uses SVD to reduce the dimensionality of data by retaining only the largest singular values.

2. **Data Compression**:
   - Images, videos, and other data types can be compressed using SVD by retaining the most significant singular values and their corresponding vectors.

3. **Noise Reduction**:
   - By removing smaller singular values, noise can be filtered out while preserving the primary structure of the data.

4. **Matrix Inversion**:
   - SVD provides a stable way to compute the pseudoinverse of a matrix, even for non-square or rank-deficient matrices.

5. **Collaborative Filtering**:
   - Recommender systems, such as those used in Netflix or Amazon, leverage SVD to identify latent features in user-item interaction matrices.

## SVD in Practice

Here is a quick example using Python's `numpy` library:

```python
import numpy as np

# Example matrix
A = np.array([[1, 2], [3, 4], [5, 6]])

# Compute SVD
U, Sigma, VT = np.linalg.svd(A)

print("Matrix A:")
print(A)

print("\nLeft singular vectors (U):")
print(U)

print("\nSingular values (Sigma):")
print(Sigma)

print("\nRight singular vectors (V^T):")
print(VT)
```

## References

- [Linear Algebra Textbooks](https://linear-algebra.org/)
- [Numpy Documentation](https://numpy.org/doc/stable/)
- [Singular Value Decomposition - Wikipedia](https://en.wikipedia.org/wiki/Singular_value_decomposition)


