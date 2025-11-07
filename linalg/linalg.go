// Package linalg provides linear algebra operations for NumGo arrays
package linalg

import (
	"fmt"
	"math"
	
	"github.com/iSundram/NumGo/tensor"
)

// Dot computes the dot product of two arrays
// For 1D arrays: sum of element-wise products
// For 2D arrays: matrix multiplication
func Dot(a, b *tensor.NDArray) *tensor.NDArray {
	if a.Ndim() == 1 && b.Ndim() == 1 {
		// Vector dot product
		if a.Size() != b.Size() {
			panic(fmt.Sprintf("arrays must have same length: %d vs %d", a.Size(), b.Size()))
		}
		
		sum := 0.0
		for i := 0; i < a.Size(); i++ {
			sum += a.GetFloat64(i) * b.GetFloat64(i)
		}
		
		return tensor.FromSliceFloat64([]float64{sum}, 1)
	}
	
	if a.Ndim() == 2 && b.Ndim() == 2 {
		// Matrix multiplication
		return MatMul(a, b)
	}
	
	if a.Ndim() == 2 && b.Ndim() == 1 {
		// Matrix-vector multiplication
		aShape := a.Shape()
		if aShape[1] != b.Size() {
			panic(fmt.Sprintf("dimension mismatch: (%d,%d) x (%d)", aShape[0], aShape[1], b.Size()))
		}
		
		result := tensor.Zeros([]int{aShape[0]}, tensor.Float64)
		for i := 0; i < aShape[0]; i++ {
			sum := 0.0
			for j := 0; j < aShape[1]; j++ {
				sum += a.GetFloat64(i, j) * b.GetFloat64(j)
			}
			result.SetFloat64(sum, i)
		}
		
		return result
	}
	
	panic(fmt.Sprintf("unsupported dimensions for dot: %dD and %dD", a.Ndim(), b.Ndim()))
}

// MatMul performs matrix multiplication
func MatMul(a, b *tensor.NDArray) *tensor.NDArray {
	if a.Ndim() != 2 || b.Ndim() != 2 {
		panic("MatMul requires 2D arrays")
	}
	
	aShape := a.Shape()
	bShape := b.Shape()
	
	if aShape[1] != bShape[0] {
		panic(fmt.Sprintf("dimension mismatch: (%d,%d) x (%d,%d)", aShape[0], aShape[1], bShape[0], bShape[1]))
	}
	
	m, n, p := aShape[0], aShape[1], bShape[1]
	result := tensor.Zeros([]int{m, p}, tensor.Float64)
	
	// Simple matrix multiplication (can be optimized with BLAS later)
	for i := 0; i < m; i++ {
		for j := 0; j < p; j++ {
			sum := 0.0
			for k := 0; k < n; k++ {
				sum += a.GetFloat64(i, k) * b.GetFloat64(k, j)
			}
			result.SetFloat64(sum, i, j)
		}
	}
	
	return result
}

// Outer computes the outer product of two vectors
func Outer(a, b *tensor.NDArray) *tensor.NDArray {
	if a.Ndim() != 1 || b.Ndim() != 1 {
		panic("Outer requires 1D arrays")
	}
	
	m, n := a.Size(), b.Size()
	result := tensor.Zeros([]int{m, n}, tensor.Float64)
	
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			result.SetFloat64(a.GetFloat64(i)*b.GetFloat64(j), i, j)
		}
	}
	
	return result
}

// Inner computes the inner product (same as dot for 1D)
func Inner(a, b *tensor.NDArray) float64 {
	if a.Ndim() != 1 || b.Ndim() != 1 {
		panic("Inner requires 1D arrays")
	}
	if a.Size() != b.Size() {
		panic("arrays must have same length")
	}
	
	sum := 0.0
	for i := 0; i < a.Size(); i++ {
		sum += a.GetFloat64(i) * b.GetFloat64(i)
	}
	
	return sum
}

// Norm computes the L2 (Euclidean) norm of a vector
func Norm(a *tensor.NDArray) float64 {
	sum := 0.0
	for i := 0; i < a.Size(); i++ {
		indices := a.Shape()
		if len(indices) == 1 {
			val := a.GetFloat64(i)
			sum += val * val
		} else {
			// For multi-dimensional arrays, flatten first
			idx := make([]int, a.Ndim())
			remaining := i
			for j := a.Ndim() - 1; j >= 0; j-- {
				idx[j] = remaining % a.Shape()[j]
				remaining /= a.Shape()[j]
			}
			val := a.GetFloat64(idx...)
			sum += val * val
		}
	}
	return math.Sqrt(sum)
}

// Trace computes the sum of diagonal elements
func Trace(a *tensor.NDArray) float64 {
	if a.Ndim() != 2 {
		panic("Trace requires a 2D array")
	}
	
	shape := a.Shape()
	n := shape[0]
	if shape[1] < n {
		n = shape[1]
	}
	
	sum := 0.0
	for i := 0; i < n; i++ {
		sum += a.GetFloat64(i, i)
	}
	
	return sum
}

// Det computes the determinant of a square matrix
// Using simple recursive method (can be optimized with LU decomposition)
func Det(a *tensor.NDArray) float64 {
	if a.Ndim() != 2 {
		panic("Det requires a 2D array")
	}
	
	shape := a.Shape()
	if shape[0] != shape[1] {
		panic("Det requires a square matrix")
	}
	
	n := shape[0]
	
	if n == 1 {
		return a.GetFloat64(0, 0)
	}
	
	if n == 2 {
		return a.GetFloat64(0, 0)*a.GetFloat64(1, 1) - a.GetFloat64(0, 1)*a.GetFloat64(1, 0)
	}
	
	// For larger matrices, use cofactor expansion (inefficient but simple)
	det := 0.0
	for j := 0; j < n; j++ {
		minor := getMinor(a, 0, j)
		cofactor := a.GetFloat64(0, j) * Det(minor)
		if j%2 == 0 {
			det += cofactor
		} else {
			det -= cofactor
		}
	}
	
	return det
}

// getMinor returns the minor matrix by removing row i and column j
func getMinor(a *tensor.NDArray, row, col int) *tensor.NDArray {
	shape := a.Shape()
	n := shape[0]
	
	data := make([]float64, 0, (n-1)*(n-1))
	for i := 0; i < n; i++ {
		if i == row {
			continue
		}
		for j := 0; j < n; j++ {
			if j == col {
				continue
			}
			data = append(data, a.GetFloat64(i, j))
		}
	}
	
	return tensor.FromSliceFloat64(data, n-1, n-1)
}

// Inv computes the inverse of a square matrix using Gauss-Jordan elimination
func Inv(a *tensor.NDArray) *tensor.NDArray {
	if a.Ndim() != 2 {
		panic("Inv requires a 2D array")
	}
	
	shape := a.Shape()
	if shape[0] != shape[1] {
		panic("Inv requires a square matrix")
	}
	
	n := shape[0]
	
	// Create augmented matrix [A | I]
	aug := tensor.Zeros([]int{n, 2 * n}, tensor.Float64)
	
	// Copy A to left side
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			aug.SetFloat64(a.GetFloat64(i, j), i, j)
		}
		// Identity on right side
		aug.SetFloat64(1.0, i, i+n)
	}
	
	// Gauss-Jordan elimination
	for i := 0; i < n; i++ {
		// Find pivot
		pivot := aug.GetFloat64(i, i)
		if math.Abs(pivot) < 1e-10 {
			panic("matrix is singular (not invertible)")
		}
		
		// Normalize pivot row
		for j := 0; j < 2*n; j++ {
			aug.SetFloat64(aug.GetFloat64(i, j)/pivot, i, j)
		}
		
		// Eliminate column
		for k := 0; k < n; k++ {
			if k == i {
				continue
			}
			factor := aug.GetFloat64(k, i)
			for j := 0; j < 2*n; j++ {
				val := aug.GetFloat64(k, j) - factor*aug.GetFloat64(i, j)
				aug.SetFloat64(val, k, j)
			}
		}
	}
	
	// Extract inverse from right side
	result := tensor.Zeros([]int{n, n}, tensor.Float64)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			result.SetFloat64(aug.GetFloat64(i, j+n), i, j)
		}
	}
	
	return result
}

// Transpose is a convenience function for matrix transpose
func Transpose(a *tensor.NDArray) *tensor.NDArray {
	return a.Transpose()
}
