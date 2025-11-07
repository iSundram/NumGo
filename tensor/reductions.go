package tensor

import (
	"math"
)

// Sum computes the sum of all elements
func (a *NDArray) Sum() float64 {
	sum := 0.0
	for i := 0; i < a.size; i++ {
		indices := a.unravelIndex(i)
		sum += a.GetFloat64(indices...)
	}
	return sum
}

// Mean computes the mean of all elements
func (a *NDArray) Mean() float64 {
	if a.size == 0 {
		return math.NaN()
	}
	return a.Sum() / float64(a.size)
}

// Min returns the minimum value
func (a *NDArray) Min() float64 {
	if a.size == 0 {
		return math.NaN()
	}
	
	indices := a.unravelIndex(0)
	min := a.GetFloat64(indices...)
	
	for i := 1; i < a.size; i++ {
		indices := a.unravelIndex(i)
		val := a.GetFloat64(indices...)
		if val < min {
			min = val
		}
	}
	return min
}

// Max returns the maximum value
func (a *NDArray) Max() float64 {
	if a.size == 0 {
		return math.NaN()
	}
	
	indices := a.unravelIndex(0)
	max := a.GetFloat64(indices...)
	
	for i := 1; i < a.size; i++ {
		indices := a.unravelIndex(i)
		val := a.GetFloat64(indices...)
		if val > max {
			max = val
		}
	}
	return max
}

// ArgMin returns the index of the minimum value (flattened)
func (a *NDArray) ArgMin() int {
	if a.size == 0 {
		return -1
	}
	
	indices := a.unravelIndex(0)
	minVal := a.GetFloat64(indices...)
	minIdx := 0
	
	for i := 1; i < a.size; i++ {
		indices := a.unravelIndex(i)
		val := a.GetFloat64(indices...)
		if val < minVal {
			minVal = val
			minIdx = i
		}
	}
	return minIdx
}

// ArgMax returns the index of the maximum value (flattened)
func (a *NDArray) ArgMax() int {
	if a.size == 0 {
		return -1
	}
	
	indices := a.unravelIndex(0)
	maxVal := a.GetFloat64(indices...)
	maxIdx := 0
	
	for i := 1; i < a.size; i++ {
		indices := a.unravelIndex(i)
		val := a.GetFloat64(indices...)
		if val > maxVal {
			maxVal = val
			maxIdx = i
		}
	}
	return maxIdx
}

// Prod computes the product of all elements
func (a *NDArray) Prod() float64 {
	prod := 1.0
	for i := 0; i < a.size; i++ {
		indices := a.unravelIndex(i)
		prod *= a.GetFloat64(indices...)
	}
	return prod
}

// Var computes the variance of all elements
func (a *NDArray) Var() float64 {
	if a.size == 0 {
		return math.NaN()
	}
	
	mean := a.Mean()
	variance := 0.0
	
	for i := 0; i < a.size; i++ {
		indices := a.unravelIndex(i)
		val := a.GetFloat64(indices...)
		diff := val - mean
		variance += diff * diff
	}
	
	return variance / float64(a.size)
}

// Std computes the standard deviation of all elements
func (a *NDArray) Std() float64 {
	return math.Sqrt(a.Var())
}

// All returns true if all elements are non-zero
func (a *NDArray) All() bool {
	for i := 0; i < a.size; i++ {
		indices := a.unravelIndex(i)
		val := a.GetFloat64(indices...)
		if val == 0.0 {
			return false
		}
	}
	return true
}

// Any returns true if any element is non-zero
func (a *NDArray) Any() bool {
	for i := 0; i < a.size; i++ {
		indices := a.unravelIndex(i)
		val := a.GetFloat64(indices...)
		if val != 0.0 {
			return true
		}
	}
	return false
}

// SumAxis computes the sum along a specific axis
func (a *NDArray) SumAxis(axis int) *NDArray {
	if axis < 0 {
		axis = a.ndim + axis
	}
	if axis < 0 || axis >= a.ndim {
		panic("axis out of bounds")
	}
	
	// Compute result shape (remove the specified axis)
	resultShape := make([]int, 0, a.ndim-1)
	for i := 0; i < a.ndim; i++ {
		if i != axis {
			resultShape = append(resultShape, a.shape[i])
		}
	}
	
	// Handle case where result is a scalar
	if len(resultShape) == 0 {
		return FromSliceFloat64([]float64{a.Sum()}, 1)
	}
	
	result := Zeros(resultShape, a.dtype)
	
	// Sum along the axis
	for i := 0; i < a.size; i++ {
		srcIndices := a.unravelIndex(i)
		
		// Create result indices by removing the axis
		dstIndices := make([]int, 0, len(resultShape))
		for j := 0; j < a.ndim; j++ {
			if j != axis {
				dstIndices = append(dstIndices, srcIndices[j])
			}
		}
		
		val := a.GetFloat64(srcIndices...)
		currentSum := result.GetFloat64(dstIndices...)
		result.SetFloat64(currentSum+val, dstIndices...)
	}
	
	return result
}

// MeanAxis computes the mean along a specific axis
func (a *NDArray) MeanAxis(axis int) *NDArray {
	sumResult := a.SumAxis(axis)
	divisor := float64(a.shape[axis])
	
	// Divide by the size of the axis
	for i := 0; i < sumResult.size; i++ {
		indices := sumResult.unravelIndex(i)
		val := sumResult.GetFloat64(indices...)
		sumResult.SetFloat64(val/divisor, indices...)
	}
	
	return sumResult
}
