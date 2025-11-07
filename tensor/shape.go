package tensor

import (
	"fmt"
)

// Reshape returns a new array with the same data but different shape
func (a *NDArray) Reshape(newShape ...int) *NDArray {
	// Calculate the new size
	newSize := computeSize(newShape)
	
	// Handle -1 in shape (infer dimension)
	inferIdx := -1
	knownSize := 1
	for i, dim := range newShape {
		if dim == -1 {
			if inferIdx != -1 {
				panic("can only specify one unknown dimension")
			}
			inferIdx = i
		} else if dim < 0 {
			panic(fmt.Sprintf("negative dimensions not allowed except -1: %d", dim))
		} else {
			knownSize *= dim
		}
	}
	
	if inferIdx != -1 {
		if a.size%knownSize != 0 {
			panic(fmt.Sprintf("cannot reshape array of size %d into shape with known size %d", a.size, knownSize))
		}
		newShape[inferIdx] = a.size / knownSize
		newSize = a.size
	}
	
	if newSize != a.size {
		panic(fmt.Sprintf("cannot reshape array of size %d into shape %v (size %d)", a.size, newShape, newSize))
	}
	
	// Create new array with new shape but same data
	newStrides := computeStrides(newShape, a.dtype.ItemSize())
	
	// For reshape, we need to copy data to ensure C-contiguous layout
	newData := make([]byte, len(a.data))
	copy(newData, a.data)
	
	return &NDArray{
		data:    newData,
		shape:   append([]int{}, newShape...),
		strides: newStrides,
		dtype:   a.dtype,
		size:    newSize,
		ndim:    len(newShape),
	}
}

// Transpose returns a transposed view of the array
// For 2D arrays, swaps rows and columns
// For ND arrays, reverses the order of axes
func (a *NDArray) Transpose(axes ...int) *NDArray {
	// If no axes specified, reverse all axes
	if len(axes) == 0 {
		axes = make([]int, a.ndim)
		for i := 0; i < a.ndim; i++ {
			axes[i] = a.ndim - 1 - i
		}
	}
	
	if len(axes) != a.ndim {
		panic(fmt.Sprintf("axes must have length %d, got %d", a.ndim, len(axes)))
	}
	
	// Validate axes
	seen := make(map[int]bool)
	for _, axis := range axes {
		if axis < 0 || axis >= a.ndim {
			panic(fmt.Sprintf("axis %d is out of bounds for array of dimension %d", axis, a.ndim))
		}
		if seen[axis] {
			panic(fmt.Sprintf("repeated axis: %d", axis))
		}
		seen[axis] = true
	}
	
	// Create new shape and strides based on permutation
	newShape := make([]int, a.ndim)
	newStrides := make([]int, a.ndim)
	for i, axis := range axes {
		newShape[i] = a.shape[axis]
		newStrides[i] = a.strides[axis]
	}
	
	// Transpose creates a view with reordered axes
	// We need to copy and reorder the data
	newArr := &NDArray{
		data:    make([]byte, len(a.data)),
		shape:   newShape,
		strides: computeStrides(newShape, a.dtype.ItemSize()),
		dtype:   a.dtype,
		size:    a.size,
		ndim:    a.ndim,
	}
	
	// Copy data in transposed order
	for i := 0; i < a.size; i++ {
		srcIndices := a.unravelIndex(i)
		dstIndices := make([]int, a.ndim)
		for j, axis := range axes {
			dstIndices[j] = srcIndices[axis]
		}
		
		srcOffset := a.flatIndex(srcIndices...)
		dstOffset := newArr.flatIndex(dstIndices...)
		
		itemsize := a.dtype.ItemSize()
		copy(newArr.data[dstOffset:dstOffset+itemsize], a.data[srcOffset:srcOffset+itemsize])
	}
	
	return newArr
}

// T returns the transpose of a 2D array
func (a *NDArray) T() *NDArray {
	if a.ndim != 2 {
		panic(fmt.Sprintf("T property only valid for 2D arrays, got %dD", a.ndim))
	}
	return a.Transpose()
}

// Flatten returns a flattened 1D copy of the array
func (a *NDArray) Flatten() *NDArray {
	return a.Reshape(a.size)
}

// Ravel returns a flattened 1D view of the array (same as Flatten for now)
func (a *NDArray) Ravel() *NDArray {
	return a.Flatten()
}

// Squeeze removes single-dimensional entries from the shape
func (a *NDArray) Squeeze() *NDArray {
	newShape := []int{}
	for _, dim := range a.shape {
		if dim != 1 {
			newShape = append(newShape, dim)
		}
	}
	
	// If all dimensions were 1, return scalar (shape [])
	if len(newShape) == 0 {
		newShape = []int{1}
	}
	
	return a.Reshape(newShape...)
}

// ExpandDims adds a new axis at the specified position
func (a *NDArray) ExpandDims(axis int) *NDArray {
	if axis < 0 {
		axis = a.ndim + axis + 1
	}
	if axis < 0 || axis > a.ndim {
		panic(fmt.Sprintf("axis %d is out of bounds for array of dimension %d", axis, a.ndim))
	}
	
	newShape := make([]int, a.ndim+1)
	copy(newShape[:axis], a.shape[:axis])
	newShape[axis] = 1
	copy(newShape[axis+1:], a.shape[axis:])
	
	return a.Reshape(newShape...)
}

// SwapAxes swaps two axes of an array
func (a *NDArray) SwapAxes(axis1, axis2 int) *NDArray {
	if axis1 < 0 {
		axis1 = a.ndim + axis1
	}
	if axis2 < 0 {
		axis2 = a.ndim + axis2
	}
	
	axes := make([]int, a.ndim)
	for i := 0; i < a.ndim; i++ {
		axes[i] = i
	}
	axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
	
	return a.Transpose(axes...)
}
