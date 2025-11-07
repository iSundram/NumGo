package tensor

import (
	"fmt"
	"math"
)

// Equal checks if two arrays are element-wise equal
func (a *NDArray) Equal(b *NDArray) bool {
	if a.ndim != b.ndim || a.size != b.size {
		return false
	}
	
	for i := 0; i < a.ndim; i++ {
		if a.shape[i] != b.shape[i] {
			return false
		}
	}
	
	for i := 0; i < a.size; i++ {
		indices := a.unravelIndex(i)
		if a.GetFloat64(indices...) != b.GetFloat64(indices...) {
			return false
		}
	}
	
	return true
}

// AllClose checks if two arrays are element-wise equal within a tolerance
func (a *NDArray) AllClose(b *NDArray, rtol, atol float64) bool {
	if a.ndim != b.ndim || a.size != b.size {
		return false
	}
	
	for i := 0; i < a.ndim; i++ {
		if a.shape[i] != b.shape[i] {
			return false
		}
	}
	
	for i := 0; i < a.size; i++ {
		indices := a.unravelIndex(i)
		aVal := a.GetFloat64(indices...)
		bVal := b.GetFloat64(indices...)
		
		if !closeEnough(aVal, bVal, rtol, atol) {
			return false
		}
	}
	
	return true
}

// closeEnough checks if two values are close within tolerance
func closeEnough(a, b, rtol, atol float64) bool {
	diff := math.Abs(a - b)
	return diff <= atol+rtol*math.Abs(b)
}

// Gt (greater than) returns element-wise comparison a > b
func (a *NDArray) Gt(b *NDArray) *NDArray {
	targetShape, err := broadcastShapes(a.shape, b.shape)
	if err != nil {
		panic(err)
	}
	
	aBroad, err := a.broadcastTo(targetShape)
	if err != nil {
		panic(err)
	}
	bBroad, err := b.broadcastTo(targetShape)
	if err != nil {
		panic(err)
	}
	
	result := Zeros(targetShape, Bool)
	for i := 0; i < result.size; i++ {
		indices := result.unravelIndex(i)
		valA := aBroad.GetFloat64(indices...)
		valB := bBroad.GetFloat64(indices...)
		if valA > valB {
			result.SetFloat64(1, indices...)
		}
	}
	
	return result
}

// Lt (less than) returns element-wise comparison a < b
func (a *NDArray) Lt(b *NDArray) *NDArray {
	targetShape, err := broadcastShapes(a.shape, b.shape)
	if err != nil {
		panic(err)
	}
	
	aBroad, err := a.broadcastTo(targetShape)
	if err != nil {
		panic(err)
	}
	bBroad, err := b.broadcastTo(targetShape)
	if err != nil {
		panic(err)
	}
	
	result := Zeros(targetShape, Bool)
	for i := 0; i < result.size; i++ {
		indices := result.unravelIndex(i)
		valA := aBroad.GetFloat64(indices...)
		valB := bBroad.GetFloat64(indices...)
		if valA < valB {
			result.SetFloat64(1, indices...)
		}
	}
	
	return result
}

// GtScalar returns element-wise comparison a > scalar
func (a *NDArray) GtScalar(scalar float64) *NDArray {
	result := Zeros(a.shape, Bool)
	for i := 0; i < a.size; i++ {
		indices := a.unravelIndex(i)
		if a.GetFloat64(indices...) > scalar {
			result.SetFloat64(1, indices...)
		}
	}
	return result
}

// LtScalar returns element-wise comparison a < scalar
func (a *NDArray) LtScalar(scalar float64) *NDArray {
	result := Zeros(a.shape, Bool)
	for i := 0; i < a.size; i++ {
		indices := a.unravelIndex(i)
		if a.GetFloat64(indices...) < scalar {
			result.SetFloat64(1, indices...)
		}
	}
	return result
}

// Clip limits the values in an array between min and max
func (a *NDArray) Clip(min, max float64) *NDArray {
	result := a.Copy()
	for i := 0; i < result.size; i++ {
		indices := result.unravelIndex(i)
		val := result.GetFloat64(indices...)
		if val < min {
			result.SetFloat64(min, indices...)
		} else if val > max {
			result.SetFloat64(max, indices...)
		}
	}
	return result
}

// Where returns elements chosen from a or b depending on condition
func Where(condition, a, b *NDArray) *NDArray {
	if condition.dtype != Bool {
		panic("condition array must be boolean")
	}
	
	result := Zeros(condition.shape, a.dtype)
	for i := 0; i < result.size; i++ {
		indices := result.unravelIndex(i)
		if condition.GetFloat64(indices...) != 0 {
			result.SetFloat64(a.GetFloat64(indices...), indices...)
		} else {
			result.SetFloat64(b.GetFloat64(indices...), indices...)
		}
	}
	
	return result
}

// Concatenate joins arrays along an existing axis
func Concatenate(arrays []*NDArray, axis int) *NDArray {
	if len(arrays) == 0 {
		panic("need at least one array to concatenate")
	}
	
	if axis < 0 {
		axis = arrays[0].ndim + axis
	}
	
	// Validate arrays have compatible shapes
	ndim := arrays[0].ndim
	dtype := arrays[0].dtype
	
	for _, arr := range arrays[1:] {
		if arr.ndim != ndim {
			panic("all arrays must have same number of dimensions")
		}
		for i := 0; i < ndim; i++ {
			if i != axis && arr.shape[i] != arrays[0].shape[i] {
				panic(fmt.Sprintf("array dimensions must match except on axis %d", axis))
			}
		}
	}
	
	// Compute result shape
	resultShape := make([]int, ndim)
	copy(resultShape, arrays[0].shape)
	
	for _, arr := range arrays[1:] {
		resultShape[axis] += arr.shape[axis]
	}
	
	result := Zeros(resultShape, dtype)
	
	// Copy data
	offset := 0
	for _, arr := range arrays {
		for i := 0; i < arr.size; i++ {
			srcIndices := arr.unravelIndex(i)
			dstIndices := make([]int, ndim)
			copy(dstIndices, srcIndices)
			dstIndices[axis] += offset
			
			result.SetFloat64(arr.GetFloat64(srcIndices...), dstIndices...)
		}
		offset += arr.shape[axis]
	}
	
	return result
}

// Stack joins arrays along a new axis
func Stack(arrays []*NDArray, axis int) *NDArray {
	if len(arrays) == 0 {
		panic("need at least one array to stack")
	}
	
	// Validate all arrays have same shape
	shape := arrays[0].shape
	for _, arr := range arrays[1:] {
		if len(arr.shape) != len(shape) {
			panic("all arrays must have same shape")
		}
		for i, dim := range arr.shape {
			if dim != shape[i] {
				panic("all arrays must have same shape")
			}
		}
	}
	
	// Expand dimensions for each array
	expanded := make([]*NDArray, len(arrays))
	for i, arr := range arrays {
		expanded[i] = arr.ExpandDims(axis)
	}
	
	return Concatenate(expanded, axis)
}

// Unique returns sorted unique elements of an array
func (a *NDArray) Unique() *NDArray {
	// Flatten and get all values
	values := a.ToSliceFloat64()
	
	// Use map to track unique values
	uniqueMap := make(map[float64]bool)
	var uniqueSlice []float64
	
	for _, val := range values {
		if !uniqueMap[val] {
			uniqueMap[val] = true
			uniqueSlice = append(uniqueSlice, val)
		}
	}
	
	// Simple bubble sort (for production, use a better algorithm)
	for i := 0; i < len(uniqueSlice); i++ {
		for j := i + 1; j < len(uniqueSlice); j++ {
			if uniqueSlice[i] > uniqueSlice[j] {
				uniqueSlice[i], uniqueSlice[j] = uniqueSlice[j], uniqueSlice[i]
			}
		}
	}
	
	return FromSliceFloat64(uniqueSlice, len(uniqueSlice))
}

// Repeat repeats elements of an array
func (a *NDArray) Repeat(repeats int) *NDArray {
	if repeats < 0 {
		panic("repeats cannot be negative")
	}
	
	data := make([]float64, a.size*repeats)
	
	for i := 0; i < a.size; i++ {
		indices := a.unravelIndex(i)
		val := a.GetFloat64(indices...)
		for j := 0; j < repeats; j++ {
			data[i*repeats+j] = val
		}
	}
	
	return FromSliceFloat64(data, len(data))
}

// Tile constructs an array by repeating a the number of times given by reps
func (a *NDArray) Tile(reps ...int) *NDArray {
	if len(reps) == 0 {
		panic("reps cannot be empty")
	}
	
	// Extend reps to match ndim if needed
	if len(reps) < a.ndim {
		newReps := make([]int, a.ndim)
		offset := a.ndim - len(reps)
		for i := 0; i < offset; i++ {
			newReps[i] = 1
		}
		copy(newReps[offset:], reps)
		reps = newReps
	} else if len(reps) > a.ndim {
		// Need to reshape a to match reps dimensions
		newShape := make([]int, len(reps))
		offset := len(reps) - a.ndim
		for i := 0; i < offset; i++ {
			newShape[i] = 1
		}
		copy(newShape[offset:], a.shape)
		a = a.Reshape(newShape...)
	}
	
	// Compute result shape
	resultShape := make([]int, len(reps))
	for i := 0; i < len(reps); i++ {
		resultShape[i] = a.shape[i] * reps[i]
	}
	
	result := Zeros(resultShape, a.dtype)
	
	// Fill the result by tiling
	for i := 0; i < result.size; i++ {
		dstIndices := result.unravelIndex(i)
		srcIndices := make([]int, a.ndim)
		for j := 0; j < len(dstIndices); j++ {
			srcIndices[j] = dstIndices[j] % a.shape[j]
		}
		result.SetFloat64(a.GetFloat64(srcIndices...), dstIndices...)
	}
	
	return result
}
