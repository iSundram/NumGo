package tensor

import (
	"fmt"
	"math"
)

// broadcastShapes computes the broadcast shape of two arrays
func broadcastShapes(shape1, shape2 []int) ([]int, error) {
	// Pad the shorter shape with 1s on the left
	maxLen := len(shape1)
	if len(shape2) > maxLen {
		maxLen = len(shape2)
	}
	
	s1 := make([]int, maxLen)
	s2 := make([]int, maxLen)
	
	// Copy shapes, right-aligned
	offset1 := maxLen - len(shape1)
	offset2 := maxLen - len(shape2)
	
	for i := 0; i < offset1; i++ {
		s1[i] = 1
	}
	for i := 0; i < offset2; i++ {
		s2[i] = 1
	}
	
	copy(s1[offset1:], shape1)
	copy(s2[offset2:], shape2)
	
	// Check compatibility and compute broadcast shape
	result := make([]int, maxLen)
	for i := 0; i < maxLen; i++ {
		if s1[i] == s2[i] {
			result[i] = s1[i]
		} else if s1[i] == 1 {
			result[i] = s2[i]
		} else if s2[i] == 1 {
			result[i] = s1[i]
		} else {
			return nil, fmt.Errorf("shapes %v and %v cannot be broadcast together", shape1, shape2)
		}
	}
	
	return result, nil
}

// broadcastTo broadcasts an array to a target shape
func (a *NDArray) broadcastTo(targetShape []int) (*NDArray, error) {
	if len(targetShape) < len(a.shape) {
		return nil, fmt.Errorf("cannot broadcast shape %v to %v", a.shape, targetShape)
	}
	
	// Check if broadcasting is valid
	offset := len(targetShape) - len(a.shape)
	for i := 0; i < len(a.shape); i++ {
		if a.shape[i] != 1 && a.shape[i] != targetShape[offset+i] {
			return nil, fmt.Errorf("cannot broadcast shape %v to %v", a.shape, targetShape)
		}
	}
	
	// Create result array
	result := Zeros(targetShape, a.dtype)
	
	// Copy data with broadcasting
	for i := 0; i < result.size; i++ {
		dstIndices := result.unravelIndex(i)
		srcIndices := make([]int, a.ndim)
		
		for j := 0; j < a.ndim; j++ {
			srcIdx := dstIndices[offset+j]
			if a.shape[j] == 1 {
				srcIdx = 0
			}
			srcIndices[j] = srcIdx
		}
		
		val := a.GetFloat64(srcIndices...)
		result.SetFloat64(val, dstIndices...)
	}
	
	return result, nil
}

// Add performs element-wise addition with broadcasting
func (a *NDArray) Add(b *NDArray) *NDArray {
	// Compute broadcast shape
	targetShape, err := broadcastShapes(a.shape, b.shape)
	if err != nil {
		panic(err)
	}
	
	// Broadcast both arrays to target shape
	aBroad, err := a.broadcastTo(targetShape)
	if err != nil {
		panic(err)
	}
	bBroad, err := b.broadcastTo(targetShape)
	if err != nil {
		panic(err)
	}
	
	// Perform element-wise addition
	result := Zeros(targetShape, a.dtype)
	for i := 0; i < result.size; i++ {
		indices := result.unravelIndex(i)
		valA := aBroad.GetFloat64(indices...)
		valB := bBroad.GetFloat64(indices...)
		result.SetFloat64(valA+valB, indices...)
	}
	
	return result
}

// Sub performs element-wise subtraction with broadcasting
func (a *NDArray) Sub(b *NDArray) *NDArray {
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
	
	result := Zeros(targetShape, a.dtype)
	for i := 0; i < result.size; i++ {
		indices := result.unravelIndex(i)
		valA := aBroad.GetFloat64(indices...)
		valB := bBroad.GetFloat64(indices...)
		result.SetFloat64(valA-valB, indices...)
	}
	
	return result
}

// Mul performs element-wise multiplication with broadcasting
func (a *NDArray) Mul(b *NDArray) *NDArray {
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
	
	result := Zeros(targetShape, a.dtype)
	for i := 0; i < result.size; i++ {
		indices := result.unravelIndex(i)
		valA := aBroad.GetFloat64(indices...)
		valB := bBroad.GetFloat64(indices...)
		result.SetFloat64(valA*valB, indices...)
	}
	
	return result
}

// Div performs element-wise division with broadcasting
func (a *NDArray) Div(b *NDArray) *NDArray {
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
	
	result := Zeros(targetShape, a.dtype)
	for i := 0; i < result.size; i++ {
		indices := result.unravelIndex(i)
		valA := aBroad.GetFloat64(indices...)
		valB := bBroad.GetFloat64(indices...)
		result.SetFloat64(valA/valB, indices...)
	}
	
	return result
}

// AddScalar adds a scalar value to all elements
func (a *NDArray) AddScalar(scalar float64) *NDArray {
	result := a.Copy()
	for i := 0; i < result.size; i++ {
		indices := result.unravelIndex(i)
		val := result.GetFloat64(indices...)
		result.SetFloat64(val+scalar, indices...)
	}
	return result
}

// MulScalar multiplies all elements by a scalar value
func (a *NDArray) MulScalar(scalar float64) *NDArray {
	result := a.Copy()
	for i := 0; i < result.size; i++ {
		indices := result.unravelIndex(i)
		val := result.GetFloat64(indices...)
		result.SetFloat64(val*scalar, indices...)
	}
	return result
}

// Pow raises each element to the power of exponent
func (a *NDArray) Pow(exponent float64) *NDArray {
	result := Zeros(a.shape, a.dtype)
	for i := 0; i < result.size; i++ {
		indices := result.unravelIndex(i)
		val := a.GetFloat64(indices...)
		result.SetFloat64(math.Pow(val, exponent), indices...)
	}
	return result
}

// Sqrt computes the square root of each element
func (a *NDArray) Sqrt() *NDArray {
	result := Zeros(a.shape, a.dtype)
	for i := 0; i < result.size; i++ {
		indices := result.unravelIndex(i)
		val := a.GetFloat64(indices...)
		result.SetFloat64(math.Sqrt(val), indices...)
	}
	return result
}

// Exp computes e^x for each element
func (a *NDArray) Exp() *NDArray {
	result := Zeros(a.shape, a.dtype)
	for i := 0; i < result.size; i++ {
		indices := result.unravelIndex(i)
		val := a.GetFloat64(indices...)
		result.SetFloat64(math.Exp(val), indices...)
	}
	return result
}

// Log computes natural logarithm for each element
func (a *NDArray) Log() *NDArray {
	result := Zeros(a.shape, a.dtype)
	for i := 0; i < result.size; i++ {
		indices := result.unravelIndex(i)
		val := a.GetFloat64(indices...)
		result.SetFloat64(math.Log(val), indices...)
	}
	return result
}

// Sin computes sine for each element
func (a *NDArray) Sin() *NDArray {
	result := Zeros(a.shape, a.dtype)
	for i := 0; i < result.size; i++ {
		indices := result.unravelIndex(i)
		val := a.GetFloat64(indices...)
		result.SetFloat64(math.Sin(val), indices...)
	}
	return result
}

// Cos computes cosine for each element
func (a *NDArray) Cos() *NDArray {
	result := Zeros(a.shape, a.dtype)
	for i := 0; i < result.size; i++ {
		indices := result.unravelIndex(i)
		val := a.GetFloat64(indices...)
		result.SetFloat64(math.Cos(val), indices...)
	}
	return result
}

// Neg computes the negation of each element
func (a *NDArray) Neg() *NDArray {
	result := Zeros(a.shape, a.dtype)
	for i := 0; i < result.size; i++ {
		indices := result.unravelIndex(i)
		val := a.GetFloat64(indices...)
		result.SetFloat64(-val, indices...)
	}
	return result
}

// Abs computes the absolute value of each element
func (a *NDArray) Abs() *NDArray {
	result := Zeros(a.shape, a.dtype)
	for i := 0; i < result.size; i++ {
		indices := result.unravelIndex(i)
		val := a.GetFloat64(indices...)
		result.SetFloat64(math.Abs(val), indices...)
	}
	return result
}
