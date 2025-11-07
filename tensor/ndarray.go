// Package tensor provides the core N-dimensional array (NDArray) implementation
// for NumGo, similar to NumPy's ndarray.
package tensor

import (
	"fmt"
)

// DType represents the data type of array elements
type DType int

const (
	Bool DType = iota
	Int8
	Int16
	Int32
	Int64
	Uint8
	Uint16
	Uint32
	Uint64
	Float32
	Float64
	Complex64
	Complex128
)

// String returns the string representation of a DType
func (dt DType) String() string {
	switch dt {
	case Bool:
		return "bool"
	case Int8:
		return "int8"
	case Int16:
		return "int16"
	case Int32:
		return "int32"
	case Int64:
		return "int64"
	case Uint8:
		return "uint8"
	case Uint16:
		return "uint16"
	case Uint32:
		return "uint32"
	case Uint64:
		return "uint64"
	case Float32:
		return "float32"
	case Float64:
		return "float64"
	case Complex64:
		return "complex64"
	case Complex128:
		return "complex128"
	default:
		return "unknown"
	}
}

// ItemSize returns the size in bytes of one element of this dtype
func (dt DType) ItemSize() int {
	switch dt {
	case Bool, Int8, Uint8:
		return 1
	case Int16, Uint16:
		return 2
	case Int32, Uint32, Float32:
		return 4
	case Int64, Uint64, Float64, Complex64:
		return 8
	case Complex128:
		return 16
	default:
		return 0
	}
}

// IsFloat returns true if the dtype is a floating point type
func (dt DType) IsFloat() bool {
	return dt == Float32 || dt == Float64
}

// IsInt returns true if the dtype is an integer type
func (dt DType) IsInt() bool {
	return dt >= Int8 && dt <= Uint64
}

// IsComplex returns true if the dtype is a complex type
func (dt DType) IsComplex() bool {
	return dt == Complex64 || dt == Complex128
}

// NDArray represents an N-dimensional array with homogeneous data type
type NDArray struct {
	data   []byte   // Underlying data buffer
	shape  []int    // Dimensions of the array
	strides []int   // Strides for each dimension (in bytes)
	dtype  DType    // Data type of elements
	size   int      // Total number of elements
	ndim   int      // Number of dimensions
}

// Shape returns the shape of the array
func (a *NDArray) Shape() []int {
	return append([]int{}, a.shape...)
}

// Strides returns the strides of the array
func (a *NDArray) Strides() []int {
	return append([]int{}, a.strides...)
}

// DType returns the data type of the array
func (a *NDArray) DType() DType {
	return a.dtype
}

// Size returns the total number of elements
func (a *NDArray) Size() int {
	return a.size
}

// Ndim returns the number of dimensions
func (a *NDArray) Ndim() int {
	return a.ndim
}

// ItemSize returns the size of one element in bytes
func (a *NDArray) ItemSize() int {
	return a.dtype.ItemSize()
}

// computeStrides calculates C-contiguous strides for a given shape
func computeStrides(shape []int, itemsize int) []int {
	if len(shape) == 0 {
		return []int{}
	}
	
	strides := make([]int, len(shape))
	stride := itemsize
	for i := len(shape) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= shape[i]
	}
	return strides
}

// computeSize calculates the total number of elements from shape
func computeSize(shape []int) int {
	if len(shape) == 0 {
		return 0
	}
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	return size
}

// String returns a string representation of the array
func (a *NDArray) String() string {
	return fmt.Sprintf("NDArray(shape=%v, dtype=%s)", a.shape, a.dtype)
}

// Data returns a copy of the underlying data buffer
func (a *NDArray) Data() []byte {
	return append([]byte{}, a.data...)
}
