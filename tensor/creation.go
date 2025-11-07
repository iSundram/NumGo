package tensor

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Zeros creates a new array filled with zeros
func Zeros(shape []int, dtype DType) *NDArray {
	size := computeSize(shape)
	itemsize := dtype.ItemSize()
	data := make([]byte, size*itemsize)
	strides := computeStrides(shape, itemsize)
	
	return &NDArray{
		data:    data,
		shape:   append([]int{}, shape...),
		strides: strides,
		dtype:   dtype,
		size:    size,
		ndim:    len(shape),
	}
}

// Ones creates a new array filled with ones
func Ones(shape []int, dtype DType) *NDArray {
	arr := Zeros(shape, dtype)
	
	// Fill with ones based on dtype
	switch dtype {
	case Float64:
		for i := 0; i < arr.size; i++ {
			offset := i * 8
			binary.LittleEndian.PutUint64(arr.data[offset:offset+8], math.Float64bits(1.0))
		}
	case Float32:
		for i := 0; i < arr.size; i++ {
			offset := i * 4
			binary.LittleEndian.PutUint32(arr.data[offset:offset+4], math.Float32bits(1.0))
		}
	case Int64:
		for i := 0; i < arr.size; i++ {
			offset := i * 8
			binary.LittleEndian.PutUint64(arr.data[offset:offset+8], uint64(1))
		}
	case Int32:
		for i := 0; i < arr.size; i++ {
			offset := i * 4
			binary.LittleEndian.PutUint32(arr.data[offset:offset+4], uint32(1))
		}
	case Int16:
		for i := 0; i < arr.size; i++ {
			offset := i * 2
			binary.LittleEndian.PutUint16(arr.data[offset:offset+2], uint16(1))
		}
	case Int8:
		for i := 0; i < arr.size; i++ {
			arr.data[i] = 1
		}
	case Uint64:
		for i := 0; i < arr.size; i++ {
			offset := i * 8
			binary.LittleEndian.PutUint64(arr.data[offset:offset+8], 1)
		}
	case Uint32:
		for i := 0; i < arr.size; i++ {
			offset := i * 4
			binary.LittleEndian.PutUint32(arr.data[offset:offset+4], 1)
		}
	case Uint16:
		for i := 0; i < arr.size; i++ {
			offset := i * 2
			binary.LittleEndian.PutUint16(arr.data[offset:offset+2], 1)
		}
	case Uint8:
		for i := 0; i < arr.size; i++ {
			arr.data[i] = 1
		}
	case Bool:
		for i := 0; i < arr.size; i++ {
			arr.data[i] = 1
		}
	}
	
	return arr
}

// FromSliceFloat64 creates an array from a float64 slice with given shape
func FromSliceFloat64(data []float64, shape ...int) *NDArray {
	size := computeSize(shape)
	if len(data) != size {
		panic(fmt.Sprintf("data length %d does not match shape size %d", len(data), size))
	}
	
	itemsize := Float64.ItemSize()
	byteData := make([]byte, size*itemsize)
	
	for i, val := range data {
		offset := i * 8
		binary.LittleEndian.PutUint64(byteData[offset:offset+8], math.Float64bits(val))
	}
	
	strides := computeStrides(shape, itemsize)
	
	return &NDArray{
		data:    byteData,
		shape:   append([]int{}, shape...),
		strides: strides,
		dtype:   Float64,
		size:    size,
		ndim:    len(shape),
	}
}

// FromSliceInt64 creates an array from an int64 slice with given shape
func FromSliceInt64(data []int64, shape ...int) *NDArray {
	size := computeSize(shape)
	if len(data) != size {
		panic(fmt.Sprintf("data length %d does not match shape size %d", len(data), size))
	}
	
	itemsize := Int64.ItemSize()
	byteData := make([]byte, size*itemsize)
	
	for i, val := range data {
		offset := i * 8
		binary.LittleEndian.PutUint64(byteData[offset:offset+8], uint64(val))
	}
	
	strides := computeStrides(shape, itemsize)
	
	return &NDArray{
		data:    byteData,
		shape:   append([]int{}, shape...),
		strides: strides,
		dtype:   Int64,
		size:    size,
		ndim:    len(shape),
	}
}

// FromSliceFloat32 creates an array from a float32 slice with given shape
func FromSliceFloat32(data []float32, shape ...int) *NDArray {
	size := computeSize(shape)
	if len(data) != size {
		panic(fmt.Sprintf("data length %d does not match shape size %d", len(data), size))
	}
	
	itemsize := Float32.ItemSize()
	byteData := make([]byte, size*itemsize)
	
	for i, val := range data {
		offset := i * 4
		binary.LittleEndian.PutUint32(byteData[offset:offset+4], math.Float32bits(val))
	}
	
	strides := computeStrides(shape, itemsize)
	
	return &NDArray{
		data:    byteData,
		shape:   append([]int{}, shape...),
		strides: strides,
		dtype:   Float32,
		size:    size,
		ndim:    len(shape),
	}
}

// Arange creates an array with evenly spaced values within a given interval
// Similar to NumPy's arange(start, stop, step)
func Arange(start, stop, step float64) *NDArray {
	if step == 0 {
		panic("step cannot be zero")
	}
	
	n := int(math.Ceil((stop - start) / step))
	if n < 0 {
		n = 0
	}
	
	data := make([]float64, n)
	for i := 0; i < n; i++ {
		data[i] = start + float64(i)*step
	}
	
	return FromSliceFloat64(data, n)
}

// Range creates an array with integers from start to stop (exclusive)
// Similar to NumPy's np.arange(start, stop) with step=1
func Range(start, stop int) *NDArray {
	if stop < start {
		return FromSliceInt64([]int64{}, 0)
	}
	
	n := stop - start
	data := make([]int64, n)
	for i := 0; i < n; i++ {
		data[i] = int64(start + i)
	}
	
	return FromSliceInt64(data, n)
}

// Full creates an array filled with a constant value
func Full(shape []int, value float64, dtype DType) *NDArray {
	arr := Zeros(shape, dtype)
	
	switch dtype {
	case Float64:
		for i := 0; i < arr.size; i++ {
			offset := i * 8
			binary.LittleEndian.PutUint64(arr.data[offset:offset+8], math.Float64bits(value))
		}
	case Float32:
		for i := 0; i < arr.size; i++ {
			offset := i * 4
			binary.LittleEndian.PutUint32(arr.data[offset:offset+4], math.Float32bits(float32(value)))
		}
	case Int64:
		for i := 0; i < arr.size; i++ {
			offset := i * 8
			binary.LittleEndian.PutUint64(arr.data[offset:offset+8], uint64(int64(value)))
		}
	case Int32:
		for i := 0; i < arr.size; i++ {
			offset := i * 4
			binary.LittleEndian.PutUint32(arr.data[offset:offset+4], uint32(int32(value)))
		}
	}
	
	return arr
}

// Eye creates a 2-D array with ones on the diagonal and zeros elsewhere
func Eye(n int, dtype DType) *NDArray {
	arr := Zeros([]int{n, n}, dtype)
	
	switch dtype {
	case Float64:
		for i := 0; i < n; i++ {
			offset := (i*n + i) * 8
			binary.LittleEndian.PutUint64(arr.data[offset:offset+8], math.Float64bits(1.0))
		}
	case Float32:
		for i := 0; i < n; i++ {
			offset := (i*n + i) * 4
			binary.LittleEndian.PutUint32(arr.data[offset:offset+4], math.Float32bits(1.0))
		}
	case Int64:
		for i := 0; i < n; i++ {
			offset := (i*n + i) * 8
			binary.LittleEndian.PutUint64(arr.data[offset:offset+8], 1)
		}
	case Int32:
		for i := 0; i < n; i++ {
			offset := (i*n + i) * 4
			binary.LittleEndian.PutUint32(arr.data[offset:offset+4], 1)
		}
	}
	
	return arr
}
