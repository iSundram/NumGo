package tensor

import (
	"encoding/binary"
	"fmt"
	"math"
)

// flatIndex converts multi-dimensional indices to a flat byte offset
func (a *NDArray) flatIndex(indices ...int) int {
	if len(indices) != a.ndim {
		panic(fmt.Sprintf("expected %d indices, got %d", a.ndim, len(indices)))
	}
	
	offset := 0
	for i, idx := range indices {
		// Handle negative indexing
		if idx < 0 {
			idx = a.shape[i] + idx
		}
		if idx < 0 || idx >= a.shape[i] {
			panic(fmt.Sprintf("index %d is out of bounds for axis %d with size %d", idx, i, a.shape[i]))
		}
		offset += idx * a.strides[i]
	}
	return offset
}

// GetFloat64 returns the element at the given indices as float64
func (a *NDArray) GetFloat64(indices ...int) float64 {
	offset := a.flatIndex(indices...)
	
	switch a.dtype {
	case Float64:
		bits := binary.LittleEndian.Uint64(a.data[offset : offset+8])
		return math.Float64frombits(bits)
	case Float32:
		bits := binary.LittleEndian.Uint32(a.data[offset : offset+4])
		return float64(math.Float32frombits(bits))
	case Int64:
		return float64(int64(binary.LittleEndian.Uint64(a.data[offset : offset+8])))
	case Int32:
		return float64(int32(binary.LittleEndian.Uint32(a.data[offset : offset+4])))
	case Int16:
		return float64(int16(binary.LittleEndian.Uint16(a.data[offset : offset+2])))
	case Int8:
		return float64(int8(a.data[offset]))
	case Uint64:
		return float64(binary.LittleEndian.Uint64(a.data[offset : offset+8]))
	case Uint32:
		return float64(binary.LittleEndian.Uint32(a.data[offset : offset+4]))
	case Uint16:
		return float64(binary.LittleEndian.Uint16(a.data[offset : offset+2]))
	case Uint8:
		return float64(a.data[offset])
	case Bool:
		if a.data[offset] != 0 {
			return 1.0
		}
		return 0.0
	default:
		panic(fmt.Sprintf("unsupported dtype for GetFloat64: %s", a.dtype))
	}
}

// GetInt64 returns the element at the given indices as int64
func (a *NDArray) GetInt64(indices ...int) int64 {
	offset := a.flatIndex(indices...)
	
	switch a.dtype {
	case Int64:
		return int64(binary.LittleEndian.Uint64(a.data[offset : offset+8]))
	case Int32:
		return int64(int32(binary.LittleEndian.Uint32(a.data[offset : offset+4])))
	case Int16:
		return int64(int16(binary.LittleEndian.Uint16(a.data[offset : offset+2])))
	case Int8:
		return int64(int8(a.data[offset]))
	case Uint64:
		return int64(binary.LittleEndian.Uint64(a.data[offset : offset+8]))
	case Uint32:
		return int64(binary.LittleEndian.Uint32(a.data[offset : offset+4]))
	case Uint16:
		return int64(binary.LittleEndian.Uint16(a.data[offset : offset+2]))
	case Uint8:
		return int64(a.data[offset])
	case Bool:
		if a.data[offset] != 0 {
			return 1
		}
		return 0
	case Float64:
		bits := binary.LittleEndian.Uint64(a.data[offset : offset+8])
		return int64(math.Float64frombits(bits))
	case Float32:
		bits := binary.LittleEndian.Uint32(a.data[offset : offset+4])
		return int64(math.Float32frombits(bits))
	default:
		panic(fmt.Sprintf("unsupported dtype for GetInt64: %s", a.dtype))
	}
}

// SetFloat64 sets the element at the given indices from a float64 value
func (a *NDArray) SetFloat64(value float64, indices ...int) {
	offset := a.flatIndex(indices...)
	
	switch a.dtype {
	case Float64:
		binary.LittleEndian.PutUint64(a.data[offset:offset+8], math.Float64bits(value))
	case Float32:
		binary.LittleEndian.PutUint32(a.data[offset:offset+4], math.Float32bits(float32(value)))
	case Int64:
		binary.LittleEndian.PutUint64(a.data[offset:offset+8], uint64(int64(value)))
	case Int32:
		binary.LittleEndian.PutUint32(a.data[offset:offset+4], uint32(int32(value)))
	case Int16:
		binary.LittleEndian.PutUint16(a.data[offset:offset+2], uint16(int16(value)))
	case Int8:
		a.data[offset] = byte(int8(value))
	case Uint64:
		binary.LittleEndian.PutUint64(a.data[offset:offset+8], uint64(value))
	case Uint32:
		binary.LittleEndian.PutUint32(a.data[offset:offset+4], uint32(value))
	case Uint16:
		binary.LittleEndian.PutUint16(a.data[offset:offset+2], uint16(value))
	case Uint8:
		a.data[offset] = byte(uint8(value))
	case Bool:
		if value != 0 {
			a.data[offset] = 1
		} else {
			a.data[offset] = 0
		}
	default:
		panic(fmt.Sprintf("unsupported dtype for SetFloat64: %s", a.dtype))
	}
}

// SetInt64 sets the element at the given indices from an int64 value
func (a *NDArray) SetInt64(value int64, indices ...int) {
	offset := a.flatIndex(indices...)
	
	switch a.dtype {
	case Int64:
		binary.LittleEndian.PutUint64(a.data[offset:offset+8], uint64(value))
	case Int32:
		binary.LittleEndian.PutUint32(a.data[offset:offset+4], uint32(int32(value)))
	case Int16:
		binary.LittleEndian.PutUint16(a.data[offset:offset+2], uint16(int16(value)))
	case Int8:
		a.data[offset] = byte(int8(value))
	case Uint64:
		binary.LittleEndian.PutUint64(a.data[offset:offset+8], uint64(value))
	case Uint32:
		binary.LittleEndian.PutUint32(a.data[offset:offset+4], uint32(value))
	case Uint16:
		binary.LittleEndian.PutUint16(a.data[offset:offset+2], uint16(value))
	case Uint8:
		a.data[offset] = byte(uint8(value))
	case Bool:
		if value != 0 {
			a.data[offset] = 1
		} else {
			a.data[offset] = 0
		}
	case Float64:
		binary.LittleEndian.PutUint64(a.data[offset:offset+8], math.Float64bits(float64(value)))
	case Float32:
		binary.LittleEndian.PutUint32(a.data[offset:offset+4], math.Float32bits(float32(value)))
	default:
		panic(fmt.Sprintf("unsupported dtype for SetInt64: %s", a.dtype))
	}
}

// ToSliceFloat64 converts the entire array to a flat float64 slice
func (a *NDArray) ToSliceFloat64() []float64 {
	result := make([]float64, a.size)
	
	// For C-contiguous arrays, we can iterate linearly
	for i := 0; i < a.size; i++ {
		indices := a.unravelIndex(i)
		result[i] = a.GetFloat64(indices...)
	}
	
	return result
}

// ToSliceInt64 converts the entire array to a flat int64 slice
func (a *NDArray) ToSliceInt64() []int64 {
	result := make([]int64, a.size)
	
	for i := 0; i < a.size; i++ {
		indices := a.unravelIndex(i)
		result[i] = a.GetInt64(indices...)
	}
	
	return result
}

// unravelIndex converts a flat index to multi-dimensional indices
func (a *NDArray) unravelIndex(flatIdx int) []int {
	indices := make([]int, a.ndim)
	remaining := flatIdx
	
	for i := a.ndim - 1; i >= 0; i-- {
		indices[i] = remaining % a.shape[i]
		remaining /= a.shape[i]
	}
	
	return indices
}

// Copy creates a deep copy of the array
func (a *NDArray) Copy() *NDArray {
	newData := make([]byte, len(a.data))
	copy(newData, a.data)
	
	return &NDArray{
		data:    newData,
		shape:   append([]int{}, a.shape...),
		strides: append([]int{}, a.strides...),
		dtype:   a.dtype,
		size:    a.size,
		ndim:    a.ndim,
	}
}
