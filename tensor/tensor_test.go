package tensor

import (
	"testing"
)

func TestZeros(t *testing.T) {
	arr := Zeros([]int{2, 3}, Float64)
	
	if arr.Size() != 6 {
		t.Errorf("expected size 6, got %d", arr.Size())
	}
	
	if arr.Ndim() != 2 {
		t.Errorf("expected ndim 2, got %d", arr.Ndim())
	}
	
	if arr.DType() != Float64 {
		t.Errorf("expected dtype Float64, got %s", arr.DType())
	}
	
	// Check all values are zero
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			val := arr.GetFloat64(i, j)
			if val != 0.0 {
				t.Errorf("expected 0.0 at [%d,%d], got %f", i, j, val)
			}
		}
	}
}

func TestOnes(t *testing.T) {
	arr := Ones([]int{3, 2}, Float64)
	
	if arr.Size() != 6 {
		t.Errorf("expected size 6, got %d", arr.Size())
	}
	
	// Check all values are one
	for i := 0; i < 3; i++ {
		for j := 0; j < 2; j++ {
			val := arr.GetFloat64(i, j)
			if val != 1.0 {
				t.Errorf("expected 1.0 at [%d,%d], got %f", i, j, val)
			}
		}
	}
}

func TestFromSliceFloat64(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	arr := FromSliceFloat64(data, 2, 3)
	
	if arr.Size() != 6 {
		t.Errorf("expected size 6, got %d", arr.Size())
	}
	
	expected := [][]float64{
		{1, 2, 3},
		{4, 5, 6},
	}
	
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			val := arr.GetFloat64(i, j)
			if val != expected[i][j] {
				t.Errorf("expected %f at [%d,%d], got %f", expected[i][j], i, j, val)
			}
		}
	}
}

func TestArange(t *testing.T) {
	arr := Arange(0, 10, 2)
	
	if arr.Size() != 5 {
		t.Errorf("expected size 5, got %d", arr.Size())
	}
	
	expected := []float64{0, 2, 4, 6, 8}
	for i := 0; i < 5; i++ {
		val := arr.GetFloat64(i)
		if val != expected[i] {
			t.Errorf("expected %f at index %d, got %f", expected[i], i, val)
		}
	}
}

func TestRange(t *testing.T) {
	arr := Range(0, 5)
	
	if arr.Size() != 5 {
		t.Errorf("expected size 5, got %d", arr.Size())
	}
	
	expected := []int64{0, 1, 2, 3, 4}
	for i := 0; i < 5; i++ {
		val := arr.GetInt64(i)
		if val != expected[i] {
			t.Errorf("expected %d at index %d, got %d", expected[i], i, val)
		}
	}
}

func TestEye(t *testing.T) {
	arr := Eye(3, Float64)
	
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			val := arr.GetFloat64(i, j)
			if i == j {
				if val != 1.0 {
					t.Errorf("expected 1.0 at [%d,%d], got %f", i, j, val)
				}
			} else {
				if val != 0.0 {
					t.Errorf("expected 0.0 at [%d,%d], got %f", i, j, val)
				}
			}
		}
	}
}

func TestGetSet(t *testing.T) {
	arr := Zeros([]int{3, 3}, Float64)
	
	arr.SetFloat64(5.5, 1, 2)
	val := arr.GetFloat64(1, 2)
	
	if val != 5.5 {
		t.Errorf("expected 5.5, got %f", val)
	}
}

func TestNegativeIndexing(t *testing.T) {
	arr := FromSliceFloat64([]float64{1, 2, 3, 4}, 2, 2)
	
	// arr[-1, -1] should be the last element (4)
	val := arr.GetFloat64(-1, -1)
	if val != 4.0 {
		t.Errorf("expected 4.0, got %f", val)
	}
	
	// arr[-2, -2] should be the first element (1)
	val = arr.GetFloat64(-2, -2)
	if val != 1.0 {
		t.Errorf("expected 1.0, got %f", val)
	}
}

func TestReshape(t *testing.T) {
	arr := Range(0, 6)
	reshaped := arr.Reshape(2, 3)
	
	if reshaped.Ndim() != 2 {
		t.Errorf("expected ndim 2, got %d", reshaped.Ndim())
	}
	
	shape := reshaped.Shape()
	if shape[0] != 2 || shape[1] != 3 {
		t.Errorf("expected shape [2,3], got %v", shape)
	}
	
	// Test with -1 (infer dimension)
	reshaped2 := arr.Reshape(-1, 2)
	shape2 := reshaped2.Shape()
	if shape2[0] != 3 || shape2[1] != 2 {
		t.Errorf("expected shape [3,2], got %v", shape2)
	}
}

func TestTranspose(t *testing.T) {
	arr := FromSliceFloat64([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	transposed := arr.Transpose()
	
	shape := transposed.Shape()
	if shape[0] != 3 || shape[1] != 2 {
		t.Errorf("expected shape [3,2], got %v", shape)
	}
	
	// Check values
	if transposed.GetFloat64(0, 0) != 1.0 {
		t.Errorf("expected 1.0 at [0,0], got %f", transposed.GetFloat64(0, 0))
	}
	if transposed.GetFloat64(0, 1) != 4.0 {
		t.Errorf("expected 4.0 at [0,1], got %f", transposed.GetFloat64(0, 1))
	}
	if transposed.GetFloat64(1, 0) != 2.0 {
		t.Errorf("expected 2.0 at [1,0], got %f", transposed.GetFloat64(1, 0))
	}
}

func TestT(t *testing.T) {
	arr := FromSliceFloat64([]float64{1, 2, 3, 4}, 2, 2)
	transposed := arr.T()
	
	if transposed.GetFloat64(0, 1) != 3.0 {
		t.Errorf("expected 3.0 at [0,1], got %f", transposed.GetFloat64(0, 1))
	}
	if transposed.GetFloat64(1, 0) != 2.0 {
		t.Errorf("expected 2.0 at [1,0], got %f", transposed.GetFloat64(1, 0))
	}
}

func TestCopy(t *testing.T) {
	arr := FromSliceFloat64([]float64{1, 2, 3}, 3)
	copied := arr.Copy()
	
	// Modify original
	arr.SetFloat64(99, 0)
	
	// Check that copy wasn't affected
	if copied.GetFloat64(0) != 1.0 {
		t.Errorf("expected copy to be unchanged, got %f", copied.GetFloat64(0))
	}
}

func TestFlatten(t *testing.T) {
	arr := FromSliceFloat64([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	flat := arr.Flatten()
	
	if flat.Ndim() != 1 {
		t.Errorf("expected ndim 1, got %d", flat.Ndim())
	}
	
	if flat.Size() != 6 {
		t.Errorf("expected size 6, got %d", flat.Size())
	}
}

func TestSqueeze(t *testing.T) {
	arr := FromSliceFloat64([]float64{1, 2, 3}, 1, 3, 1)
	squeezed := arr.Squeeze()
	
	shape := squeezed.Shape()
	if len(shape) != 1 || shape[0] != 3 {
		t.Errorf("expected shape [3], got %v", shape)
	}
}

func TestToSlice(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	arr := FromSliceFloat64(data, 2, 3)
	
	result := arr.ToSliceFloat64()
	
	for i, val := range result {
		if val != data[i] {
			t.Errorf("expected %f at index %d, got %f", data[i], i, val)
		}
	}
}
