package linalg

import (
	"math"
	"testing"
	
	"github.com/iSundram/NumGo/tensor"
)

func TestDotVectors(t *testing.T) {
	a := tensor.FromSliceFloat64([]float64{1, 2, 3}, 3)
	b := tensor.FromSliceFloat64([]float64{4, 5, 6}, 3)
	
	result := Dot(a, b)
	
	// 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
	expected := 32.0
	val := result.GetFloat64(0)
	
	if val != expected {
		t.Errorf("expected %f, got %f", expected, val)
	}
}

func TestMatMul(t *testing.T) {
	// 2x3 matrix
	a := tensor.FromSliceFloat64([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	// 3x2 matrix
	b := tensor.FromSliceFloat64([]float64{7, 8, 9, 10, 11, 12}, 3, 2)
	
	result := MatMul(a, b)
	
	shape := result.Shape()
	if shape[0] != 2 || shape[1] != 2 {
		t.Errorf("expected shape [2,2], got %v", shape)
	}
	
	// Expected result:
	// [1*7+2*9+3*11,  1*8+2*10+3*12]   = [58, 64]
	// [4*7+5*9+6*11,  4*8+5*10+6*12]   = [139, 154]
	
	if result.GetFloat64(0, 0) != 58 {
		t.Errorf("expected 58 at [0,0], got %f", result.GetFloat64(0, 0))
	}
	if result.GetFloat64(0, 1) != 64 {
		t.Errorf("expected 64 at [0,1], got %f", result.GetFloat64(0, 1))
	}
	if result.GetFloat64(1, 0) != 139 {
		t.Errorf("expected 139 at [1,0], got %f", result.GetFloat64(1, 0))
	}
	if result.GetFloat64(1, 1) != 154 {
		t.Errorf("expected 154 at [1,1], got %f", result.GetFloat64(1, 1))
	}
}

func TestMatrixVectorMul(t *testing.T) {
	// 2x3 matrix
	a := tensor.FromSliceFloat64([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	// 3x1 vector
	b := tensor.FromSliceFloat64([]float64{1, 2, 3}, 3)
	
	result := Dot(a, b)
	
	// [1*1+2*2+3*3]  = [14]
	// [4*1+5*2+6*3]  = [32]
	
	if result.GetFloat64(0) != 14 {
		t.Errorf("expected 14 at [0], got %f", result.GetFloat64(0))
	}
	if result.GetFloat64(1) != 32 {
		t.Errorf("expected 32 at [1], got %f", result.GetFloat64(1))
	}
}

func TestOuter(t *testing.T) {
	a := tensor.FromSliceFloat64([]float64{1, 2}, 2)
	b := tensor.FromSliceFloat64([]float64{3, 4, 5}, 3)
	
	result := Outer(a, b)
	
	shape := result.Shape()
	if shape[0] != 2 || shape[1] != 3 {
		t.Errorf("expected shape [2,3], got %v", shape)
	}
	
	// Expected:
	// [1*3, 1*4, 1*5] = [3, 4, 5]
	// [2*3, 2*4, 2*5] = [6, 8, 10]
	
	if result.GetFloat64(0, 0) != 3 {
		t.Errorf("expected 3 at [0,0], got %f", result.GetFloat64(0, 0))
	}
	if result.GetFloat64(1, 2) != 10 {
		t.Errorf("expected 10 at [1,2], got %f", result.GetFloat64(1, 2))
	}
}

func TestInner(t *testing.T) {
	a := tensor.FromSliceFloat64([]float64{1, 2, 3}, 3)
	b := tensor.FromSliceFloat64([]float64{4, 5, 6}, 3)
	
	result := Inner(a, b)
	
	// 1*4 + 2*5 + 3*6 = 32
	expected := 32.0
	
	if result != expected {
		t.Errorf("expected %f, got %f", expected, result)
	}
}

func TestNorm(t *testing.T) {
	a := tensor.FromSliceFloat64([]float64{3, 4}, 2)
	
	norm := Norm(a)
	
	// sqrt(3^2 + 4^2) = sqrt(25) = 5
	expected := 5.0
	
	if math.Abs(norm-expected) > 1e-10 {
		t.Errorf("expected %f, got %f", expected, norm)
	}
}

func TestTrace(t *testing.T) {
	a := tensor.FromSliceFloat64([]float64{1, 2, 3, 4, 5, 6, 7, 8, 9}, 3, 3)
	
	trace := Trace(a)
	
	// 1 + 5 + 9 = 15
	expected := 15.0
	
	if trace != expected {
		t.Errorf("expected %f, got %f", expected, trace)
	}
}

func TestDet2x2(t *testing.T) {
	a := tensor.FromSliceFloat64([]float64{1, 2, 3, 4}, 2, 2)
	
	det := Det(a)
	
	// 1*4 - 2*3 = 4 - 6 = -2
	expected := -2.0
	
	if det != expected {
		t.Errorf("expected %f, got %f", expected, det)
	}
}

func TestDet3x3(t *testing.T) {
	a := tensor.FromSliceFloat64([]float64{
		1, 2, 3,
		0, 1, 4,
		5, 6, 0,
	}, 3, 3)
	
	det := Det(a)
	
	// Determinant = 1*(1*0-4*6) - 2*(0*0-4*5) + 3*(0*6-1*5)
	//             = 1*(-24) - 2*(-20) + 3*(-5)
	//             = -24 + 40 - 15 = 1
	expected := 1.0
	
	if math.Abs(det-expected) > 1e-10 {
		t.Errorf("expected %f, got %f", expected, det)
	}
}

func TestInv2x2(t *testing.T) {
	// Simple 2x2 matrix
	a := tensor.FromSliceFloat64([]float64{4, 7, 2, 6}, 2, 2)
	
	inv := Inv(a)
	
	// Verify A * A^-1 = I
	identity := MatMul(a, inv)
	
	// Check diagonal elements are close to 1
	if math.Abs(identity.GetFloat64(0, 0)-1.0) > 1e-10 {
		t.Errorf("expected 1.0 at [0,0], got %f", identity.GetFloat64(0, 0))
	}
	if math.Abs(identity.GetFloat64(1, 1)-1.0) > 1e-10 {
		t.Errorf("expected 1.0 at [1,1], got %f", identity.GetFloat64(1, 1))
	}
	
	// Check off-diagonal elements are close to 0
	if math.Abs(identity.GetFloat64(0, 1)) > 1e-10 {
		t.Errorf("expected 0.0 at [0,1], got %f", identity.GetFloat64(0, 1))
	}
	if math.Abs(identity.GetFloat64(1, 0)) > 1e-10 {
		t.Errorf("expected 0.0 at [1,0], got %f", identity.GetFloat64(1, 0))
	}
}

func TestInv3x3(t *testing.T) {
	// 3x3 matrix
	a := tensor.FromSliceFloat64([]float64{
		1, 2, 3,
		0, 1, 4,
		5, 6, 0,
	}, 3, 3)
	
	inv := Inv(a)
	
	// Verify A * A^-1 = I
	identity := MatMul(a, inv)
	
	// Check it's close to identity matrix
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			expected := 0.0
			if i == j {
				expected = 1.0
			}
			val := identity.GetFloat64(i, j)
			if math.Abs(val-expected) > 1e-9 {
				t.Errorf("expected %f at [%d,%d], got %f", expected, i, j, val)
			}
		}
	}
}

func TestTranspose(t *testing.T) {
	a := tensor.FromSliceFloat64([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	
	result := Transpose(a)
	
	shape := result.Shape()
	if shape[0] != 3 || shape[1] != 2 {
		t.Errorf("expected shape [3,2], got %v", shape)
	}
	
	if result.GetFloat64(0, 0) != 1 {
		t.Errorf("expected 1 at [0,0], got %f", result.GetFloat64(0, 0))
	}
	if result.GetFloat64(1, 0) != 2 {
		t.Errorf("expected 2 at [1,0], got %f", result.GetFloat64(1, 0))
	}
}
