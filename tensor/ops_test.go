package tensor

import (
	"math"
	"testing"
)

func TestAdd(t *testing.T) {
	a := FromSliceFloat64([]float64{1, 2, 3}, 3)
	b := FromSliceFloat64([]float64{4, 5, 6}, 3)
	
	c := a.Add(b)
	
	expected := []float64{5, 7, 9}
	for i := 0; i < 3; i++ {
		val := c.GetFloat64(i)
		if val != expected[i] {
			t.Errorf("expected %f at index %d, got %f", expected[i], i, val)
		}
	}
}

func TestBroadcasting(t *testing.T) {
	// 2x3 array + 1x3 array (should broadcast)
	a := FromSliceFloat64([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	b := FromSliceFloat64([]float64{10, 20, 30}, 1, 3)
	
	c := a.Add(b)
	
	// First row: [1,2,3] + [10,20,30] = [11,22,33]
	// Second row: [4,5,6] + [10,20,30] = [14,25,36]
	if c.GetFloat64(0, 0) != 11 {
		t.Errorf("expected 11, got %f", c.GetFloat64(0, 0))
	}
	if c.GetFloat64(1, 2) != 36 {
		t.Errorf("expected 36, got %f", c.GetFloat64(1, 2))
	}
}

func TestSub(t *testing.T) {
	a := FromSliceFloat64([]float64{5, 7, 9}, 3)
	b := FromSliceFloat64([]float64{1, 2, 3}, 3)
	
	c := a.Sub(b)
	
	expected := []float64{4, 5, 6}
	for i := 0; i < 3; i++ {
		val := c.GetFloat64(i)
		if val != expected[i] {
			t.Errorf("expected %f at index %d, got %f", expected[i], i, val)
		}
	}
}

func TestMul(t *testing.T) {
	a := FromSliceFloat64([]float64{2, 3, 4}, 3)
	b := FromSliceFloat64([]float64{5, 6, 7}, 3)
	
	c := a.Mul(b)
	
	expected := []float64{10, 18, 28}
	for i := 0; i < 3; i++ {
		val := c.GetFloat64(i)
		if val != expected[i] {
			t.Errorf("expected %f at index %d, got %f", expected[i], i, val)
		}
	}
}

func TestDiv(t *testing.T) {
	a := FromSliceFloat64([]float64{10, 20, 30}, 3)
	b := FromSliceFloat64([]float64{2, 4, 5}, 3)
	
	c := a.Div(b)
	
	expected := []float64{5, 5, 6}
	for i := 0; i < 3; i++ {
		val := c.GetFloat64(i)
		if val != expected[i] {
			t.Errorf("expected %f at index %d, got %f", expected[i], i, val)
		}
	}
}

func TestScalarOps(t *testing.T) {
	a := FromSliceFloat64([]float64{1, 2, 3}, 3)
	
	// Add scalar
	b := a.AddScalar(10)
	if b.GetFloat64(0) != 11 {
		t.Errorf("expected 11, got %f", b.GetFloat64(0))
	}
	
	// Mul scalar
	c := a.MulScalar(2)
	if c.GetFloat64(1) != 4 {
		t.Errorf("expected 4, got %f", c.GetFloat64(1))
	}
}

func TestMathFunctions(t *testing.T) {
	a := FromSliceFloat64([]float64{0, 1, 2}, 3)
	
	// Exp
	exp := a.Exp()
	if math.Abs(exp.GetFloat64(0)-1.0) > 1e-10 {
		t.Errorf("expected e^0=1, got %f", exp.GetFloat64(0))
	}
	
	// Sqrt
	b := FromSliceFloat64([]float64{4, 9, 16}, 3)
	sqrt := b.Sqrt()
	expected := []float64{2, 3, 4}
	for i := 0; i < 3; i++ {
		if sqrt.GetFloat64(i) != expected[i] {
			t.Errorf("expected %f at index %d, got %f", expected[i], i, sqrt.GetFloat64(i))
		}
	}
	
	// Sin
	c := FromSliceFloat64([]float64{0, math.Pi / 2}, 2)
	sin := c.Sin()
	if math.Abs(sin.GetFloat64(0)-0) > 1e-10 {
		t.Errorf("expected sin(0)=0, got %f", sin.GetFloat64(0))
	}
	if math.Abs(sin.GetFloat64(1)-1) > 1e-10 {
		t.Errorf("expected sin(π/2)=1, got %f", sin.GetFloat64(1))
	}
}

func TestSum(t *testing.T) {
	a := FromSliceFloat64([]float64{1, 2, 3, 4, 5}, 5)
	sum := a.Sum()
	
	if sum != 15.0 {
		t.Errorf("expected sum 15, got %f", sum)
	}
}

func TestMean(t *testing.T) {
	a := FromSliceFloat64([]float64{1, 2, 3, 4, 5}, 5)
	mean := a.Mean()
	
	if mean != 3.0 {
		t.Errorf("expected mean 3, got %f", mean)
	}
}

func TestMinMax(t *testing.T) {
	a := FromSliceFloat64([]float64{5, 2, 8, 1, 9}, 5)
	
	min := a.Min()
	if min != 1.0 {
		t.Errorf("expected min 1, got %f", min)
	}
	
	max := a.Max()
	if max != 9.0 {
		t.Errorf("expected max 9, got %f", max)
	}
}

func TestArgMinArgMax(t *testing.T) {
	a := FromSliceFloat64([]float64{5, 2, 8, 1, 9}, 5)
	
	argmin := a.ArgMin()
	if argmin != 3 {
		t.Errorf("expected argmin 3, got %d", argmin)
	}
	
	argmax := a.ArgMax()
	if argmax != 4 {
		t.Errorf("expected argmax 4, got %d", argmax)
	}
}

func TestVar(t *testing.T) {
	a := FromSliceFloat64([]float64{2, 4, 6, 8}, 4)
	variance := a.Var()
	
	// Mean = 5, variance = ((2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2) / 4 = (9+1+1+9)/4 = 5
	expected := 5.0
	if math.Abs(variance-expected) > 1e-10 {
		t.Errorf("expected variance %f, got %f", expected, variance)
	}
}

func TestStd(t *testing.T) {
	a := FromSliceFloat64([]float64{2, 4, 6, 8}, 4)
	std := a.Std()
	
	// std = sqrt(5) ≈ 2.236
	expected := math.Sqrt(5.0)
	if math.Abs(std-expected) > 1e-10 {
		t.Errorf("expected std %f, got %f", expected, std)
	}
}

func TestProd(t *testing.T) {
	a := FromSliceFloat64([]float64{2, 3, 4}, 3)
	prod := a.Prod()
	
	if prod != 24.0 {
		t.Errorf("expected product 24, got %f", prod)
	}
}

func TestAllAny(t *testing.T) {
	a := FromSliceFloat64([]float64{1, 2, 3}, 3)
	if !a.All() {
		t.Error("expected All() to be true")
	}
	
	b := FromSliceFloat64([]float64{1, 0, 3}, 3)
	if b.All() {
		t.Error("expected All() to be false")
	}
	if !b.Any() {
		t.Error("expected Any() to be true")
	}
	
	c := FromSliceFloat64([]float64{0, 0, 0}, 3)
	if c.Any() {
		t.Error("expected Any() to be false")
	}
}

func TestSumAxis(t *testing.T) {
	// 2x3 array
	a := FromSliceFloat64([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	
	// Sum along axis 0 (sum rows -> result is 1x3)
	sum0 := a.SumAxis(0)
	expected0 := []float64{5, 7, 9} // [1+4, 2+5, 3+6]
	
	for i := 0; i < 3; i++ {
		if sum0.GetFloat64(i) != expected0[i] {
			t.Errorf("SumAxis(0): expected %f at index %d, got %f", expected0[i], i, sum0.GetFloat64(i))
		}
	}
	
	// Sum along axis 1 (sum columns -> result is 2x1)
	sum1 := a.SumAxis(1)
	expected1 := []float64{6, 15} // [1+2+3, 4+5+6]
	
	for i := 0; i < 2; i++ {
		if sum1.GetFloat64(i) != expected1[i] {
			t.Errorf("SumAxis(1): expected %f at index %d, got %f", expected1[i], i, sum1.GetFloat64(i))
		}
	}
}

func TestMeanAxis(t *testing.T) {
	a := FromSliceFloat64([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	
	// Mean along axis 1
	mean1 := a.MeanAxis(1)
	expected := []float64{2, 5} // [(1+2+3)/3, (4+5+6)/3]
	
	for i := 0; i < 2; i++ {
		if mean1.GetFloat64(i) != expected[i] {
			t.Errorf("expected %f at index %d, got %f", expected[i], i, mean1.GetFloat64(i))
		}
	}
}
