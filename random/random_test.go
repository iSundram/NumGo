package random

import (
	"math"
	"testing"
	
	"github.com/iSundram/NumGo/tensor"
)

func TestUniform(t *testing.T) {
	rng := New(42)
	arr := rng.Uniform(0, 10, 100)
	
	if arr.Size() != 100 {
		t.Errorf("expected size 100, got %d", arr.Size())
	}
	
	// Check all values are in range [0, 10)
	for i := 0; i < 100; i++ {
		val := arr.GetFloat64(i)
		if val < 0 || val >= 10 {
			t.Errorf("value %f out of range [0, 10)", val)
		}
	}
}

func TestNormal(t *testing.T) {
	rng := New(42)
	arr := rng.Normal(5, 2, 1000)
	
	if arr.Size() != 1000 {
		t.Errorf("expected size 1000, got %d", arr.Size())
	}
	
	// Check mean is close to 5
	mean := arr.Mean()
	if math.Abs(mean-5) > 0.5 {
		t.Errorf("expected mean close to 5, got %f", mean)
	}
	
	// Check std is close to 2
	std := arr.Std()
	if math.Abs(std-2) > 0.5 {
		t.Errorf("expected std close to 2, got %f", std)
	}
}

func TestStandardNormal(t *testing.T) {
	rng := New(42)
	arr := rng.StandardNormal(1000)
	
	// Check mean is close to 0
	mean := arr.Mean()
	if math.Abs(mean) > 0.2 {
		t.Errorf("expected mean close to 0, got %f", mean)
	}
	
	// Check std is close to 1
	std := arr.Std()
	if math.Abs(std-1) > 0.2 {
		t.Errorf("expected std close to 1, got %f", std)
	}
}

func TestBinomial(t *testing.T) {
	rng := New(42)
	arr := rng.Binomial(10, 0.5, 1000)
	
	// Mean should be close to n*p = 10*0.5 = 5
	mean := arr.Mean()
	if math.Abs(mean-5) > 0.5 {
		t.Errorf("expected mean close to 5, got %f", mean)
	}
}

func TestPoisson(t *testing.T) {
	rng := New(42)
	arr := rng.Poisson(5, 1000)
	
	// Mean should be close to lambda = 5
	mean := arr.Mean()
	if math.Abs(mean-5) > 0.5 {
		t.Errorf("expected mean close to 5, got %f", mean)
	}
}

func TestExponential(t *testing.T) {
	rng := New(42)
	arr := rng.Exponential(2, 1000)
	
	// Mean should be close to scale = 2
	mean := arr.Mean()
	if math.Abs(mean-2) > 0.3 {
		t.Errorf("expected mean close to 2, got %f", mean)
	}
}

func TestGamma(t *testing.T) {
	rng := New(42)
	arr := rng.Gamma(2, 2, 1000)
	
	// Mean should be close to shape*scale = 2*2 = 4
	mean := arr.Mean()
	if math.Abs(mean-4) > 0.5 {
		t.Errorf("expected mean close to 4, got %f", mean)
	}
}

func TestBeta(t *testing.T) {
	rng := New(42)
	arr := rng.Beta(2, 5, 1000)
	
	// Mean should be close to alpha/(alpha+beta) = 2/7 ≈ 0.286
	mean := arr.Mean()
	expected := 2.0 / 7.0
	if math.Abs(mean-expected) > 0.1 {
		t.Errorf("expected mean close to %f, got %f", expected, mean)
	}
	
	// All values should be in [0, 1]
	for i := 0; i < arr.Size(); i++ {
		val := arr.GetFloat64(i)
		if val < 0 || val > 1 {
			t.Errorf("value %f out of range [0, 1]", val)
		}
	}
}

func TestRand(t *testing.T) {
	rng := New(42)
	arr := rng.Rand(100)
	
	// All values should be in [0, 1)
	for i := 0; i < 100; i++ {
		val := arr.GetFloat64(i)
		if val < 0 || val >= 1 {
			t.Errorf("value %f out of range [0, 1)", val)
		}
	}
}

func TestRandint(t *testing.T) {
	rng := New(42)
	arr := rng.Randint(0, 10, 100)
	
	// All values should be in [0, 10)
	for i := 0; i < 100; i++ {
		val := arr.GetInt64(i)
		if val < 0 || val >= 10 {
			t.Errorf("value %d out of range [0, 10)", val)
		}
	}
}

func TestChoice(t *testing.T) {
	rng := New(42)
	arr := tensor.FromSliceFloat64([]float64{1, 2, 3, 4, 5}, 5)
	
	choices := rng.Choice(arr, 100)
	
	if choices.Size() != 100 {
		t.Errorf("expected size 100, got %d", choices.Size())
	}
	
	// All values should be from the original array
	for i := 0; i < 100; i++ {
		val := choices.GetFloat64(i)
		valid := false
		for j := 0; j < 5; j++ {
			if val == float64(j+1) {
				valid = true
				break
			}
		}
		if !valid {
			t.Errorf("value %f not in original array", val)
		}
	}
}

func TestPermutation(t *testing.T) {
	rng := New(42)
	perm := rng.Permutation(10)
	
	if perm.Size() != 10 {
		t.Errorf("expected size 10, got %d", perm.Size())
	}
	
	// Check that all values [0, 9] appear exactly once
	seen := make(map[int64]bool)
	for i := 0; i < 10; i++ {
		val := perm.GetInt64(i)
		if val < 0 || val >= 10 {
			t.Errorf("value %d out of range [0, 10)", val)
		}
		if seen[val] {
			t.Errorf("duplicate value %d", val)
		}
		seen[val] = true
	}
}

func TestShuffle(t *testing.T) {
	rng := New(42)
	arr := tensor.FromSliceFloat64([]float64{1, 2, 3, 4, 5}, 5)
	
	original := arr.ToSliceFloat64()
	rng.Shuffle(arr)
	shuffled := arr.ToSliceFloat64()
	
	// Check that the arrays are different (very likely with seed 42)
	same := true
	for i := 0; i < 5; i++ {
		if original[i] != shuffled[i] {
			same = false
			break
		}
	}
	if same {
		t.Error("array was not shuffled")
	}
	
	// Check that all original values are still present
	for i := 0; i < 5; i++ {
		found := false
		for j := 0; j < 5; j++ {
			if original[i] == shuffled[j] {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("value %f not found after shuffle", original[i])
		}
	}
}

func TestSeed(t *testing.T) {
	rng1 := New(123)
	rng2 := New(123)
	
	arr1 := rng1.Uniform(0, 1, 10)
	arr2 := rng2.Uniform(0, 1, 10)
	
	// Same seed should produce same values
	for i := 0; i < 10; i++ {
		if arr1.GetFloat64(i) != arr2.GetFloat64(i) {
			t.Errorf("same seed produced different values at index %d", i)
		}
	}
}

func TestMultiDimensional(t *testing.T) {
	rng := New(42)
	arr := rng.Normal(0, 1, 3, 4, 5)
	
	shape := arr.Shape()
	if len(shape) != 3 || shape[0] != 3 || shape[1] != 4 || shape[2] != 5 {
		t.Errorf("expected shape [3,4,5], got %v", shape)
	}
	
	if arr.Size() != 60 {
		t.Errorf("expected size 60, got %d", arr.Size())
	}
}
