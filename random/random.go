// Package random provides random number generation for NumGo arrays
package random

import (
	"math"
	"math/rand"
	"time"
	
	"github.com/iSundram/NumGo/tensor"
)

// RNG represents a random number generator
type RNG struct {
	source *rand.Rand
}

// New creates a new random number generator with the given seed
func New(seed int64) *RNG {
	return &RNG{
		source: rand.New(rand.NewSource(seed)),
	}
}

// NewDefault creates a new random number generator with a time-based seed
func NewDefault() *RNG {
	return New(time.Now().UnixNano())
}

// Uniform generates random floats from a uniform distribution [low, high)
func (rng *RNG) Uniform(low, high float64, shape ...int) *tensor.NDArray {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	
	data := make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = low + (high-low)*rng.source.Float64()
	}
	
	return tensor.FromSliceFloat64(data, shape...)
}

// Normal generates random floats from a normal (Gaussian) distribution
func (rng *RNG) Normal(mean, std float64, shape ...int) *tensor.NDArray {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	
	data := make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = mean + std*rng.source.NormFloat64()
	}
	
	return tensor.FromSliceFloat64(data, shape...)
}

// StandardNormal generates random floats from a standard normal distribution (mean=0, std=1)
func (rng *RNG) StandardNormal(shape ...int) *tensor.NDArray {
	return rng.Normal(0, 1, shape...)
}

// Binomial generates random integers from a binomial distribution
func (rng *RNG) Binomial(n int, p float64, shape ...int) *tensor.NDArray {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	
	data := make([]float64, size)
	for i := 0; i < size; i++ {
		count := 0
		for j := 0; j < n; j++ {
			if rng.source.Float64() < p {
				count++
			}
		}
		data[i] = float64(count)
	}
	
	return tensor.FromSliceFloat64(data, shape...)
}

// Poisson generates random integers from a Poisson distribution
func (rng *RNG) Poisson(lambda float64, shape ...int) *tensor.NDArray {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	
	data := make([]float64, size)
	L := math.Exp(-lambda)
	
	for i := 0; i < size; i++ {
		k := 0
		p := 1.0
		
		for p > L {
			k++
			p *= rng.source.Float64()
		}
		
		data[i] = float64(k - 1)
	}
	
	return tensor.FromSliceFloat64(data, shape...)
}

// Exponential generates random floats from an exponential distribution
func (rng *RNG) Exponential(scale float64, shape ...int) *tensor.NDArray {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	
	data := make([]float64, size)
	for i := 0; i < size; i++ {
		data[i] = -scale * math.Log(rng.source.Float64())
	}
	
	return tensor.FromSliceFloat64(data, shape...)
}

// Gamma generates random floats from a gamma distribution
// Uses the Marsaglia and Tsang method
func (rng *RNG) Gamma(shape, scale float64, size ...int) *tensor.NDArray {
	n := 1
	for _, dim := range size {
		n *= dim
	}
	
	data := make([]float64, n)
	
	d := shape - 1.0/3.0
	c := 1.0 / math.Sqrt(9.0*d)
	
	for i := 0; i < n; i++ {
		for {
			x := rng.source.NormFloat64()
			v := 1.0 + c*x
			
			if v <= 0 {
				continue
			}
			
			v = v * v * v
			u := rng.source.Float64()
			
			if u < 1.0-0.0331*(x*x)*(x*x) {
				data[i] = scale * d * v
				break
			}
			
			if math.Log(u) < 0.5*x*x+d*(1.0-v+math.Log(v)) {
				data[i] = scale * d * v
				break
			}
		}
	}
	
	return tensor.FromSliceFloat64(data, size...)
}

// Beta generates random floats from a beta distribution
func (rng *RNG) Beta(alpha, beta float64, shape ...int) *tensor.NDArray {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	
	// Use the property: if X ~ Gamma(alpha, 1) and Y ~ Gamma(beta, 1)
	// then X/(X+Y) ~ Beta(alpha, beta)
	
	data := make([]float64, size)
	gammaAlpha := rng.Gamma(alpha, 1.0, size)
	gammaBeta := rng.Gamma(beta, 1.0, size)
	
	for i := 0; i < size; i++ {
		x := gammaAlpha.GetFloat64(i)
		y := gammaBeta.GetFloat64(i)
		data[i] = x / (x + y)
	}
	
	return tensor.FromSliceFloat64(data, shape...)
}

// Rand generates random floats from a uniform distribution [0, 1)
func (rng *RNG) Rand(shape ...int) *tensor.NDArray {
	return rng.Uniform(0, 1, shape...)
}

// Randint generates random integers in [low, high)
func (rng *RNG) Randint(low, high int, shape ...int) *tensor.NDArray {
	size := 1
	for _, dim := range shape {
		size *= dim
	}
	
	data := make([]int64, size)
	span := high - low
	
	for i := 0; i < size; i++ {
		data[i] = int64(low + rng.source.Intn(span))
	}
	
	return tensor.FromSliceInt64(data, shape...)
}

// Choice randomly selects elements from an array
func (rng *RNG) Choice(arr *tensor.NDArray, size int) *tensor.NDArray {
	n := arr.Size()
	data := make([]float64, size)
	
	for i := 0; i < size; i++ {
		idx := rng.source.Intn(n)
		indices := make([]int, arr.Ndim())
		remaining := idx
		for j := arr.Ndim() - 1; j >= 0; j-- {
			indices[j] = remaining % arr.Shape()[j]
			remaining /= arr.Shape()[j]
		}
		data[i] = arr.GetFloat64(indices...)
	}
	
	return tensor.FromSliceFloat64(data, size)
}

// Permutation returns a random permutation of integers [0, n)
func (rng *RNG) Permutation(n int) *tensor.NDArray {
	data := make([]int64, n)
	for i := 0; i < n; i++ {
		data[i] = int64(i)
	}
	
	// Fisher-Yates shuffle
	for i := n - 1; i > 0; i-- {
		j := rng.source.Intn(i + 1)
		data[i], data[j] = data[j], data[i]
	}
	
	return tensor.FromSliceInt64(data, n)
}

// Shuffle randomly shuffles an array in-place (modifies the array)
func (rng *RNG) Shuffle(arr *tensor.NDArray) {
	n := arr.Size()
	
	for i := n - 1; i > 0; i-- {
		j := rng.source.Intn(i + 1)
		
		// Get indices for both positions
		indicesI := make([]int, arr.Ndim())
		indicesJ := make([]int, arr.Ndim())
		
		remainingI := i
		remainingJ := j
		
		for k := arr.Ndim() - 1; k >= 0; k-- {
			indicesI[k] = remainingI % arr.Shape()[k]
			remainingI /= arr.Shape()[k]
			
			indicesJ[k] = remainingJ % arr.Shape()[k]
			remainingJ /= arr.Shape()[k]
		}
		
		// Swap values
		valI := arr.GetFloat64(indicesI...)
		valJ := arr.GetFloat64(indicesJ...)
		arr.SetFloat64(valJ, indicesI...)
		arr.SetFloat64(valI, indicesJ...)
	}
}

// Seed sets the seed for the random number generator
func (rng *RNG) Seed(seed int64) {
	rng.source = rand.New(rand.NewSource(seed))
}
