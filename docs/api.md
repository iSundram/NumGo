# NumGo API Documentation

## Core Package: tensor

### Array Creation

#### Zeros
```go
func Zeros(shape []int, dtype DType) *NDArray
```
Creates a new array filled with zeros.

#### Ones
```go
func Ones(shape []int, dtype DType) *NDArray
```
Creates a new array filled with ones.

#### FromSliceFloat64
```go
func FromSliceFloat64(data []float64, shape ...int) *NDArray
```
Creates an array from a float64 slice with the specified shape.

#### Arange
```go
func Arange(start, stop, step float64) *NDArray
```
Creates an array with evenly spaced values within a given interval.

#### Range
```go
func Range(start, stop int) *NDArray
```
Creates an array with integers from start to stop (exclusive).

#### Eye
```go
func Eye(n int, dtype DType) *NDArray
```
Creates a 2-D array with ones on the diagonal and zeros elsewhere.

### Array Properties

- `Shape() []int` - Returns the shape of the array
- `Size() int` - Returns the total number of elements
- `Ndim() int` - Returns the number of dimensions
- `DType() DType` - Returns the data type
- `ItemSize() int` - Returns the size of one element in bytes

### Indexing and Slicing

- `GetFloat64(indices ...int) float64` - Get element as float64
- `GetInt64(indices ...int) int64` - Get element as int64
- `SetFloat64(value float64, indices ...int)` - Set element from float64
- `SetInt64(value int64, indices ...int)` - Set element from int64

### Shape Operations

- `Reshape(newShape ...int) *NDArray` - Returns array with new shape
- `Transpose(axes ...int) *NDArray` - Returns transposed array
- `T() *NDArray` - Returns transpose of 2D array
- `Flatten() *NDArray` - Returns flattened 1D array
- `Squeeze() *NDArray` - Removes single-dimensional entries
- `Copy() *NDArray` - Creates a deep copy

### Arithmetic Operations

- `Add(b *NDArray) *NDArray` - Element-wise addition
- `Sub(b *NDArray) *NDArray` - Element-wise subtraction
- `Mul(b *NDArray) *NDArray` - Element-wise multiplication
- `Div(b *NDArray) *NDArray` - Element-wise division
- `AddScalar(scalar float64) *NDArray` - Add scalar to all elements
- `MulScalar(scalar float64) *NDArray` - Multiply all elements by scalar

### Math Functions

- `Exp() *NDArray` - Exponential (e^x)
- `Log() *NDArray` - Natural logarithm
- `Sin() *NDArray` - Sine
- `Cos() *NDArray` - Cosine
- `Sqrt() *NDArray` - Square root
- `Pow(exponent float64) *NDArray` - Power
- `Abs() *NDArray` - Absolute value
- `Neg() *NDArray` - Negation

### Reductions

- `Sum() float64` - Sum of all elements
- `Mean() float64` - Mean of all elements
- `Min() float64` - Minimum value
- `Max() float64` - Maximum value
- `ArgMin() int` - Index of minimum value
- `ArgMax() int` - Index of maximum value
- `Prod() float64` - Product of all elements
- `Var() float64` - Variance
- `Std() float64` - Standard deviation
- `All() bool` - Returns true if all elements are non-zero
- `Any() bool` - Returns true if any element is non-zero

### Axis-based Reductions

- `SumAxis(axis int) *NDArray` - Sum along specified axis
- `MeanAxis(axis int) *NDArray` - Mean along specified axis

## Linear Algebra Package: linalg

### Basic Operations

#### Dot
```go
func Dot(a, b *NDArray) *NDArray
```
Computes the dot product of two arrays.

#### MatMul
```go
func MatMul(a, b *NDArray) *NDArray
```
Performs matrix multiplication.

#### Outer
```go
func Outer(a, b *NDArray) *NDArray
```
Computes the outer product of two vectors.

#### Inner
```go
func Inner(a, b *NDArray) float64
```
Computes the inner product of two vectors.

### Matrix Operations

#### Transpose
```go
func Transpose(a *NDArray) *NDArray
```
Returns the transpose of a matrix.

#### Trace
```go
func Trace(a *NDArray) float64
```
Computes the sum of diagonal elements.

#### Det
```go
func Det(a *NDArray) float64
```
Computes the determinant of a square matrix.

#### Inv
```go
func Inv(a *NDArray) *NDArray
```
Computes the inverse of a square matrix.

#### Norm
```go
func Norm(a *NDArray) float64
```
Computes the L2 (Euclidean) norm.

## Random Package: random

### RNG Creation

```go
func New(seed int64) *RNG
func NewDefault() *RNG
```

### Distributions

#### Uniform
```go
func (rng *RNG) Uniform(low, high float64, shape ...int) *NDArray
```
Generates random floats from a uniform distribution [low, high).

#### Normal
```go
func (rng *RNG) Normal(mean, std float64, shape ...int) *NDArray
```
Generates random floats from a normal (Gaussian) distribution.

#### StandardNormal
```go
func (rng *RNG) StandardNormal(shape ...int) *NDArray
```
Generates random floats from a standard normal distribution (mean=0, std=1).

#### Binomial
```go
func (rng *RNG) Binomial(n int, p float64, shape ...int) *NDArray
```
Generates random integers from a binomial distribution.

#### Poisson
```go
func (rng *RNG) Poisson(lambda float64, shape ...int) *NDArray
```
Generates random integers from a Poisson distribution.

#### Exponential
```go
func (rng *RNG) Exponential(scale float64, shape ...int) *NDArray
```
Generates random floats from an exponential distribution.

#### Gamma
```go
func (rng *RNG) Gamma(shape, scale float64, size ...int) *NDArray
```
Generates random floats from a gamma distribution.

#### Beta
```go
func (rng *RNG) Beta(alpha, beta float64, shape ...int) *NDArray
```
Generates random floats from a beta distribution.

### Sampling

#### Rand
```go
func (rng *RNG) Rand(shape ...int) *NDArray
```
Generates random floats from a uniform distribution [0, 1).

#### Randint
```go
func (rng *RNG) Randint(low, high int, shape ...int) *NDArray
```
Generates random integers in [low, high).

#### Choice
```go
func (rng *RNG) Choice(arr *NDArray, size int) *NDArray
```
Randomly selects elements from an array.

#### Permutation
```go
func (rng *RNG) Permutation(n int) *NDArray
```
Returns a random permutation of integers [0, n).

#### Shuffle
```go
func (rng *RNG) Shuffle(arr *NDArray)
```
Randomly shuffles an array in-place.

## Data Types

The following data types are supported:

- `Bool` - Boolean
- `Int8`, `Int16`, `Int32`, `Int64` - Signed integers
- `Uint8`, `Uint16`, `Uint32`, `Uint64` - Unsigned integers
- `Float32`, `Float64` - Floating point
- `Complex64`, `Complex128` - Complex numbers
