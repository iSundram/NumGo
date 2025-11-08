# NumPy to NumGo Migration Guide

This guide helps NumPy users transition to NumGo, highlighting the similarities and differences.

## Basic Array Creation

| NumPy | NumGo |
|-------|-------|
| `np.array([1, 2, 3])` | `tensor.FromSliceFloat64([]float64{1, 2, 3}, 3)` |
| `np.zeros((3, 4))` | `tensor.Zeros([]int{3, 4}, tensor.Float64)` |
| `np.ones((2, 3))` | `tensor.Ones([]int{2, 3}, tensor.Float64)` |
| `np.arange(0, 10, 2)` | `tensor.Arange(0, 10, 2)` |
| `np.arange(5)` | `tensor.Range(0, 5)` |
| `np.eye(3)` | `tensor.Eye(3, tensor.Float64)` |
| `np.full((2, 2), 7)` | `tensor.Full([]int{2, 2}, 7, tensor.Float64)` |

## Array Properties

| NumPy | NumGo |
|-------|-------|
| `a.shape` | `a.Shape()` |
| `a.size` | `a.Size()` |
| `a.ndim` | `a.Ndim()` |
| `a.dtype` | `a.DType()` |
| `a.itemsize` | `a.ItemSize()` |

## Indexing and Slicing

| NumPy | NumGo |
|-------|-------|
| `a[0]` | `a.GetFloat64(0)` or `a.GetInt64(0)` |
| `a[i, j]` | `a.GetFloat64(i, j)` |
| `a[0] = 5` | `a.SetFloat64(5, 0)` |
| `a[-1]` | `a.GetFloat64(-1)` (negative indexing supported) |

## Shape Manipulation

| NumPy | NumGo |
|-------|-------|
| `a.reshape(2, 3)` | `a.Reshape(2, 3)` |
| `a.reshape(-1, 3)` | `a.Reshape(-1, 3)` |
| `a.T` | `a.T()` or `a.Transpose()` |
| `a.flatten()` | `a.Flatten()` |
| `a.ravel()` | `a.Ravel()` |
| `a.squeeze()` | `a.Squeeze()` |
| `a.copy()` | `a.Copy()` |
| `np.transpose(a)` | `a.Transpose()` |

## Arithmetic Operations

| NumPy | NumGo |
|-------|-------|
| `a + b` | `a.Add(b)` |
| `a - b` | `a.Sub(b)` |
| `a * b` | `a.Mul(b)` |
| `a / b` | `a.Div(b)` |
| `a + 5` | `a.AddScalar(5)` |
| `a * 2.5` | `a.MulScalar(2.5)` |

## Math Functions

| NumPy | NumGo |
|-------|-------|
| `np.exp(a)` | `a.Exp()` |
| `np.log(a)` | `a.Log()` |
| `np.sin(a)` | `a.Sin()` |
| `np.cos(a)` | `a.Cos()` |
| `np.sqrt(a)` | `a.Sqrt()` |
| `np.power(a, 2)` | `a.Pow(2)` |
| `np.abs(a)` | `a.Abs()` |
| `-a` | `a.Neg()` |

## Reductions

| NumPy | NumGo |
|-------|-------|
| `a.sum()` | `a.Sum()` |
| `a.mean()` | `a.Mean()` |
| `a.min()` | `a.Min()` |
| `a.max()` | `a.Max()` |
| `a.argmin()` | `a.ArgMin()` |
| `a.argmax()` | `a.ArgMax()` |
| `a.prod()` | `a.Prod()` |
| `a.var()` | `a.Var()` |
| `a.std()` | `a.Std()` |
| `a.all()` | `a.All()` |
| `a.any()` | `a.Any()` |
| `a.sum(axis=0)` | `a.SumAxis(0)` |
| `a.mean(axis=1)` | `a.MeanAxis(1)` |

## Linear Algebra

| NumPy | NumGo |
|-------|-------|
| `np.dot(a, b)` | `linalg.Dot(a, b)` |
| `a @ b` or `np.matmul(a, b)` | `linalg.MatMul(a, b)` |
| `np.outer(a, b)` | `linalg.Outer(a, b)` |
| `np.inner(a, b)` | `linalg.Inner(a, b)` |
| `np.linalg.norm(a)` | `linalg.Norm(a)` |
| `np.trace(a)` | `linalg.Trace(a)` |
| `np.linalg.det(a)` | `linalg.Det(a)` |
| `np.linalg.inv(a)` | `linalg.Inv(a)` |

## Random Numbers

### Setup

**NumPy:**
```python
import numpy as np
rng = np.random.default_rng(42)
```

**NumGo:**
```go
import "github.com/iSundram/NumGo/random"
rng := random.New(42)
```

### Distributions

| NumPy | NumGo |
|-------|-------|
| `rng.uniform(0, 1, size=(3, 4))` | `rng.Uniform(0, 1, 3, 4)` |
| `rng.normal(0, 1, size=(100,))` | `rng.Normal(0, 1, 100)` |
| `rng.standard_normal((10, 10))` | `rng.StandardNormal(10, 10)` |
| `rng.binomial(10, 0.5, 100)` | `rng.Binomial(10, 0.5, 100)` |
| `rng.poisson(5, 100)` | `rng.Poisson(5, 100)` |
| `rng.exponential(2, 100)` | `rng.Exponential(2, 100)` |
| `rng.gamma(2, 2, 100)` | `rng.Gamma(2, 2, 100)` |
| `rng.beta(2, 5, 100)` | `rng.Beta(2, 5, 100)` |
| `rng.random((3, 4))` | `rng.Rand(3, 4)` |
| `rng.integers(0, 10, 20)` | `rng.Randint(0, 10, 20)` |
| `rng.choice(arr, 10)` | `rng.Choice(arr, 10)` |
| `rng.permutation(10)` | `rng.Permutation(10)` |
| `rng.shuffle(arr)` | `rng.Shuffle(arr)` |

## Key Differences

### 1. Method Calls
NumPy uses methods on arrays and module functions. NumGo uses methods on arrays for operations.

**NumPy:**
```python
result = np.exp(a)  # or a.exp() for some ops
```

**NumGo:**
```go
result := a.Exp()
```

### 2. Indexing
NumPy uses bracket notation. NumGo uses method calls.

**NumPy:**
```python
val = a[i, j]
a[i, j] = 5
```

**NumGo:**
```go
val := a.GetFloat64(i, j)
a.SetFloat64(5, i, j)
```

### 3. Broadcasting
Both support broadcasting automatically in arithmetic operations.

**NumPy:**
```python
c = a + b  # broadcasts automatically
```

**NumGo:**
```go
c := a.Add(b)  // broadcasts automatically
```

### 4. Type Safety
NumGo is strongly typed, so you need to specify types explicitly.

**NumPy:**
```python
a = np.array([1, 2, 3])  # type inferred
```

**NumGo:**
```go
a := tensor.FromSliceFloat64([]float64{1, 2, 3}, 3)
// or
b := tensor.FromSliceInt64([]int64{1, 2, 3}, 3)
```

### 5. Immutability
Most NumGo operations return new arrays rather than modifying in-place (except `Shuffle`).

**NumPy:**
```python
a += 1  # modifies a in-place
```

**NumGo:**
```go
a = a.AddScalar(1)  // returns new array
```

## Example Code Comparison

### NumPy
```python
import numpy as np

# Create array
a = np.array([[1, 2, 3], [4, 5, 6]])

# Operations
b = a + 10
c = np.sqrt(b)
mean = c.mean()

# Linear algebra
d = np.array([[1, 2], [3, 4]])
e = np.array([[5, 6], [7, 8]])
f = d @ e

# Random
rng = np.random.default_rng(42)
rand_arr = rng.normal(0, 1, (100,))
```

### NumGo
```go
import (
    "github.com/iSundram/NumGo/tensor"
    "github.com/iSundram/NumGo/linalg"
    "github.com/iSundram/NumGo/random"
)

// Create array
a := tensor.FromSliceFloat64([]float64{1, 2, 3, 4, 5, 6}, 2, 3)

// Operations
b := a.AddScalar(10)
c := b.Sqrt()
mean := c.Mean()

// Linear algebra
d := tensor.FromSliceFloat64([]float64{1, 2, 3, 4}, 2, 2)
e := tensor.FromSliceFloat64([]float64{5, 6, 7, 8}, 2, 2)
f := linalg.MatMul(d, e)

// Random
rng := random.New(42)
randArr := rng.Normal(0, 1, 100)
```

## Performance Notes

1. **Memory Layout**: NumGo uses C-contiguous memory layout by default, similar to NumPy.

2. **Broadcasting**: Both libraries implement broadcasting with similar semantics.

3. **Type Conversion**: NumGo requires explicit type conversions, while NumPy may do implicit conversions.

4. **Compilation**: NumGo is compiled to native code, potentially offering better performance for CPU-bound operations.

## Not Yet Implemented

The following NumPy features are planned but not yet available:

- Advanced indexing (boolean masks, fancy indexing)
- FFT operations
- Polynomial operations
- Advanced I/O (NPY, NPZ, HDF5)
- GPU acceleration
- BLAS/LAPACK integration (basic implementations exist)
- Full SciPy-equivalent functionality

See the project roadmap for implementation timeline.
