# NumGo (go-numpy)

[![Go Version](https://img.shields.io/badge/Go-%3E%3D%201.18-blue.svg)](https://golang.org/dl/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A production-ready, idiomatic Go implementation of the NumPy ecosystem. NumGo provides a high-performance N-dimensional array/tensor library with NumPy-like APIs for numerical computing in Go.

## Features

- **N-dimensional arrays**: Full support for multi-dimensional arrays with arbitrary rank and shape
- **Multiple data types**: float64, float32, int64, int32, uint64, uint32, bool, and more
- **Broadcasting**: NumPy-compatible broadcasting for efficient array operations
- **Indexing & Slicing**: Advanced indexing including boolean masking and fancy indexing
- **Linear Algebra**: Matrix operations, decompositions (SVD, QR, eigenvalues)
- **Random**: Random number generation with various distributions
- **I/O**: Support for NPY, NPZ, CSV, and more
- **Performance**: Optimized CPU operations with optional BLAS/LAPACK integration

## Installation

```bash
go get github.com/iSundram/NumGo
```

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/iSundram/NumGo/tensor"
)

func main() {
    // Create arrays
    a := tensor.FromSlice([]float64{1, 2, 3, 4, 5, 6}, 2, 3) // 2x3 array
    b := tensor.Zeros([]int{3, 3}, tensor.Float64)            // 3x3 zeros
    
    // Elementwise operations
    c := a.Add(b) // Broadcasting supported
    
    // Indexing and slicing
    row := a.At(0)      // Get first row
    val := a.Get(0, 1)  // Get element at [0,1]
    
    // Reductions
    sum := a.Sum()
    mean := a.Mean()
    
    fmt.Println("Array a:", a)
    fmt.Println("Sum:", sum)
    fmt.Println("Mean:", mean)
}
```

## Module Architecture

NumGo is organized into modular packages:

- **tensor/**: Core NDArray implementation, dtypes, indexing, broadcasting
- **ufunc/**: Universal functions and elementwise operations
- **linalg/**: Linear algebra operations, BLAS/LAPACK wrappers
- **fft/**: Fast Fourier Transform implementations
- **random/**: Random number generation and distributions
- **io/**: I/O operations (NPY, NPZ, CSV, HDF5)
- **stats/**: Statistical functions
- **special/**: Special mathematical functions
- **utils/**: Utilities for memory management and threading

## NumPy Compatibility

NumGo aims for functional parity with NumPy. Here's a comparison:

| NumPy | NumGo |
|-------|-------|
| `np.array([1,2,3])` | `tensor.FromSlice([]float64{1,2,3})` |
| `np.zeros((3,3))` | `tensor.Zeros([]int{3,3}, tensor.Float64)` |
| `a + b` | `a.Add(b)` |
| `a.reshape(2,3)` | `a.Reshape(2, 3)` |
| `a.T` | `a.Transpose()` |
| `np.dot(a,b)` | `linalg.Dot(a, b)` |

See [docs/migration.md](docs/migration.md) for a complete migration guide.

## Performance

NumGo is designed for high performance:

- Optimized memory layouts (C-contiguous and Fortran-contiguous)
- Optional BLAS/LAPACK integration for linear algebra
- Vectorized operations where possible
- Memory-efficient views (no hidden copies)
- Parallelization for large operations

## Documentation

- [API Documentation](docs/api.md)
- [NumPy Migration Guide](docs/migration.md)
- [Performance Tuning](docs/performance.md)
- [Examples](examples/)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Development Status

NumGo is under active development. Current status:

- ✅ Core tensor implementation
- ✅ Basic elementwise operations
- 🚧 Linear algebra (in progress)
- 🚧 Random number generation (in progress)
- 📋 FFT (planned)
- 📋 Advanced I/O (planned)

## License

NumGo is released under the MIT License. See [LICENSE](LICENSE) for details.

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for planned features and milestones.

## Credits

Inspired by NumPy, the fundamental package for scientific computing with Python.
