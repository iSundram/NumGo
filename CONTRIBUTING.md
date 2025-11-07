# Contributing to NumGo

Thank you for your interest in contributing to NumGo! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct:
- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other community members

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with:
- A clear, descriptive title
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Go version and OS information
- Code samples demonstrating the issue

### Suggesting Features

Feature suggestions are welcome! Please:
- Check existing issues first to avoid duplicates
- Provide a clear use case
- Explain why this feature would be useful
- Consider backward compatibility

### Pull Requests

1. **Fork the repository** and create a branch from `main`
2. **Make your changes** following our coding standards
3. **Add tests** for any new functionality
4. **Update documentation** as needed
5. **Run tests** to ensure nothing breaks: `go test ./...`
6. **Submit a pull request** with a clear description

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/NumGo.git
cd NumGo

# Install dependencies (if any)
go mod download

# Run tests
go test ./...

# Run a specific test
go test ./tensor -v -run TestAdd

# Build examples
cd examples
go build basic_example.go
./basic_example
```

## Coding Standards

### Go Style
- Follow the official [Go Code Review Comments](https://github.com/golang/go/wiki/CodeReviewComments)
- Use `gofmt` to format your code
- Use meaningful variable and function names
- Write clear comments for exported functions

### Testing
- Write unit tests for all new functionality
- Aim for high test coverage (>90%)
- Use table-driven tests where appropriate
- Test edge cases and error conditions

### Example Test
```go
func TestNewFeature(t *testing.T) {
    // Arrange
    input := tensor.FromSliceFloat64([]float64{1, 2, 3}, 3)
    
    // Act
    result := input.NewOperation()
    
    // Assert
    expected := 6.0
    if result != expected {
        t.Errorf("expected %f, got %f", expected, result)
    }
}
```

### Documentation
- Document all exported functions, types, and constants
- Include usage examples in documentation
- Update README.md if adding major features
- Keep docs/api.md and docs/migration.md up to date

### Example Documentation
```go
// NewFeature performs operation X on the array
// 
// Parameters:
//   - param1: description of param1
//   - param2: description of param2
//
// Returns:
//   - *NDArray: description of return value
//
// Example:
//   arr := tensor.Zeros([]int{3, 3}, tensor.Float64)
//   result := arr.NewFeature(param1, param2)
func (a *NDArray) NewFeature(param1, param2 int) *NDArray {
    // implementation
}
```

## Project Structure

```
NumGo/
├── tensor/        # Core NDArray implementation
├── linalg/        # Linear algebra operations
├── random/        # Random number generation
├── fft/           # FFT operations (planned)
├── io/            # I/O operations (planned)
├── stats/         # Statistical functions (planned)
├── examples/      # Example programs
├── docs/          # Documentation
└── bench/         # Benchmarks (planned)
```

## Adding a New Feature

1. **Discuss first**: For major features, open an issue first to discuss the design
2. **Start small**: Break large features into smaller, reviewable PRs
3. **Write tests**: Add comprehensive tests alongside your code
4. **Document**: Update relevant documentation
5. **Benchmark**: For performance-critical code, add benchmarks

## Testing Guidelines

### Unit Tests
```bash
# Run all tests
go test ./...

# Run with coverage
go test ./... -cover

# Run with verbose output
go test ./... -v

# Run specific package
go test ./tensor -v
```

### Writing Good Tests
- Test the public API, not internal implementation
- Use descriptive test names: `TestFeatureName_Condition_ExpectedBehavior`
- Test edge cases: empty arrays, single elements, large arrays
- Test error conditions
- Use helper functions to reduce duplication

### Benchmarks
```go
func BenchmarkMatMul(b *testing.B) {
    a := tensor.Ones([]int{100, 100}, tensor.Float64)
    m := tensor.Ones([]int{100, 100}, tensor.Float64)
    
    b.ResetTimer()
    for i := 0; i < b.N; i++ {
        linalg.MatMul(a, m)
    }
}
```

## Performance Considerations

- Avoid unnecessary allocations
- Use in-place operations when safe
- Consider memory layout and cache efficiency
- Profile before optimizing: `go test -bench=. -cpuprofile=cpu.prof`
- Document performance characteristics of algorithms

## Commit Messages

Write clear, descriptive commit messages:

```
Add matrix inverse function to linalg package

- Implement Gauss-Jordan elimination for matrix inversion
- Add comprehensive tests for 2x2, 3x3, and singular matrices
- Update API documentation

Fixes #123
```

Format:
- Use present tense ("Add feature" not "Added feature")
- First line is a summary (50 chars or less)
- Blank line, then detailed description if needed
- Reference issues: "Fixes #123" or "Relates to #456"

## Review Process

1. All PRs require review from at least one maintainer
2. CI tests must pass
3. Code coverage should not decrease
4. Documentation must be updated
5. Follow-up on feedback promptly

## Getting Help

- Open an issue for questions
- Check existing issues and documentation
- Reach out to maintainers for guidance

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Recognition

Contributors will be recognized in the project README and release notes.

Thank you for contributing to NumGo!
