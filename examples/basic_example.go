// Example program demonstrating NumGo capabilities
package main

import (
	"fmt"
	
	"github.com/iSundram/NumGo/linalg"
	"github.com/iSundram/NumGo/random"
	"github.com/iSundram/NumGo/tensor"
)

func main() {
	fmt.Println("=== NumGo Example ===")
	fmt.Println()
	
	// 1. Create arrays
	fmt.Println("1. Creating arrays:")
	a := tensor.FromSliceFloat64([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
	fmt.Printf("Array a (2x3):\n%v\n", a)
	fmt.Printf("Shape: %v, Size: %d, DType: %s\n\n", a.Shape(), a.Size(), a.DType())
	
	// 2. Array operations
	fmt.Println("2. Array operations:")
	b := tensor.Ones([]int{2, 3}, tensor.Float64)
	c := a.Add(b)
	fmt.Printf("a + ones(2,3) = \n")
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			fmt.Printf("%.1f ", c.GetFloat64(i, j))
		}
		fmt.Println()
	}
	fmt.Println()
	
	// 3. Broadcasting
	fmt.Println("3. Broadcasting:")
	d := tensor.FromSliceFloat64([]float64{10, 20, 30}, 1, 3)
	e := a.Add(d)
	fmt.Printf("Broadcasting (2x3) + (1x3):\n")
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			fmt.Printf("%.1f ", e.GetFloat64(i, j))
		}
		fmt.Println()
	}
	fmt.Println()
	
	// 4. Math functions
	fmt.Println("4. Math functions:")
	f := tensor.FromSliceFloat64([]float64{1, 4, 9, 16}, 4)
	g := f.Sqrt()
	fmt.Printf("sqrt([1, 4, 9, 16]) = [")
	for i := 0; i < 4; i++ {
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Printf("%.1f", g.GetFloat64(i))
	}
	fmt.Println("]")
	fmt.Println()
	
	// 5. Reductions
	fmt.Println("5. Reductions:")
	h := tensor.FromSliceFloat64([]float64{1, 2, 3, 4, 5}, 5)
	fmt.Printf("Array: [1, 2, 3, 4, 5]\n")
	fmt.Printf("Sum: %.1f\n", h.Sum())
	fmt.Printf("Mean: %.1f\n", h.Mean())
	fmt.Printf("Std: %.3f\n", h.Std())
	fmt.Printf("Min: %.1f, Max: %.1f\n", h.Min(), h.Max())
	fmt.Println()
	
	// 6. Shape operations
	fmt.Println("6. Shape operations:")
	i := tensor.Range(0, 12)
	j := i.Reshape(3, 4)
	fmt.Printf("Reshape [0..11] to (3x4):\n")
	for row := 0; row < 3; row++ {
		for col := 0; col < 4; col++ {
			fmt.Printf("%2d ", int(j.GetFloat64(row, col)))
		}
		fmt.Println()
	}
	k := j.Transpose()
	fmt.Printf("\nTransposed (4x3):\n")
	for row := 0; row < 4; row++ {
		for col := 0; col < 3; col++ {
			fmt.Printf("%2d ", int(k.GetFloat64(row, col)))
		}
		fmt.Println()
	}
	fmt.Println()
	
	// 7. Linear algebra
	fmt.Println("7. Linear algebra:")
	m1 := tensor.FromSliceFloat64([]float64{1, 2, 3, 4}, 2, 2)
	m2 := tensor.FromSliceFloat64([]float64{5, 6, 7, 8}, 2, 2)
	m3 := linalg.MatMul(m1, m2)
	fmt.Printf("Matrix multiplication:\n")
	fmt.Println("[[1 2]   [[5 6]   [[19 22]")
	fmt.Println(" [3 4]] x [7 8]] = [43 50]]")
	fmt.Printf("Result: [[%.0f %.0f] [%.0f %.0f]]\n",
		m3.GetFloat64(0, 0), m3.GetFloat64(0, 1),
		m3.GetFloat64(1, 0), m3.GetFloat64(1, 1))
	fmt.Println()
	
	// 8. Identity and inverse
	fmt.Println("8. Matrix inverse:")
	eye := tensor.Eye(3, tensor.Float64)
	fmt.Printf("Identity matrix (3x3):\n")
	for row := 0; row < 3; row++ {
		for col := 0; col < 3; col++ {
			fmt.Printf("%.0f ", eye.GetFloat64(row, col))
		}
		fmt.Println()
	}
	fmt.Println()
	
	// 9. Random numbers
	fmt.Println("9. Random numbers:")
	rng := random.New(42)
	randArr := rng.Normal(0, 1, 10)
	fmt.Printf("10 random numbers from N(0,1): [")
	for i := 0; i < 10; i++ {
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Printf("%.2f", randArr.GetFloat64(i))
	}
	fmt.Println("]")
	fmt.Printf("Mean: %.3f, Std: %.3f\n\n", randArr.Mean(), randArr.Std())
	
	// 10. Random integers
	fmt.Println("10. Random integers:")
	randInt := rng.Randint(0, 10, 15)
	fmt.Printf("15 random integers in [0, 10): [")
	for i := 0; i < 15; i++ {
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Printf("%d", randInt.GetInt64(i))
	}
	fmt.Println("]")
	fmt.Println()
	
	fmt.Println("=== End of examples ===")
}
