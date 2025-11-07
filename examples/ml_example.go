// Advanced example demonstrating NumGo for machine learning tasks
package main

import (
	"fmt"
	"math"
	
	"github.com/iSundram/NumGo/linalg"
	"github.com/iSundram/NumGo/random"
	"github.com/iSundram/NumGo/tensor"
)

func main() {
	fmt.Println("=== NumGo Advanced Example: Simple Linear Regression ===")
	fmt.Println()
	
	// Set random seed for reproducibility
	rng := random.New(42)
	
	// 1. Generate synthetic data: y = 2x + 3 + noise
	fmt.Println("1. Generating synthetic data...")
	n := 100
	X := rng.Uniform(0, 10, n)
	noise := rng.Normal(0, 0.5, n)
	
	// y = 2X + 3 + noise
	y := X.MulScalar(2).AddScalar(3).Add(noise)
	
	fmt.Printf("Generated %d data points\n", n)
	fmt.Printf("X sample: [%.2f, %.2f, %.2f, ...]\n", 
		X.GetFloat64(0), X.GetFloat64(1), X.GetFloat64(2))
	fmt.Printf("y sample: [%.2f, %.2f, %.2f, ...]\n\n",
		y.GetFloat64(0), y.GetFloat64(1), y.GetFloat64(2))
	
	// 2. Prepare data for linear regression: add bias term
	fmt.Println("2. Preparing data matrix...")
	ones := tensor.Ones([]int{n}, tensor.Float64)
	
	// Stack [ones, X] to create design matrix
	XWithBias := tensor.Stack([]*tensor.NDArray{ones, X}, 1)
	fmt.Printf("Design matrix shape: %v\n\n", XWithBias.Shape())
	
	// 3. Solve normal equations: beta = (X^T X)^-1 X^T y
	fmt.Println("3. Solving normal equations...")
	
	// X^T X
	XT := XWithBias.Transpose()
	XTX := linalg.MatMul(XT, XWithBias)
	
	// X^T y (need to reshape y for matrix multiplication)
	yReshaped := y.Reshape(n, 1)
	XTy := linalg.MatMul(XT, yReshaped)
	
	// (X^T X)^-1
	XTXInv := linalg.Inv(XTX)
	
	// beta = (X^T X)^-1 X^T y
	beta := linalg.MatMul(XTXInv, XTy)
	
	intercept := beta.GetFloat64(0, 0)
	slope := beta.GetFloat64(1, 0)
	
	fmt.Printf("Fitted parameters:\n")
	fmt.Printf("  Intercept: %.4f (true: 3.0)\n", intercept)
	fmt.Printf("  Slope: %.4f (true: 2.0)\n\n", slope)
	
	// 4. Make predictions
	fmt.Println("4. Making predictions...")
	yPred := linalg.MatMul(XWithBias, beta)
	
	// Calculate residuals
	residuals := yReshaped.Sub(yPred)
	
	// 5. Evaluate model performance
	fmt.Println("5. Model evaluation...")
	
	// Mean Squared Error
	mse := residuals.Pow(2).Mean()
	rmse := math.Sqrt(mse)
	
	// R-squared
	yMean := y.Mean()
	ssTot := y.AddScalar(-yMean).Pow(2).Sum()
	ssRes := residuals.Pow(2).Sum()
	r2 := 1 - (ssRes / ssTot)
	
	fmt.Printf("  MSE: %.4f\n", mse)
	fmt.Printf("  RMSE: %.4f\n", rmse)
	fmt.Printf("  R²: %.4f\n\n", r2)
	
	// 6. Display some predictions vs actual
	fmt.Println("6. Sample predictions:")
	fmt.Println("   X       y_true   y_pred   error")
	fmt.Println("  ----    ------   ------   -----")
	for i := 0; i < 10; i++ {
		xVal := X.GetFloat64(i)
		yTrue := y.GetFloat64(i)
		yPredVal := yPred.GetFloat64(i, 0)
		error := math.Abs(yTrue - yPredVal)
		fmt.Printf("  %.2f    %.2f    %.2f    %.3f\n", 
			xVal, yTrue, yPredVal, error)
	}
	fmt.Println()
	
	// 7. Demonstrate matrix operations
	fmt.Println("7. Additional matrix operations:")
	
	// Covariance matrix
	covMatrix := XTX.MulScalar(1.0 / float64(n-1))
	fmt.Printf("Covariance matrix:\n")
	for i := 0; i < 2; i++ {
		fmt.Print("  [")
		for j := 0; j < 2; j++ {
			fmt.Printf(" %8.4f", covMatrix.GetFloat64(i, j))
		}
		fmt.Println(" ]")
	}
	fmt.Println()
	
	// 8. Statistical analysis
	fmt.Println("8. Statistical analysis of data:")
	fmt.Printf("X statistics:\n")
	fmt.Printf("  Mean: %.4f\n", X.Mean())
	fmt.Printf("  Std Dev: %.4f\n", X.Std())
	fmt.Printf("  Min: %.4f\n", X.Min())
	fmt.Printf("  Max: %.4f\n\n", X.Max())
	
	fmt.Printf("y statistics:\n")
	fmt.Printf("  Mean: %.4f\n", y.Mean())
	fmt.Printf("  Std Dev: %.4f\n", y.Std())
	fmt.Printf("  Min: %.4f\n", y.Min())
	fmt.Printf("  Max: %.4f\n\n", y.Max())
	
	// 9. Demonstrate comparison operations
	fmt.Println("9. Finding outliers (residuals > 1.0):")
	absResiduals := residuals.Abs().Flatten()
	outliers := absResiduals.GtScalar(1.0)
	outlierCount := outliers.Sum()
	fmt.Printf("Number of outliers: %.0f out of %d (%.1f%%)\n\n",
		outlierCount, n, (outlierCount/float64(n))*100)
	
	// 10. Cross-validation split demonstration
	fmt.Println("10. Train/test split demonstration:")
	trainSize := int(0.8 * float64(n))
	testSize := n - trainSize
	
	// Shuffle indices
	indices := rng.Permutation(n)
	
	fmt.Printf("Training set size: %d\n", trainSize)
	fmt.Printf("Test set size: %d\n", testSize)
	fmt.Printf("Sample shuffled indices: [%d, %d, %d, ...]\n\n",
		indices.GetInt64(0), indices.GetInt64(1), indices.GetInt64(2))
	
	fmt.Println("=== End of advanced example ===")
}
