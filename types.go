package mlp

// Matrix is the Data Structure for the Matrix Operations
type Matrix struct {
	data [][]float64
	rows int
	cols int
}

// ActivationFunction is the DataStructure to hold the Activation Functions
type ActivationFunction struct {
	function  func(float64) float64
	dfunction func(float64) float64
}

// Classifier is the Data Structure to hold an Classifier
type Classifier struct {
	inputNodes          int
	hiddenNodes         int
	outputNodes         int
	biasHidden          *Matrix
	biasOutput          *Matrix
	weightsInputHidden  *Matrix
	weightsHiddenOutput *Matrix
	learningRate        float64
	activationFunc      ActivationFunction
}

// StandardScalar is the Data Structure to hold the Standard Scalar Object
type StandardScalar struct {
	mean []float64
	dev  []float64
}

// Normalizer is the Data Structure to hold the Normalizer Object
type Normalizer struct {
	max []float64
	min []float64
}
