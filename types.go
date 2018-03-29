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

// Layer is the Data Structure to hold a layer of the neural network
type Layer struct {
	inputNodes     int
	outputNodes    int
	weights        *Matrix
	bias           *Matrix
	activationFunc ActivationFunction
	learningRate   float64
}

// Classifier is the Data Structure to hold an Classifier
type Classifier struct {
	inputNodes   int
	outputNodes  int
	hiddenLayers int
	layers       []*Layer
	classes      []float64
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
