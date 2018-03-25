package mlp

type Matrix struct {
	data [][]float64 
	rows int
	cols int
}

type ActivationFunction struct {
	function  func(float64) float64
	dfunction func(float64) float64
}

type MLPClassifier struct {
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

type StandardScalar struct {
	mean []float64
	dev  []float64
}

type Normalizer struct {
	max []float64
	min []float64
}