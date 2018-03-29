package mlp

import (
	"fmt"
)

// NewLayer returns a pointer to a new layer
func NewLayer(inputNodes, outputNodes int) (*Layer, error) {
	if inputNodes < 1 || outputNodes < 1 {
		return &Layer{}, ErrNodeValue
	}

	bias, err := NewMatrix(outputNodes, 1)
	if err != nil {
		return nil, err
	}
	bias.Randomize(2, 1)

	weights, err := NewMatrix(outputNodes, inputNodes)
	if err != nil {
		return nil, err
	}
	weights.Randomize(2, 1)

	activationFunc := sigmoid
	learningRate := 0.001

	return &Layer{
		inputNodes,
		outputNodes,
		weights,
		bias,
		activationFunc,
		learningRate,
	}, nil
}

// NewLayerFromArrays creates a new layer from the provided weight and bias matrix
func NewLayerFromArrays(weightsArr, biasArr [][]float64) (*Layer, error) {
	weights, err := ConvertFromArray2DToMatrix(weightsArr)
	if err != nil {
		return nil, err
	}

	bias, err := ConvertFromArray2DToMatrix(biasArr)
	if err != nil {
		return nil, err
	}

	outputNodes := weights.rows
	inputNodes := weights.cols

	activationFunc := sigmoid
	learningRate := 0.001

	return &Layer{
		inputNodes,
		outputNodes,
		weights,
		bias,
		activationFunc,
		learningRate,
	}, nil
}

// FeedForward calculates the output of the layer for the given input
func (layer *Layer) FeedForward(input *Matrix) (*Matrix, error) {
	output, err := Multiply(layer.weights, input)
	if err != nil {
		return &Matrix{}, err
	}
	output, err = Add(output, layer.bias)
	if err != nil {
		return &Matrix{}, err
	}

	output.Map(layer.activationFunc.function)
	return output, nil
}

// BackPropogate used to backpropogate the error generated
func (layer *Layer) BackPropogate(outputs, inputs, layerError *Matrix) (*Matrix, error) {
	gradients := Map(outputs, layer.activationFunc.dfunction)
	var err error
	gradients, err = MapMultiply(gradients, layerError)
	if err != nil {
		return nil, err
	}
	gradients.Multiply(layer.learningRate)

	inputsT := inputs.Transpose()
	weightDeltas, err := Multiply(gradients, inputsT)
	if err != nil {
		return nil, err
	}

	layer.weights, err = Add(layer.weights, weightDeltas)
	if err != nil {
		return nil, err
	}

	layer.bias, err = Add(layer.bias, gradients)
	if err != nil {
		return nil, err
	}

	return gradients, nil
}

// WriteDataToFile writes the weight and bias matrix to a file
func (layer *Layer) WriteDataToFile(index int) error {
	err := WriteData(fmt.Sprintf("weights_matrix%d.csv", index), layer.weights.ConvertFromMatrixToArray2D())
	if err != nil {
		return err
	}
	err = WriteData(fmt.Sprintf("bias_matrix%d.csv", index), layer.bias.ConvertFromMatrixToArray2D())
	if err != nil {
		return err
	}

	return nil
}
