package mlp

import "fmt"

// NewClassifier return a new pointer to the Classifier Class
func NewClassifier(inputNodes, outputNodes int, hiddenNodes ...int) (*Classifier, error) {
	if inputNodes < 0 || outputNodes < 0 || len(hiddenNodes) < 1 {
		return &Classifier{}, ErrNodeValue
	}

	var layers []*Layer
	hiddenLayers := len(hiddenNodes)

	hiddenNodes = append(hiddenNodes, outputNodes)

	inNodes := inputNodes
	outNodes := hiddenNodes[0]
	layer, err := NewLayer(inNodes, outNodes)
	if err != nil {
		return &Classifier{}, err
	}
	layers = append(layers, layer)

	for i := 1; i < len(hiddenNodes); i++ {
		inNodes = hiddenNodes[i-1]
		outNodes = hiddenNodes[i]
		layer, err = NewLayer(inNodes, outNodes)
		if err != nil {
			return &Classifier{}, err
		}
		layers = append(layers, layer)
	}

	var classes []float64

	return &Classifier{
		inputNodes,
		outputNodes,
		hiddenLayers,
		layers,
		classes,
	}, nil
}

// NewClassifierFromFiles return a new pointer to the Classifier Class from CSV files
func NewClassifierFromFiles(hiddenLayers int) (*Classifier, error) {
	var layers []*Layer

	for i := 0; i <= hiddenLayers; i++ {
		weightsArr, err := ReadData(fmt.Sprintf("weights_matrix%d.csv", i+1))
		if err != nil {
			return nil, err
		}

		biasArr, err := ReadData(fmt.Sprintf("bias_matrix%d.csv", i+1))
		if err != nil {
			return nil, err
		}

		layer, err := NewLayerFromArrays(weightsArr, biasArr)
		if err != nil {
			return nil, err
		}

		layers = append(layers, layer)
	}

	inputNodes := layers[0].inputNodes
	outputNodes := layers[len(layers)-1].outputNodes
	var classes []float64

	return &Classifier{
		inputNodes,
		outputNodes,
		hiddenLayers,
		layers,
		classes,
	}, nil
}

// Predict return a slice of the predicted output values of a trained neural network
func (mlp *Classifier) Predict(inputArr []float64) (int, error) {
	inputs, err := ConvertFromArrayToMatrix1D(inputArr)
	if err != nil {
		return 0, err
	}

	output := inputs
	for _, layer := range mlp.layers {
		output, err = layer.FeedForward(output)
		if err != nil {
			return 0, err
		}
	}

	if mlp.outputNodes > 1 {
		return output.FindGreatestIndex(), nil
	}

	outputArr := GreatestIntegerFunction(output.ConvertFromMatrixToArray1D())
	return outputArr[0], nil
}

// Train is used to train a neural network
func (mlp *Classifier) Train(data, targetArr [][]float64, epochs int) error {
	mlp.classes = ReturnTargetClasses(targetArr)
	transformedTarget := TransformTargets(targetArr, mlp.classes, mlp.outputNodes)
	for iter := 0; iter < epochs; iter++ {
		for range data {
			index, inputArr := RandomDataSet(data)
			inputs, err := ConvertFromArrayToMatrix1D(inputArr)
			if err != nil {
				return err
			}

			var outputs []*Matrix

			output := inputs
			outputs = append(outputs, output)

			for _, layer := range mlp.layers {
				output, err = layer.FeedForward(output)
				if err != nil {
					return err
				}

				outputs = append(outputs, output)
			}

			target, err := ConvertFromArrayToMatrix1D(transformedTarget[index])
			if err != nil {
				return err
			}

			layerError, err := Subtract(target, output)
			if err != nil {
				return err
			}

			var layerErrors []*Matrix
			layerErrors = append(layerErrors, layerError)
			for j := len(mlp.layers) - 1; j > 0; j-- {
				layerError, err = mlp.layers[j].BackPropogate(outputs[j+1], outputs[j], layerError)
				if err != nil {
					return err
				}

				layerErrors = append(layerErrors, layerError)

				weightsT := mlp.layers[j].weights.Transpose()

				layerError, err = Multiply(weightsT, layerError)
				if err != nil {
					return err
				}
			}
		}
	}

	// err := WriteData("weights_input_hidden.csv", mlp.weightsInputHidden.ConvertFromMatrixToArray2D())
	// if err != nil {
	// 	return err
	// }

	// err = WriteData("weights_hidden_output.csv", mlp.weightsHiddenOutput.ConvertFromMatrixToArray2D())
	// if err != nil {
	// 	return err
	// }

	// err = WriteData("bias_hidden.csv", mlp.biasHidden.ConvertFromMatrixToArray2D())
	// if err != nil {
	// 	return err
	// }

	// err = WriteData("bias_output.csv", mlp.biasOutput.ConvertFromMatrixToArray2D())
	// if err != nil {
	// 	return err
	// }

	for index, layer := range mlp.layers {
		err := layer.WriteDataToFile(index + 1)
		if err != nil {
			return err
		}
	}

	return nil
}

// Score return the various parameters of a Nerual Network used for checking efficiency and accuracy
func (mlp *Classifier) Score(data [][]float64, target [][]float64) (float64, error) {
	confusionMatrix := make([][]int, 2)
	confusionMatrix[0] = make([]int, len(mlp.classes))
	confusionMatrix[1] = make([]int, len(mlp.classes))
	var score float64
	for i, row := range data {
		prediction, err := mlp.Predict(row)
		if err != nil {
			return score, err
		}
		if prediction == int(target[i][0]) {
			pos := FindInArray(mlp.classes, float64(prediction))
			confusionMatrix[1][pos] = confusionMatrix[1][pos] + 1
		} else {
			pos := FindInArray(mlp.classes, float64(prediction))
			confusionMatrix[0][pos] = confusionMatrix[0][pos] + 1
		}
	}
	var total, accurate = 0.0, 0.0
	for index, row := range confusionMatrix {
		for _, element := range row {
			if index == 1 {
				accurate = accurate + float64(element)
			}
			total = total + float64(element)
		}
	}
	score = accurate / total
	return score, nil
}
