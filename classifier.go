package mlp

// NewClassifier return a new pointer to the Classifier Class
func NewClassifier(inputNodes, hiddenNodes, outputNodes int) (*Classifier, error) {
	if inputNodes < 0 || outputNodes < 0 || hiddenNodes < 0 {
		return &Classifier{}, ErrNodeValue
	}

	biasHidden, err := NewMatrix(hiddenNodes, 1)
	if err != nil {
		return &Classifier{}, err
	}
	biasOutput, err := NewMatrix(outputNodes, 1)
	if err != nil {
		return &Classifier{}, err
	}
	weightsInputHidden, err := NewMatrix(hiddenNodes, inputNodes)
	if err != nil {
		return &Classifier{}, err
	}
	weightsInputHidden.Randomize(2, 1)

	weightsHiddenOutput, err := NewMatrix(outputNodes, hiddenNodes)
	if err != nil {
		return &Classifier{}, err
	}
	weightsHiddenOutput.Randomize(2, 1)

	learningRate := 0.005
	activationFunc := sigmoid
	var classes []float64

	return &Classifier{
		inputNodes,
		hiddenNodes,
		outputNodes,
		biasHidden,
		biasOutput,
		weightsInputHidden,
		weightsHiddenOutput,
		learningRate,
		activationFunc,
		classes,
	}, nil
}

// NewClassifierFromFiles return a new pointer to the Classifier Class from CSV files
func NewClassifierFromFiles(weightsInputHiddenFile, weightsHiddenOutputFile, biasHiddenFile, biasOutputFile string, stringHandler func(string) string) (*Classifier, error) {
	weightsInputHiddenArr, err := ReadData(weightsInputHiddenFile, stringHandler)
	if err != nil {
		return &Classifier{}, err
	}
	weightsInputHidden, err := ConvertFromArray2DToMatrix(weightsInputHiddenArr)
	if err != nil {
		return &Classifier{}, err
	}

	weightsHiddenOutputArr, err := ReadData(weightsHiddenOutputFile, stringHandler)
	if err != nil {
		return &Classifier{}, err
	}
	weightsHiddenOutput, err := ConvertFromArray2DToMatrix(weightsHiddenOutputArr)
	if err != nil {
		return &Classifier{}, err
	}

	biasHiddenArr, err := ReadData(biasHiddenFile, stringHandler)
	if err != nil {
		return &Classifier{}, err
	}
	biasHidden, err := ConvertFromArray2DToMatrix(biasHiddenArr)
	if err != nil {
		return &Classifier{}, err
	}

	biasOutputArr, err := ReadData(biasOutputFile, stringHandler)
	if err != nil {
		return &Classifier{}, err
	}
	biasOutput, err := ConvertFromArray2DToMatrix(biasOutputArr)
	if err != nil {
		return &Classifier{}, err
	}

	learningRate := 0.005
	activationFunc := sigmoid

	inputNodes := len(weightsInputHiddenArr[0])
	hiddenNodes := len(weightsInputHiddenArr)
	outputNodes := len(weightsHiddenOutputArr)
	var classes []float64

	return &Classifier{
		inputNodes,
		hiddenNodes,
		outputNodes,
		biasHidden,
		biasOutput,
		weightsInputHidden,
		weightsHiddenOutput,
		learningRate,
		activationFunc,
		classes,
	}, nil
}

// Predict return a slice of the predicted output values of a trained neural network
func (mlp *Classifier) Predict(inputArr []float64) ([]float64, error) {
	inputs, err := ConvertFromArrayToMatrix1D(inputArr)
	if err != nil {
		return make([]float64, 0), err
	}
	hidden, err := Multiply(mlp.weightsInputHidden, inputs)
	if err != nil {
		return make([]float64, 0), err
	}
	hidden, err = Add(hidden, mlp.biasHidden)
	if err != nil {
		return make([]float64, 0), err
	}

	hidden.Map(mlp.activationFunc.function)

	output, err := Multiply(mlp.weightsHiddenOutput, hidden)
	if err != nil {
		return make([]float64, 0), err
	}
	output, err = Add(output, mlp.biasOutput)
	if err != nil {
		return make([]float64, 0), err
	}

	output.Map(mlp.activationFunc.function)

	return GreatestIntegerFunction(output.ConvertFromMatrixToArray1D()), nil
}

// Train is used to train a neural network
func (mlp *Classifier) Train(data, targetArr [][]float64, epochs int) error {
	mlp.classes = ReturnTargetClasses(targetArr)
	for iter := 0; iter < epochs; iter++ {
		for range data {
			index, inputArr := RandomDataSet(data)
			inputs, err := ConvertFromArrayToMatrix1D(inputArr)
			if err != nil {
				return err
			}
			hidden, err := Multiply(mlp.weightsInputHidden, inputs)
			if err != nil {
				return err
			}
			hidden, err = Add(hidden, mlp.biasHidden)
			if err != nil {
				return err
			}

			hidden.Map(mlp.activationFunc.function)

			output, err := Multiply(mlp.weightsHiddenOutput, hidden)
			if err != nil {
				return err
			}
			output, err = Add(output, mlp.biasOutput)
			if err != nil {
				return err
			}

			output.Map(mlp.activationFunc.function)

			target, err := ConvertFromArrayToMatrix1D(targetArr[index])
			if err != nil {
				return err
			}

			outputError, err := Subtract(target, output)
			if err != nil {
				return err
			}

			gradients := Map(output, mlp.activationFunc.dfunction)
			gradients, err = Multiply(gradients, outputError)
			if err != nil {
				return err
			}
			gradients.Multiply(mlp.learningRate)

			hiddenT := hidden.Transpose()
			weightHiddenOutputDeltas, err := Multiply(gradients, hiddenT)
			if err != nil {
				return err
			}

			mlp.weightsHiddenOutput, err = Add(mlp.weightsHiddenOutput, weightHiddenOutputDeltas)
			if err != nil {
				return err
			}

			mlp.biasOutput, err = Add(mlp.biasOutput, gradients)
			if err != nil {
				return err
			}

			weightsHiddenOutputT := mlp.weightsHiddenOutput.Transpose()
			hiddenErrors, err := Multiply(weightsHiddenOutputT, gradients)
			if err != nil {

				return err
			}

			hiddenGradient := Map(hidden, mlp.activationFunc.dfunction)
			hiddenGradient, err = MapMultiply(hiddenGradient, hiddenErrors)
			if err != nil {

				return err
			}
			hiddenGradient.Multiply(mlp.learningRate)

			inputsT := inputs.Transpose()
			weightsInputHiddenDeltas, err := Multiply(hiddenGradient, inputsT)
			if err != nil {

				return err
			}
			mlp.weightsInputHidden, err = Add(mlp.weightsInputHidden, weightsInputHiddenDeltas)
			if err != nil {
				return err
			}
			mlp.biasHidden, err = Add(mlp.biasHidden, hiddenGradient)
			if err != nil {
				return err
			}
		}
	}

	err := WriteData("weights_input_hidden.csv", mlp.weightsInputHidden.ConvertFromMatrixToArray2D())
	if err != nil {
		return err
	}

	err = WriteData("weights_hidden_output.csv", mlp.weightsHiddenOutput.ConvertFromMatrixToArray2D())
	if err != nil {
		return err
	}

	err = WriteData("bias_hidden.csv", mlp.biasHidden.ConvertFromMatrixToArray2D())
	if err != nil {
		return err
	}

	err = WriteData("bias_output.csv", mlp.biasOutput.ConvertFromMatrixToArray2D())
	if err != nil {
		return err
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
		if prediction[0] == target[i][0] {
			pos := FindInArray(mlp.classes, prediction[0])
			confusionMatrix[1][pos] = confusionMatrix[1][pos] + 1
		} else {
			pos := FindInArray(mlp.classes, prediction[0])
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
