package mlp

// RangeDetermine is used to determine the range of values taken by the neural network
var RangeDetermine *Range

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

	learningRate := 0.01
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

	learningRate := 0.01
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
func (mlp *Classifier) Predict(inputArr []float64) (int, error) {
	inputs, err := ConvertFromArrayToMatrix1D(inputArr)
	if err != nil {
		return 0, err
	}
	hidden, err := Multiply(mlp.weightsInputHidden, inputs)
	if err != nil {
		return 0, err
	}
	hidden, err = Add(hidden, mlp.biasHidden)
	if err != nil {
		return 0, err
	}

	hidden.Map(mlp.activationFunc.function)

	output, err := Multiply(mlp.weightsHiddenOutput, hidden)
	if err != nil {
		return 0, err
	}
	output, err = Add(output, mlp.biasOutput)
	if err != nil {
		return 0, err
	}

	output.Map(mlp.activationFunc.function)
	if mlp.outputNodes > 1 {
		return output.FindGreatestIndex(), nil
	}

	outputArr := GreatestIntegerFunction(output.ConvertFromMatrixToArray1D())
	return outputArr[0], nil
}

// Train is used to train a neural network
func (mlp *Classifier) Train(data, targetArr [][]float64, epochs int) error {
	mlp.classes = ReturnTargetClasses(targetArr)
	RangeDetermine.UpdateRangeArray1D(mlp.classes)
	transformedTarget := TransformTargets(targetArr, mlp.classes, mlp.outputNodes)
	RangeDetermine.UpdateRangeArray2D(transformedTarget)
	for iter := 0; iter < epochs; iter++ {
		for range data {
			index, inputArr := RandomDataSet(data)
			inputs, err := ConvertFromArrayToMatrix1D(inputArr)
			if err != nil {
				return err
			}
			RangeDetermine.UpdateRangeMatrix(inputs)
			hidden, err := Multiply(mlp.weightsInputHidden, inputs)
			if err != nil {
				return err
			}
			RangeDetermine.UpdateRangeMatrix(hidden)
			hidden, err = Add(hidden, mlp.biasHidden)
			if err != nil {
				return err
			}
			RangeDetermine.UpdateRangeMatrix(hidden)
			hidden.Map(mlp.activationFunc.function)
			RangeDetermine.UpdateRangeMatrix(hidden)

			output, err := Multiply(mlp.weightsHiddenOutput, hidden)
			if err != nil {
				return err
			}
			RangeDetermine.UpdateRangeMatrix(output)
			output, err = Add(output, mlp.biasOutput)
			if err != nil {
				return err
			}
			RangeDetermine.UpdateRangeMatrix(output)

			output.Map(mlp.activationFunc.function)
			RangeDetermine.UpdateRangeMatrix(output)
			target, err := ConvertFromArrayToMatrix1D(transformedTarget[index])
			if err != nil {
				return err
			}

			outputError, err := Subtract(target, output)
			if err != nil {
				return err
			}
			RangeDetermine.UpdateRangeMatrix(outputError)

			gradients := Map(output, mlp.activationFunc.dfunction)
			RangeDetermine.UpdateRangeMatrix(gradients)
			gradients, err = MapMultiply(gradients, outputError)
			RangeDetermine.UpdateRangeMatrix(gradients)
			if err != nil {
				return err
			}
			gradients.Multiply(mlp.learningRate)
			RangeDetermine.UpdateRangeMatrix(gradients)

			hiddenT := hidden.Transpose()
			weightHiddenOutputDeltas, err := Multiply(gradients, hiddenT)
			if err != nil {
				return err
			}
			RangeDetermine.UpdateRangeMatrix(weightHiddenOutputDeltas)

			mlp.weightsHiddenOutput, err = Add(mlp.weightsHiddenOutput, weightHiddenOutputDeltas)
			if err != nil {
				return err
			}
			RangeDetermine.UpdateRangeMatrix(weightHiddenOutputDeltas)

			mlp.biasOutput, err = Add(mlp.biasOutput, gradients)
			if err != nil {
				return err
			}

			weightsHiddenOutputT := mlp.weightsHiddenOutput.Transpose()
			hiddenErrors, err := Multiply(weightsHiddenOutputT, gradients)
			if err != nil {
				return err
			}
			RangeDetermine.UpdateRangeMatrix(hiddenErrors)

			hiddenGradient := Map(hidden, mlp.activationFunc.dfunction)
			RangeDetermine.UpdateRangeMatrix(hiddenGradient)
			hiddenGradient, err = MapMultiply(hiddenGradient, hiddenErrors)
			RangeDetermine.UpdateRangeMatrix(hiddenGradient)
			if err != nil {
				return err
			}
			hiddenGradient.Multiply(mlp.learningRate)
			RangeDetermine.UpdateRangeMatrix(hiddenGradient)

			inputsT := inputs.Transpose()
			weightsInputHiddenDeltas, err := Multiply(hiddenGradient, inputsT)
			if err != nil {
				return err
			}
			RangeDetermine.UpdateRangeMatrix(weightsInputHiddenDeltas)
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
	RangeDetermine.UpdateRangeArray1D([]float64{accurate, total})
	return score, nil
}
