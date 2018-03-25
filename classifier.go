package mlp

func NewMLPClassifier(inputNodes, hiddenNodes, outputNodes int) (MLPClassifier, error) {
	if inputNodes < 0 || outputNodes < 0 || hiddenNodes < 0 {
		return MLPClassifier{}, ErrNodeValue
	}

	biasHidden, err := NewMatrix(hiddenNodes, 1)
	if err != nil {
		return MLPClassifier{}, err
	}
	biasOutput, err := NewMatrix(outputNodes, 1)
	if err != nil {
		return MLPClassifier{}, err
	}
	weightsInputHidden, err := NewMatrix(hiddenNodes, inputNodes)
	if err != nil {
		return MLPClassifier{}, err
	}
	weightsInputHidden.Randomize(2, 1)

	weightsHiddenOutput, err := NewMatrix(outputNodes, hiddenNodes)
	if err != nil {
		return MLPClassifier{}, err
	}
	weightsHiddenOutput.Randomize(2, 1)

	learningRate := 0.005
	activationFunc := sigmoid

	return MLPClassifier{
			inputNodes,
			hiddenNodes,
			outputNodes,
			biasHidden,
			biasOutput,
			weightsInputHidden,
			weightsHiddenOutput,
			learningRate,
			activationFunc,
		}, nil
}

func NewMLPClassifierFromFiles(weightsInputHiddenFile, weightsHiddenOutputFile, biasHiddenFile, biasOutputFile string, stringHandler func(string) string) (MLPClassifier, error) {
	weightsInputHiddenArr, err := ReadData(weightsInputHiddenFile, stringHandler)
	if err != nil {
		return MLPClassifier{}, err
	}
	weightsInputHidden, err := ConvertFromArray2DToMatrix(weightsInputHiddenArr)
	if err != nil {
		return MLPClassifier{}, err
	}

	weightsHiddenOutputArr, err := ReadData(weightsHiddenOutputFile, stringHandler)
	if err != nil {
		return MLPClassifier{}, err
	}
	weightsHiddenOutput, err := ConvertFromArray2DToMatrix(weightsHiddenOutputArr)
	if err != nil {
		return MLPClassifier{}, err
	}

	biasHiddenArr, err := ReadData(biasHiddenFile, stringHandler)
	if err != nil {
		return MLPClassifier{}, err
	}
	biasHidden, err := ConvertFromArray2DToMatrix(biasHiddenArr)
	if err != nil {
		return MLPClassifier{}, err
	}
	
	biasOutputArr, err := ReadData(biasOutputFile, stringHandler)
	if err != nil {
		return MLPClassifier{}, err
	}
	biasOutput, err := ConvertFromArray2DToMatrix(biasOutputArr)
	if err != nil {
		return MLPClassifier{}, err
	}

	learningRate := 0.005
	activationFunc := sigmoid

	inputNodes := len(weightsInputHiddenArr[0])
	hiddenNodes := len(weightsInputHiddenArr)
	outputNodes := len(weightsHiddenOutputArr)

	return MLPClassifier{
			inputNodes,
			hiddenNodes,
			outputNodes,
			biasHidden,
			biasOutput,
			weightsInputHidden,
			weightsHiddenOutput,
			learningRate,
			activationFunc,
		}, nil
}

func (mlp MLPClassifier) Predict(input_arr []float64) ([]float64, error) {
	inputs, err := ConvertFromArrayToMatrix1D(input_arr)
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

	return output.ConvertFromMatrixToArray1D(), nil
}

func (mlp MLPClassifier) Train(data, target_arr [][]float64, epochs int) error {
	for iter := 0; iter < epochs; iter++ {
		for _ = range(data) {
			index, input_arr := RandomDataSet(data)
			inputs, err := ConvertFromArrayToMatrix1D(input_arr)
			if err != nil {
				return err
			}
			hidden, err := Multiply(mlp.weightsInputHidden, inputs)
			if err != nil {
				panic(err)
				return err
			}
			hidden, err = Add(hidden, mlp.biasHidden)
			if err != nil {
				return err
			}

			hidden.Map(mlp.activationFunc.function)

			output, err := Multiply(mlp.weightsHiddenOutput, hidden)
			if err != nil {
				panic(err)
				return err
			}
			output, err = Add(output, mlp.biasOutput)
			if err != nil {
				return err
			}

			output.Map(mlp.activationFunc.function)

			target, err := ConvertFromArrayToMatrix1D(target_arr[index])
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
				panic(err)
				return err
			}
			gradients.Multiply(mlp.learningRate)

			hidden_t := hidden.Transpose()
			weightHiddenOutputDeltas, err := Multiply(gradients, hidden_t)
			if err != nil {
				panic(err)
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

			weightsHiddenOutput_t := mlp.weightsHiddenOutput.Transpose()
			hiddenErrors, err := Multiply(weightsHiddenOutput_t, gradients)
			if err != nil {
				panic(err)
				return err
			}

			hiddenGradient := Map(hidden, mlp.activationFunc.dfunction)
			hiddenGradient, err = MapMultiply(hiddenGradient, hiddenErrors)
			if err != nil {
				panic(err)
				return err
			}
			hiddenGradient.Multiply(mlp.learningRate)

			inputs_t := inputs.Transpose()
			weightsInputHiddenDeltas, err := Multiply(hiddenGradient, inputs_t)
			if err != nil {
				panic(err)
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

func (mlp MLPClassifier) Score(data [][]float64, target[][] float64) ([]float64, error) {
	var TP, TN, FP, FN = 0, 0, 0, 0
	var score []float64
	for i, row := range data {
		prediction, err := mlp.Predict(row)
		if err != nil {
			panic(err)
			return score, err
		}
		if prediction[0] > 0.5 && target[i][0] == 1 {
			TP = TP + 1
		} else if prediction[0] <= 0.5 && target[i][0] == 0 {
			TN = TN + 1
		} else if prediction[0] > 0.5 && target[i][0] == 0 {
			FP = FP + 1
		} else if prediction[0] <= 0.5 && target[i][0] == 1 {
			FN = FN + 1
		}
	}
	sensitivity := ( float64(TP) / float64(TP+FN) ) * 100;
	specificity := ( float64(TN) / float64(TN+FP) ) * 100;
	accuracy := ( float64(TP+TN) / float64(TP+TN+FP+FN) ) * 100;
	efficiency := float64( sensitivity + specificity + accuracy ) / 3; 
	score = []float64{sensitivity, specificity, accuracy, efficiency}
	return score, nil
}