package main

import (
	"fmt"
	"strconv"

	mlp "github.com/reficul31/mlp_classifier"
)

var epochs = 10

func dummyHandler(input string) string {
	strFloat, err := strconv.ParseFloat(input, 64)
	if err != nil {
		panic(err)
	}
	return fmt.Sprintf("%0.2f", strFloat)
}

func main() {
	data, err := mlp.ReadData("training.csv", dummyHandler)
	if err != nil {
		panic(err)
	}

	inputs, targets := mlp.PartitionData(data, "start")
	scalar := mlp.NewStandardScalar(len(inputs[0]))
	scalar.Fit(inputs)
	scaled := scalar.Transform(inputs)

	brain, err := mlp.NewClassifierFromFiles("weights_input_hidden.csv", "weights_hidden_output.csv", "bias_hidden.csv", "bias_output.csv", dummyHandler)
	// brain, err := mlp.NewClassifier(28, 10, 6)
	if err != nil {
		panic(err)
	}
	brain.Classes = mlp.ReturnTargetClasses(targets)

	// err = brain.Train(scaled, targets, epochs)
	// if err != nil {
	// 	panic(err)
	// }

	score, err := brain.Score(scaled, targets)
	if err != nil {
		panic(err)
	}
	fmt.Println("Score:", score)
}
