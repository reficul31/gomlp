package main

import (
	"fmt"

	mlp "github.com/reficul31/mlp_classifier"
)

var epochs = 800

func dummyHandler(input string) string {
	return input
}

func main() {
	data, err := mlp.ReadData("training.csv", dummyHandler)
	if err != nil {
		panic(err)
	}

	mlp.RangeDetermine = mlp.NewRange()

	inputs, targets := mlp.PartitionData(data, "start")
	scalar := mlp.NewStandardScalar(len(inputs[0]))
	scalar.Fit(inputs)
	scaled := scalar.Transform(inputs)
	mlp.RangeDetermine.UpdateRangeArray2D(scaled)

	brain, err := mlp.NewClassifierFromFiles("weights_input_hidden.csv", "weights_hidden_output.csv", "bias_hidden.csv", "bias_output.csv", dummyHandler)
	// brain, err := mlp.NewClassifier(28, 10, 6)
	if err != nil {
		panic(err)
	}

	err = brain.Train(scaled, targets, epochs)
	if err != nil {
		panic(err)
	}

	score, err := brain.Score(scaled, targets)
	if err != nil {
		panic(err)
	}
	fmt.Println("Score:", score)
	min, max := mlp.RangeDetermine.ReturnRange()
	fmt.Println("MIN:", min, "MAX:", max)
}
