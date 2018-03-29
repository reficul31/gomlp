package main

import (
	"fmt"

	mlp "github.com/reficul31/mlp_classifier"
)

var epochs = 100

func main() {
	data, err := mlp.ReadData("training.csv")
	if err != nil {
		panic(err)
	}

	inputs, targets := mlp.PartitionData(data, "start")
	scalar := mlp.NewStandardScalar(len(inputs[0]))
	scalar.Fit(inputs)
	scaled := scalar.Transform(inputs)

	brain, err := mlp.NewClassifierFromFiles(1)
	// brain, err := mlp.NewClassifier(28, 6, 10)
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
}
