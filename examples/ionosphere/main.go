package main

import (
	"fmt"

	mlp "github.com/reficul31/mlp_classifier"
)

var epochs = 10

func main() {
	data, err := mlp.ReadData("ionosphere.csv")
	if err != nil {
		panic(err)
	}

	inputs, targets := mlp.PartitionData(data, "end")
	normalizer := mlp.NewNormalizer(len(inputs[0]))
	normalizer.Fit(inputs)
	normalized := normalizer.Transform(inputs, 1, -1)

	brain, err := mlp.NewClassifierFromFiles(1)
	// brain, err := mlp.NewClassifier(34, 1, 10)
	if err != nil {
		panic(err)
	}

	err = brain.Train(normalized, targets, epochs)
	if err != nil {
		panic(err)
	}

	score, err := brain.Score(normalized, targets)
	if err != nil {
		panic(err)
	}
	fmt.Println("Score:", score)
}
