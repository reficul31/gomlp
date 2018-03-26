package main

import (
	mlp "github.com/reficul31/mlp_classifier"
	"strings"
	"fmt"
)

var epochs = 5000

func stringHandler(input string) string {
	if strings.Compare(input, "g") == 0 {
		return "1"
	} else if strings.Compare(input, "b") == 0 {
		return "0"
	} else {
		return input
	}
}

func dummyHandler(input string) string {
	return input
}

func main() {
	data, err := mlp.ReadData("ionosphere.csv", stringHandler)
	if err != nil {
		panic(err)
		return
	}

	inputs, targets := mlp.PartitionData(data, "end")
	normalizer := mlp.NewNormalizer(len(inputs[0]))
	normalizer.Fit(inputs)
	normalized := normalizer.Transform(inputs, 1, -1)

	brain, err := mlp.NewMLPClassifierFromFiles("weights_input_hidden.csv", "weights_hidden_output.csv", "bias_hidden.csv", "bias_output.csv", dummyHandler)
	// brain, err := mlp.NewMLPClassifier(34, 15, 1)
	if err != nil {
		panic(err)
		return
	}

	// err = brain.Train(normalized, targets, epochs)
	// if err != nil {
	// 	panic(err)
	// 	return
	// }

	score, err := brain.Score(normalized, targets)
	if err != nil {
		panic(err)
		return
	}
	fmt.Println("Score:", score)
}
