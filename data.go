package mlp

import (
	"math/rand"
	"sort"
	"strings"
)

// PartitionData is used to take a data set and return the input and target data set
func PartitionData(data [][]float64, position string) ([][]float64, [][]float64) {
	var rowLen int
	var targets [][]float64
	var inputs [][]float64
	if strings.Compare(position, "start") == 0 {
		for _, row := range data {
			rowLen = len(row)
			targets = append(targets, row[0:1])
			inputs = append(inputs, row[1:rowLen])
		}
	} else {
		for _, row := range data {
			rowLen = len(row)
			targets = append(targets, row[rowLen-1:rowLen])
			inputs = append(inputs, row[0:rowLen-1])
		}
	}
	return inputs, targets
}

// RandomDataSet is used to provide random dataset values for training
func RandomDataSet(data [][]float64) (int, []float64) {
	max := len(data) - 1
	min := 0
	index := rand.Intn(max-min) + min
	return index, data[index]
}

// FindInArray returns the position of an element in the slice
func FindInArray(arr []float64, key float64) int {
	for index, element := range arr {
		if element == key {
			return index
		}
	}
	return -1
}

// ReturnTargetClasses returns the various target classes sorted by value present in the target dataset
func ReturnTargetClasses(target [][]float64) []float64 {
	var classes = []float64{0}
	for _, row := range target {
		if FindInArray(classes, row[0]) == -1 {
			classes = append(classes, row[0])
		}
	}
	sort.Slice(classes, func(i, j int) bool {
		return classes[i] < classes[j]
	})
	return classes
}
