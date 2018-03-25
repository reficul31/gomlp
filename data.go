package mlp

import (
	"math/rand"
	"strings"
)

func PartitionData(data [][]float64, position string) ([][]float64, [][]float64) {
	var rowLen int
	var targets [][]float64
	var inputs [][]float64
	if strings.Compare(position, "start") == 0 {
		for _, row := range(data) {
			rowLen = len(row)
			targets = append(targets, row[0:1])
			inputs = append(inputs, row[1:rowLen])
		}
	} else {
		for _, row := range(data) {
			rowLen = len(row)
			targets = append(targets, row[rowLen-1:rowLen])
			inputs = append(inputs, row[0:rowLen-1])
		}
	}
	return inputs, targets
}

func RandomDataSet(data [][]float64) (int, []float64) {
	max := len(data)-1
	min := 0
    index := rand.Intn(max - min) + min
    return index, data[index]
}