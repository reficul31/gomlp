package gomlp

import "math"

var precisionFactor = 0.65

// NewStandardScalar return a new StandarScalar pointer
func NewStandardScalar(columns int) *StandardScalar {
	return &StandardScalar{
		make([]float64, columns),
		make([]float64, columns),
	}
}

// NewNormalizer return a new Normalizer pointer
func NewNormalizer(columns int) *Normalizer {
	return &Normalizer{
		make([]float64, columns),
		make([]float64, columns),
	}
}

// GreatestIntegerFunction returns the greates integer of a slice
func GreatestIntegerFunction(data []float64) []int {
	output := make([]int, len(data))
	for index := range data {
		floor := math.Floor(data[index])
		if data[index] > (floor + precisionFactor) {
			output[index] = int(floor) + 1
		} else {
			output[index] = int(floor)
		}
	}
	return output
}

// Fit is used to populate the fields of StandardScalar
func (ss *StandardScalar) Fit(data [][]float64) {
	dataLen := len(data)
	for j := 0; j < len(data[0]); j++ {
		var sum = 0.0
		for i := 0; i < dataLen; i++ {
			sum = sum + data[i][j]
		}
		ss.mean[j] = sum / float64(dataLen)
		var variance = 0.0
		for i := 0; i < dataLen; i++ {
			variance = variance + math.Pow((data[i][j]-ss.mean[j]), 2)
		}
		variance = variance / float64(dataLen)
		ss.dev[j] = math.Sqrt(variance)
	}
}

// Fit is used to populate the fields of Normalizer
func (n *Normalizer) Fit(data [][]float64) {
	dataLen := len(data)
	for j := 0; j < len(data[0]); j++ {
		var min = 999.99
		var max = -999.99
		for i := 0; i < dataLen; i++ {
			if data[i][j] > max {
				max = data[i][j]
			}
			if data[i][j] < min {
				min = data[i][j]
			}
		}
		n.min[j] = min
		n.max[j] = max
	}
}

// Transform is used to standardize the values of the given slice
func (ss *StandardScalar) Transform(data [][]float64) [][]float64 {
	scaled := make([][]float64, len(data))
	for index := range data {
		scaled[index] = make([]float64, len(data[0]))
	}
	for j := 0; j < len(data[0]); j++ {
		for i := 0; i < len(data); i++ {
			scaled[i][j] = (data[i][j] - ss.mean[j]) / ss.dev[j]
		}
	}
	return scaled
}

// Transform is used to normalize the values of the given slice
func (n *Normalizer) Transform(data [][]float64, high, low float64) [][]float64 {
	scaled := make([][]float64, len(data))
	for index := range data {
		scaled[index] = make([]float64, len(data[0]))
	}
	for j := 0; j < len(data[0]); j++ {
		for i := 0; i < len(data); i++ {
			if n.min[j] == n.max[j] {
				scaled[i][j] = (high + low) / 2
			} else {
				scaled[i][j] = (data[i][j] - n.min[j]) / (n.max[j] - n.min[j])
				scaled[i][j] = (high-low)*scaled[i][j] + low
			}
		}
	}
	return scaled
}
