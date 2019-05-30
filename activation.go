package gomlp

import "math"

var sigmoid = ActivationFunction{
	function: func(x float64) float64 {
		return 1 / (1 + math.Exp(-1*x))
	},
	dfunction: func(y float64) float64 {
		return y*(1-y)
	},
}

var tanh = ActivationFunction{
	function: func(x float64) float64 {
		return math.Atanh(x)
	},
	dfunction: func(y float64) float64 {
		return 1-y*y
	},
}