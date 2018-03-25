package mlp

import "errors"

var (
	ErrNodeValue              = errors.New("Node values cannot be negative")
	ErrOnlyCSVFiles           = errors.New("Only CSV files are allowed right now")
	ErrRowColumnRange         = errors.New("Rows or Columns not in range")
	ErrRowColumnDimension     = errors.New("Rows and Columns not of the same dimension")
	ErrMuliplicationDimension = errors.New("Rows of the second matrix donot match the Columns of the first matrix")
)