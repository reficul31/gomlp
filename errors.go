package mlp

import "errors"

var (
	// ErrNodeValue returns an error if a Node value is Negative
	ErrNodeValue = errors.New("Node values cannot be negative or zero")
	// ErrOnlyCSVFiles returns an error when a file other than the CSV file is provided
	ErrOnlyCSVFiles = errors.New("Only CSV files are allowed right now")
	// ErrRowColumnRange returns an error if the Row or Columns are not in range
	ErrRowColumnRange = errors.New("Rows or Columns not in range")
	// ErrRowColumnDimension returns an error when there is disparity in the number of rows and columns
	ErrRowColumnDimension = errors.New("Rows and Columns not of the same dimension")
	// ErrMuliplicationDimension returns an error when number of rows and columns are not equal
	ErrMuliplicationDimension = errors.New("Rows of the second matrix donot match the Columns of the first matrix")
)
