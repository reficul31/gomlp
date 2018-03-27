package mlp

import "math/rand"

// NewMatrix is used to return a Pointer to Matrix
func NewMatrix(rows, cols int) (*Matrix, error) {
	if rows > 0 && cols > 0 {
		data := make([][]float64, rows)
		for i := 0; i < rows; i++ {
			data[i] = make([]float64, cols)
		}
		return &Matrix{
			data: data,
			rows: rows,
			cols: cols,
		}, nil
	}

	return &Matrix{}, ErrRowColumnRange
}

// MapMultiply is used for dot product of two matrices
func MapMultiply(a, b *Matrix) (*Matrix, error) {
	if a.rows != b.rows && a.cols != b.cols {
		return &Matrix{}, ErrRowColumnDimension
	}

	m, _ := NewMatrix(a.rows, a.cols)
	for i := 0; i < a.rows; i++ {
		for j := 0; j < a.cols; j++ {
			m.data[i][j] = a.data[i][j] * b.data[i][j]
		}
	}
	return m, nil
}

// Map takes a function and applies it to all the elements of a slice
func Map(m *Matrix, mapFunc func(float64) float64) *Matrix {
	new, _ := NewMatrix(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			new.data[i][j] = mapFunc(m.data[i][j])
		}
	}
	return new
}

// Subtract takes to elements and subtracts them
func Subtract(a, b *Matrix) (*Matrix, error) {
	if a.rows != b.rows && a.cols != b.cols {
		return &Matrix{}, ErrRowColumnDimension
	}

	m, _ := NewMatrix(a.rows, a.cols)
	for i := 0; i < a.rows; i++ {
		for j := 0; j < a.cols; j++ {
			m.data[i][j] = a.data[i][j] - b.data[i][j]
		}
	}
	return m, nil
}

// Add takes two matrices and adds them
func Add(a, b *Matrix) (*Matrix, error) {
	if a.rows != b.rows && a.cols != b.cols {
		return &Matrix{}, ErrRowColumnDimension
	}

	m, _ := NewMatrix(a.rows, a.cols)
	for i := 0; i < a.rows; i++ {
		for j := 0; j < a.cols; j++ {
			m.data[i][j] = a.data[i][j] + b.data[i][j]
		}
	}
	return m, nil
}

// Multiply is used to perform matrix multiplication
func Multiply(a, b *Matrix) (*Matrix, error) {
	if a.cols != b.rows {
		return &Matrix{}, ErrMuliplicationDimension
	}

	var sum float64
	m, _ := NewMatrix(a.rows, b.cols)
	for i := 0; i < a.rows; i++ {
		for j := 0; j < b.cols; j++ {
			sum = 0
			for k := 0; k < a.cols; k++ {
				sum = sum + a.data[i][k]*b.data[k][j]
			}
			m.data[i][j] = sum
		}
	}
	return m, nil
}

// ConvertFromArrayToMatrix1D converts an Array object to a Matrix
func ConvertFromArrayToMatrix1D(data []float64) (*Matrix, error) {
	m, err := NewMatrix(len(data), 1)
	if err != nil {
		return &Matrix{}, err
	}
	for i := 0; i < len(data); i++ {
		m.data[i][0] = data[i]
	}
	return m, nil
}

// ConvertFromArray2DToMatrix converts an Array object to a Matrix
func ConvertFromArray2DToMatrix(data [][]float64) (*Matrix, error) {
	m, err := NewMatrix(len(data), len(data[0]))
	if err != nil {
		return m, err
	}

	for i, row := range data {
		for j, element := range row {
			m.data[i][j] = element
		}
	}
	return m, nil
}

// ConvertFromMatrixToArray1D converts a Matrix object to an array
func (m *Matrix) ConvertFromMatrixToArray1D() []float64 {
	var data []float64
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			data = append(data, m.data[i][j])
		}
	}
	return data
}

// ConvertFromMatrixToArray2D converts a Matrix object to an array
func (m *Matrix) ConvertFromMatrixToArray2D() [][]float64 {
	var data [][]float64
	for _, row := range m.data {
		dataRow := make([]float64, m.cols)
		for i := 0; i < m.cols; i++ {
			dataRow[i] = row[i]
		}
		data = append(data, dataRow)
	}
	return data
}

// Randomize is used to initialize all the values to a random value
func (m *Matrix) Randomize(max, min float64) {
	diff := max - min
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.data[i][j] = diff*rand.Float64() + min
		}
	}
}

// Copy creates a copy of a Matrix
func (m *Matrix) Copy() *Matrix {
	m1, _ := NewMatrix(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m1.data[i][j] = m.data[i][j]
		}
	}
	return m1
}

// Add a value to each element of a matrix
func (m *Matrix) Add(n float64) {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.data[i][j] = m.data[i][j] + n
		}
	}
}

// Subtract a value from each element of a matrix
func (m *Matrix) Subtract(n float64) {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.data[i][j] = m.data[i][j] - n
		}
	}
}

// Multiply a value from each element of a matrix
func (m *Matrix) Multiply(n float64) {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.data[i][j] = m.data[i][j] * n
		}
	}
}

// Transpose a Matrix
func (m *Matrix) Transpose() *Matrix {
	m1, _ := NewMatrix(m.cols, m.rows)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m1.data[j][i] = m.data[i][j]
		}
	}
	return m1
}

// Map applies a function to all the elements of a Matrix
func (m *Matrix) Map(mapFunc func(float64) float64) {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.data[i][j] = mapFunc(m.data[i][j])
		}
	}
}

// FindGreatestIndex finds the greatest index of an element in the array
func (m *Matrix) FindGreatestIndex() int {
	var max = -999.99
	var maxPos = 0
	for index, row := range m.data {
		for jindex, element := range row {
			if element > max {
				max = element
				maxPos = index*len(m.data[0]) + jindex
			}
		}
	}
	return maxPos
}
