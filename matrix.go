package mlp

import "math/rand"

func NewMatrix(rows, cols int) (*Matrix, error) {
	if rows>0 && cols>0 {
		data := make([][]float64, rows)
		for i := 0; i < rows; i++ {
			data[i] = make([]float64, cols)
		}
		return &Matrix{
			data: data,
			rows: rows,
			cols: cols,
		}, nil
	} else {
		return &Matrix{}, ErrRowColumnRange
	}
}

func MapMultiply(a,b *Matrix) (*Matrix, error) {
	if a.rows != b.rows && a.cols != b.cols {
		return &Matrix{}, ErrRowColumnDimension
	}

	m, _ := NewMatrix(a.rows, a.cols)
	for i := 0; i < a.rows; i++ {
		for j := 0; j < a.cols; j++ {
			m.data[i][j] = a.data[i][j]*b.data[i][j]
		}
	}
	return m, nil
}

func Map(m *Matrix, mapFunc func(float64) float64) *Matrix {
	new, _ := NewMatrix(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			new.data[i][j] = mapFunc(m.data[i][j]);
		}
	}
	return new
}

func Subtract(a,b *Matrix) (*Matrix, error) {
	if a.rows != b.rows && a.cols != b.cols {
		return &Matrix{}, ErrRowColumnDimension
	}

	m, _ := NewMatrix(a.rows, a.cols)
	for i := 0; i < a.rows; i++ {
		for j := 0; j<a.cols; j++ {
			m.data[i][j] = a.data[i][j] - b.data[i][j]
		}
	}
	return m, nil
}

func Add(a,b *Matrix) (*Matrix, error) {
	if a.rows != b.rows && a.cols != b.cols {
		return &Matrix{}, ErrRowColumnDimension
	}

	m, _ := NewMatrix(a.rows, a.cols)
	for i := 0; i < a.rows; i++ {
		for j := 0; j<a.cols; j++ {
			m.data[i][j] = a.data[i][j] + b.data[i][j]
		}
	}
	return m, nil
}

func Multiply(a,b *Matrix) (*Matrix, error) {
	if a.cols != b.rows {
		return &Matrix{}, ErrMuliplicationDimension
	}

	var sum float64
	m, _ := NewMatrix(a.rows, b.cols)
	for i := 0; i < a.rows; i++ {
		for j := 0; j < b.cols; j++{
			sum = 0
			for k := 0; k < a.cols; k++ {
				sum = sum + a.data[i][k]*b.data[k][j];
			}
			m.data[i][j] = sum
		}
	}
	return m, nil
}

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

func (m *Matrix) ConvertFromMatrixToArray1D() []float64 {
	var data []float64
	for i := 0; i < m.rows; i++ {
		for j := 0; j<m.cols; j++ {
			data = append(data, m.data[i][j])
		}
	}
	return data
}

func (m *Matrix) ConvertFromMatrixToArray2D() [][]float64 {
	var data [][]float64
	for _, row := range(m.data) {
		dataRow := make([]float64, m.cols)
		for i := 0; i < m.cols; i++ {
			dataRow[i] = row[i]
		}
		data = append(data, dataRow)
	}
	return data
}

func (m *Matrix) Randomize(max, min float64) {
	diff := max-min
	for i := 0; i < m.rows; i++ {
		for j := 0; j<m.cols; j++ {
			m.data[i][j] = diff*rand.Float64()+min
		}
	}
}

func (m *Matrix) Copy() *Matrix {
	m1, _ := NewMatrix(m.rows, m.cols)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m1.data[i][j] = m.data[i][j];
		}
	}
	return m1
}

func (m *Matrix) Add(n float64) {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.data[i][j] = m.data[i][j]+n;
		}
	}
}

func (m *Matrix) Subtract(n float64) {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.data[i][j] = m.data[i][j]-n;
		}
	}
}

func (m *Matrix) Multiply(n float64) {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.data[i][j] = m.data[i][j]*n;
		}
	}
}

func (m *Matrix) Transpose() *Matrix {
	m1, _ := NewMatrix(m.cols, m.rows)
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m1.data[j][i] = m.data[i][j];
		}
	}
	return m1
}

func (m *Matrix) Map(mapFunc func(float64) float64) {
	for i := 0; i < m.rows; i++ {
		for j := 0; j < m.cols; j++ {
			m.data[i][j] = mapFunc(m.data[i][j]);
		}
	}
}