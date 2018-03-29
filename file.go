package mlp

import (
	"bufio"
	"encoding/csv"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// ReadData reads the data from a CSV file and returns the dataset
func ReadData(filename string) ([][]float64, error) {
	var data [][]float64

	file, err := os.Open(filename)
	if err != nil {
		return data, err
	}

	if strings.Compare(filepath.Ext(filename), ".csv") != 0 {
		return data, ErrOnlyCSVFiles
	}

	defer file.Close()

	reader := csv.NewReader(bufio.NewReader(file))
	if err != nil {
		return data, err
	}

	for {
		row, err := reader.Read()
		if err == io.EOF {
			break
		}

		dataRow := make([]float64, len(row))
		for i := 0; i < len(row); i++ {
			dataRow[i], err = strconv.ParseFloat(row[i], 64)
			if err != nil {
				return data, err
			}
		}
		data = append(data, dataRow)
	}

	return data, nil
}

// WriteData writes data to a CSV file
func WriteData(filename string, data [][]float64) error {
	file, err := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE, 0644)
	if err != nil {
		return err
	}

	if strings.Compare(filepath.Ext(filename), ".csv") != 0 {
		return ErrOnlyCSVFiles
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	for _, row := range data {
		writeData := make([]string, len(row))
		for i := 0; i < len(row); i++ {
			writeData[i] = strconv.FormatFloat(row[i], 'f', 6, 64)
		}
		err := writer.Write(writeData)
		if err != nil {
			return err
		}
	}
	return err
}
