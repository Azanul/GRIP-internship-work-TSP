package main

import (
	"fmt"
	"log"
	"math"
	"os"
	"strconv"

	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
	"github.com/sajari/regression"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func main() {
	f, err := os.Open("Task2/student_scores - student_scores.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	studentDF := dataframe.ReadCSV(f)

	// Describe the read dataframe
	fmt.Println(studentDF.Describe())

	// Create a Scatter plot for the dataframe for
	// visual evaluation.
	// Assign X & Y columns for the plot
	yVals := studentDF.Col("Scores").Float()
	colName := "Hours"

	// pts will hold the values for plotting
	pts := make(plotter.XYs, studentDF.Nrow())

	// Fill pts with data.
	for i, floatVal := range studentDF.Col(colName).Float() {
		pts[i].X = floatVal
		pts[i].Y = yVals[i]
	}

	// Create the plot.
	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}
	p.X.Label.Text = colName
	p.Y.Label.Text = "Scores"
	p.Add(plotter.NewGrid())
	s, err := plotter.NewScatter(pts)
	if err != nil {
		log.Fatal(err)
	}
	s.GlyphStyle.Radius = vg.Points(3)

	// Save the plot to a PNG file.
	p.Add(s)
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "Task2/Hours V Scores.png"); err != nil {
		log.Fatal(err)
	}

	// Train-Test Split
	// Calculate the number of elements in each set.
	trainingNum := (4 * studentDF.Nrow()) / 5
	testNum := studentDF.Nrow() / 5
	if trainingNum+testNum < studentDF.Nrow() {
		trainingNum++
	}

	// Create the subset indices.
	trainingIdx := make([]int, trainingNum)
	testIdx := make([]int, testNum)

	// Enumerate the training & testing indices.
	for i := 0; i < trainingNum; i++ {
		trainingIdx[i] = i
	}
	for i := 0; i < testNum; i++ {
		testIdx[i] = trainingNum + i
	}

	// Create the subset dataframes.
	trainingDF := studentDF.Subset(trainingIdx)
	testDF := studentDF.Subset(testIdx)

	// In this case we are going to try and model our Scores (y)
	// by the Hours plus an intercept. As such, create
	// the struct needed to train a model using github.com/sajari/regression.
	var r regression.Regression
	r.SetObserved("Scores")
	r.SetVar(0, "Hours")

	// Loop of records in the CSV, adding the training data to the regression value.
	for i, record := range trainingDF.Records() {

		// Skip the header.
		if i == 0 {
			continue
		}

		// Parse the Scores regression measure, or "y".
		yVal, err := strconv.ParseFloat(record[1], 64)
		if err != nil {
			log.Fatal(err)
		}

		// Parse the Hours value.
		tvVal, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			log.Fatal(err)
		}

		// Add these points to the regression value.
		r.Train(regression.DataPoint(yVal, []float64{tvVal}))
	}

	// Train/fit the regression model.
	r.Run()
	// Output the trained model parameters.
	fmt.Printf("\nRegression Formula:\n%v\n\n", r.Formula)

	// Loop over the test data predicting y and evaluating the prediction
	// with the mean absolute error while creating a dataframe for
	// the predicted values for comparision with observed ones.
	var mAE float64
	var predicted, observed []float64
	for i, record := range testDF.Records() {

		// Skip the header.
		if i == 0 {
			continue
		}

		// Parse the observed Scores, or "y".
		yObserved, err := strconv.ParseFloat(record[1], 64)
		if err != nil {
			log.Fatal(err)
		}
		observed = append(observed, yObserved)

		// Parse the Hours value.
		tvVal, err := strconv.ParseFloat(record[0], 64)
		if err != nil {
			log.Fatal(err)
		}

		// Predict y with our trained model.
		yPredicted, err := r.Predict([]float64{tvVal})

		predicted = append(predicted, yPredicted)
		// Add the to the mean absolute error.
		mAE += math.Abs(yObserved-yPredicted) / float64(len(testDF.Records()))
	}
	predDF := dataframe.New(
		series.New(observed, series.Int, "Observed"),
		series.New(predicted, series.Float, "Predicted"),
	)
	fmt.Println(predDF)
	// Output the MAE to standard out.
	fmt.Printf("MAE = %0.2f\n\n", mAE)

	// Define the prediction function
	// predict := func (x float64) float64 { return 3.0313 + x*9.5204 }

	// Create the regression line
	line := plotter.NewFunction(func(x float64) float64 {
		res, _ := r.Predict([]float64{x})
		return res
	})

	plotter.DefaultLineStyle.Width = vg.Points(1)
	plotter.DefaultGlyphStyle.Radius = vg.Points(2)

	p.Add(s, line)
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "Task2/Regression Line.png"); err != nil {
		log.Fatal(err)
	}

	finalScore, _ := r.Predict([]float64{9.25})
	fmt.Printf("a student studied for 9.25 hrs his score will be %f", finalScore)
	// We could obtain an even more accurate value by replace 'trainingDF' with 'studentDF' in
	// line 97. Since it would give more training data to the model.
}
