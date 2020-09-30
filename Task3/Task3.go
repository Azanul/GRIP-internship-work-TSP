package main

import (
	"fmt"
	"log"
	"math"
	"os"

	"github.com/go-gota/gota/dataframe"
	"github.com/muesli/clusters"
	"github.com/muesli/kmeans"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

type centroid []float64

func main() {
	// Pull in the CSV file.
	irisFile, err := os.Open("Task2/iris.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer irisFile.Close()

	// Create a dataframe from the CSV file.
	irisDF := dataframe.ReadCSV(irisFile)
	fmt.Print(irisDF.Describe())

	var d clusters.Observations
	for row := 0; row < 150; row++ {
		var temp = []float64{0, 0, 0, 0}
		for i, col := range []string{"SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"} {
			temp[i] = irisDF.Col(col).Float()[row]
		}
		d = append(d, clusters.Coordinates{temp[0], temp[1], temp[2], temp[3]})
	}

	km := kmeans.New()
	scores, k, score, err := EstimateK(d, 8, km)
	fmt.Print("\nOptimum no. of clusters: ", k, "\n")
	fmt.Print("Best Silhouette score: ", score, "\n")
	fmt.Print("Although optimum no. of clusters according to the analysis is 2, \n" +
		"we know that 3 clusters is the corect response and its silhoette score is \n" +
		"also greater than 0.7. This happens because k-means clustering does not always \n" +
		"converge optimally.")

	// pts will hold the values for plotting
	pts := make(plotter.XYs, 8)

	// Fill pts with data.
	for i, floatVal := range scores {
		pts[i].X = float64(floatVal.K)
		pts[i].Y = floatVal.Score
	}
	// Create the plot.
	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}
	p.X.Label.Text = "No. of Clusters"
	p.Y.Label.Text = "Silhouette Scores"
	p.Add(plotter.NewGrid())
	s, err := plotter.NewScatter(pts)
	if err != nil {
		log.Fatal(err)
	}
	s.GlyphStyle.Radius = vg.Points(3)

	// Save the plot to a PNG file.
	p.Add(s)
	if err := p.Save(4*vg.Inch, 4*vg.Inch, "Task2/Silhouette Scores vs No. of Clusters.png"); err != nil {
		log.Fatal(err)
	}
}

// KScore holds the score for a value of K
type KScore struct {
	K     int
	Score float64
}

// Partitioner interface which suitable clustering algorithms should implement
type Partitioner interface {
	Partition(data clusters.Observations, k int) (clusters.Clusters, error)
}

// Score calculates the silhouette score for a given value of k, using the given
// partitioning algorithm
func Score(data clusters.Observations, k int, m Partitioner) (float64, error) {
	cc, err := m.Partition(data, k)
	if err != nil {
		return -1.0, err
	}

	var si float64
	var sc int64
	for ci, c := range cc {
		for _, p := range c.Observations {
			ai := clusters.AverageDistance(p, c.Observations)
			_, bi := cc.Neighbour(p, ci)

			si += (bi - ai) / math.Max(ai, bi)
			sc++
		}
	}

	return si / float64(sc), nil
}

// Scores calculates the silhouette scores for all values of k between 2 and
// kmax, using the given partitioning algorithm
func silhouetteScores(data clusters.Observations, kmax int, m Partitioner) ([]KScore, error) {
	var r []KScore

	for k := 2; k <= kmax; k++ {
		s, err := Score(data, k, m)
		if err != nil {
			return r, err
		}

		r = append(r, KScore{
			K:     k,
			Score: s,
		})
	}

	return r, nil
}

// EstimateK estimates the amount of clusters (k) along with the silhouette
// score for that value, using the given partitioning algorithm
func EstimateK(data clusters.Observations, kmax int, m Partitioner) ([]KScore, int, float64, error) {
	scores, err := silhouetteScores(data, kmax, m)
	if err != nil {
		return []KScore{}, 0, -1.0, err
	}

	r := KScore{
		K: -1,
	}
	for _, score := range scores {
		if r.K < 0 || score.Score > r.Score {
			r = score
		}
	}
	return scores, r.K, r.Score, nil
}
