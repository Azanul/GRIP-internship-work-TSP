package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path"

	"github.com/go-echarts/go-echarts/charts"
	"github.com/go-gota/gota/dataframe"
	"github.com/sjwhitworth/golearn/base"
	"github.com/sjwhitworth/golearn/evaluation"
	"github.com/sjwhitworth/golearn/trees"
)

var tree *trees.ID3DecisionTree

func main() {
	// Open the CSV file.
	irisFile, err := os.Open("Task3/Iris.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer irisFile.Close()

	// Create a dataframe from the CSV file.
	// The types of the columns will be inferred.
	irisDF := dataframe.ReadCSV(irisFile)

	//Write the modified csv to disk.
	f, err := os.Create("Task3/Iris-mod.csv")
	if err != nil {
		log.Fatal(err)
	}
	irisDF.Select([]int{1, 2, 3, 4}).WriteCSV(f)

	// Read in the iris data set into golearn "instances".
	irisData, err := base.ParseCSVToInstances("Task3/Iris-mod.csv", true)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Print(irisData)
	// This is to seed the random processes involved in building the
	// decision tree.
	rand.Seed(44111342)

	// We will use the ID3 algorithm to build our decision tree.  Also, we
	// will start with a parameter of 0.6 that controls the train-prune split.
	tree = trees.NewID3DecisionTree(0.6)

	// Use cross-fold validation to successively train and evaluate the model
	// on 5 folds of the data set.
	cv, err := evaluation.GenerateCrossFoldValidationConfusionMatrices(irisData, tree, 5)
	if err != nil {
		log.Fatal(err)
	}

	// Get the mean, variance and standard deviation of the accuracy for the
	// cross validation.
	mean, variance := evaluation.GetCrossValidatedMetric(cv, evaluation.GetAccuracy)
	stdev := math.Sqrt(variance) // Output the cross metrics to standard out.
	fmt.Printf("\nAccuracy\n%.2f (+/- %.2f)\n\n", mean, stdev*2)

	fmt.Println(tree.Root.SplitRule.String())

	//Simple approach for small tree
	var graphNodes = []charts.GraphNode{
		{Name: tree.Root.SplitRule.String()},
		{Name: "DecisionTreeRule(SepalWidthCm <= 3.050000)"},
		{Name: "Leaf 1"},
		{Name: "Leaf 2"},
	}

	// DFS approach
	// var queue = []*trees.DecisionTreeNode{tree.Root}
	// for len(queue) > 0 {
	// 	curr := queue[0]
	// 	queue = queue[1:]
	// 	for _, child := range curr.Children {
	// 		fmt.Println("0", child)
	// 		if child.SplitRule.SplitAttr != nil {
	// 			fmt.Println("77", child)
	// 			queue = append(queue, child)
	// 			graphNodes = append(graphNodes, charts.GraphNode{Name: child.SplitRule.String()})
	// 		} else {
	// 			graphNodes = append(graphNodes, charts.GraphNode{Name: child.String()})
	// 		}
	// 	}
	// }

	fmt.Println(tree)
	genLinks := func() []charts.GraphLink {
		// queue := []*trees.DecisionTreeNode{tree.Root}
		links := make([]charts.GraphLink, 0)
		// for len(queue) > 0 {
		// 	curr := queue[0]
		// 	queue = queue[1:]
		// 	for _, child := range curr.Children {
		// 		if child.SplitRule.SplitAttr != nil {
		// 			queue = append(queue, child)
		// 			links = append(links, charts.GraphLink{Source: curr.SplitRule.String(), Target: child.SplitRule.String()})
		// 		}
		// 	}
		// }
		links = append(links, charts.GraphLink{Source: tree.Root.SplitRule.String(), Target: "DecisionTreeRule(SepalWidthCm <= 3.050000)"})
		links = append(links, charts.GraphLink{Source: "DecisionTreeRule(SepalWidthCm <= 3.050000)", Target: "Leaf 2"})
		links = append(links, charts.GraphLink{Source: tree.Root.SplitRule.String(), Target: "Leaf 1"})
		return links
	}

	graphCircle := func(nodes []charts.GraphNode) *charts.Graph {
		graph := charts.NewGraph()
		graph.SetGlobalOptions(charts.TitleOpts{Title: "Graph"})
		graph.Add("graph", nodes, genLinks(),
			charts.GraphOpts{Layout: "circular", Force: charts.GraphForce{Repulsion: 10000}},
			charts.LabelTextOpts{Show: true, Position: "right"},
			charts.LineStyleOpts{Curveness: -0.2},
		)
		return graph
	}

	graphHandler := func(w http.ResponseWriter, _ *http.Request) {
		page := charts.NewPage(myRouter.RouterOpts)
		page.Add(
			graphCircle(graphNodes),
		)
		f, err := os.Create(getRenderPath("graph.html"))
		if err != nil {
			log.Println(err)
		}
		page.Render(w, f)
	}

	http.HandleFunc("/", logTracing(graphHandler))
	http.HandleFunc("/graph", logTracing(graphHandler))
	log.Println("\nRun server at " + host)
	http.ListenAndServe("127.0.0.1:8080", nil)
}

func logTracing(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		log.Printf("\n\nTracing request for %s\n", r.RequestURI)
		next.ServeHTTP(w, r)
	}
}

func getRenderPath(f string) string {
	return path.Join("html", f)
}

const host = "http://127.0.0.1:8080"

type router struct {
	name string
	charts.RouterOpts
}

var myRouter = router{"graph", charts.RouterOpts{URL: host + "/graph", Text: "Graph"}}
