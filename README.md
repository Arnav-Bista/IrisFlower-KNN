# Iris Flower Classification - KNN Classification

## What?

The [Iris Flower Dataset](https://archive.ics.uci.edu/ml/datasets/iris) is a
multivariate classification problem that is solved here using the K Nearest
Neighbours (KNN) algorithm.

The dataset consists of 5 values:

1. Sepal length in cm (continuous)
2. Sepal width in cm (continuous)
3. Petal length in cm (continuous)
4. Petal width in cm (continuous)
5. Class (String)

Values 1 to 4 will be utilised as our inputs and the 5th value will be treated
as the output.

## KNN - K Nearest Neighbours

Whilst KNN falls under supervised learning, the model does not really *"learn"* anything in particular
but it instead simply memorises the training dataset.


KNN sticks to the philosophy of "birds of a feather flock together". 
Given an integer K, and a data point X, it will seek out the K nearest points from X. 
Then, whichever class is the most common in those K points is assigned to X. Hence, the name K Nearest Neighbours.

In Pseudocode:

```nolanguage
KNN(X,K):
    Calculate the distances between X and the datset
    Find the K closest points about X
    Count the occurances of each class
    The class with the highest occurance is the prediction
end
```

## Code

The code is quite straight forward. I used `Julia` as my language of choice for
this project as I love how math notation is integrated within the language and its easy-to-read python like syntax.

I've implemented the Minkowski Distance formula to compute the distance if we
wish to compute both the Euclidean and Manhattan distance.


## Conclusion

With the standard 70:30 Split in training and testing, I consistently get about 90% correctly classified with `K = 3`. 
Although this is a lazy algorithm, it's quite good at classifying.

Besides from classifications, it can also be used to fill in missing values from a dataset to then train a more sophisticated model.

This model does have its limitations. 
It works wonderfully in medium to small sized datasets, however, when datasets get large with tens 
of thousands of datapoints - each with their own tens or hundreds of features - it becomes computationally expensive.
Not only does the entire dataset has to be in memory, we need to calculate the
distance from one dataset to all other datasets.
