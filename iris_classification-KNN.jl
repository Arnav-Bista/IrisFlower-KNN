using Tables
using CSV
using PlotlyJS

function bubblesort(arr,mat,y)
    # arr -> Distance
    # mat -> X
    # y -> Y
    m = size(arr)[1]
    for i in 1:m
        for j in 1:(m-i-1)
            if arr[j] > arr[j+1]
                arr[j], arr[j+1] = arr[j+1], arr[j]
                mat[j,:], mat[j+1,:] = mat[j+1,:], mat[j,:]
                y[j], y[j+1] = y[j+1], y[j]
            end
        end
    end
    # y
    return arr,mat,y
end

function load(split = 70)
    a = CSV.File("iris.csv") |> Tables.matrix
    m,p = size(a)
    X = a[:,1:4]
    Y = a[:,5]
    # m = trunc(Int,m * split/100)
    # # TrainX = a[1:m,1:4]
    # # TrainY = a[1:m,5]
    # TrainData = a[1:m,:]
    # TestX = a[m+1:end,1:4]
    # TestY = a[m+1:end,5]
    # # TrainX,TrainY,TestX,TestY
    # TrainData,TestX,TestY
    X,Y
end

function minkowski_distance(a,b,p=1)
    (sum(abs.(a-b).^p))^(1/p)
end

function compare_all(data,X)
    m,p = size(X)
    n = fill(0.0,m)
    for i in 1:m
        n[i] = minkowski_distance(data,X[i,:])
    end
    n
end

#!!
function knn(data,Y,k = 5)
    data = data[1:k]
    a = Set(Y)
    # there are 4 classes in this dataset.
    nn = ["" 0;"" 0; "" 0; "" 0]
    z = 1
    for i in a
        nn[z,1] = i
        nn[z,2] = count(f->(f == i),data)
        z = z + 1
    end
    #return the nearest neighour
    max,idx = 0,1
    for i in 1:4
        if nn[i,2] > max
            max = nn[i,2]
            idx = i
        end
    end
    # debugging
    nn,nn[idx,1]
    # nn[idx,1]
end

function predict(data,X,Y)
    n = compare_all(data,X) #Gets the unordered distance to all the points
    tem,temp,yval = bubblesort(n,X,Y) #temp for debugging
    knn(yval,Y)
end

function predict_all(x,y,X,Y)
    m,p = size(x)
    correct = 0
    for i in 1:m
        prediction = predict(x[i,:],X,Y)
        if prediction == y[i]
            correct = correct + 1
        end
    end
    println(correct/m)
end

# trainx,trainy,testx,testy = load()

# HOW TO USE
# Load main dataset
X,Y = load()

# Say X[1,:] was an unknown dataset
result = predict(X[1,:],X,Y)
println(result)
# The neighbour with the highest number is the prediction

