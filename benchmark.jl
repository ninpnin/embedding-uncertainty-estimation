using Dates
dimensionality = 100
observations = 150 * 5

A = rand(dimensionality,dimensionality)
C = rand(dimensionality,dimensionality)
B = inv(A)

ts1 = now()
B = inv(A)
ts2 = now()

println(ts2 - ts1)

ts1 = now()
for i in 1:50
    X = rand(dimensionality,dimensionality)
    Y = inv(X)
end
ts2 = now()

println(ts2 - ts1)

ts1 = now()
A = rand(dimensionality,observations)
for i in 1:50
    b = rand(observations)
    Z = (b'.*A ) *A'
end
ts2 = now()
println(ts2 - ts1)


function tt(v, A)
    Z = (b'.*A ) *A'
end 

    
using Distributions 
A = rand(Gamma(1,2),10)

ts1 = now()
b = rand(observations)
for i in 1:5000
    b = rand(observations)
    O = tt(b, A)
end
ts2 = now()
println(ts2 - ts1)
