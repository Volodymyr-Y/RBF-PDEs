
using Distances
using LinearAlgebra 
using FastHalton
using SparseArrays
using NearestNeighbors
function abcd(x)
    return x*x
end



function random_collocation_points(N,x1,x2,y1,y2)
    # generates N 2D points in a rectange between x1 x2 and y1 y2 
    list = rand(Float64,(2,N))
    list[1,:]*= (x2-x1)
    list[2,:]*= (y2-y1)
    list[1,:]= list[1,:] .+x1
    list[2,:]= list[2,:] .+y1
    return list
end

function generate_2D_Halton_points(N) # 2-3 Halton sequence looks random enough 
    list = zeros(Float64,(2,N))
    list[1,:] = collect(HaltonSeq(2, N,0 ))
    list[2,:] = collect(HaltonSeq(3, N,0 ))
    return float(list)
end

function linear(r,ϵ) # Linear RBF
    return abs(r)
end

function gaussian(r,ϵ) # guassian RBF
    return exp(-1*r*r*ϵ*ϵ)
end

function sinusoid(x,y,period) # random function for interpolation
    return sin(2*x*π/period)*sin(2*y*π/period)
end

function frankes_func(x,y) # literature standard benchmark for interpolation
    f1 = 0.75*exp(-((9*x.-2).^2 + (9*y.-2).^2)/4)
    f2 = 0.75*exp((-1/49)*(9*x+1).^2 - (1/10)*(9*y+1).^2)
    f3 =0.5*exp((-1/4)*((9*x-7).^2 + (9*y-3).^2))
    f4 = 0.2*exp(-(9*x-4).^2 - (9*y-7).^2)
    return f1+f2+f3-f4
end