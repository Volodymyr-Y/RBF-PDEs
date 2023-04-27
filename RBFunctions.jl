
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

function generate_2D_equally_spaced_points(N) # interiror + boundary points on unit square 
    x = LinRange(0,1,N)
    y = LinRange(0,1,N)
    y = y * ones(N)'
    x = ones(N) * x'
    points = hcat(vec(x),vec(y))'
    N_internal = (N-2)^2
    N_boundary = N^2 - N_internal
    boundary_points = zeros((2,N_boundary))
    internal_points = zeros((2,N_internal))
    idx_b, idx_i = 1,1
    for i in 1:N^2
        if 0.0 ∈ points[:,i] || 1.0 ∈ points[:,i]
            boundary_points[:,idx_b] = points[:,i]
            idx_b += 1
        else
            internal_points[:,idx_i] = points[:,i]
            idx_i +=1
        end
    end
    return internal_points,boundary_points
end

function point_difference_tensor(points1,points2) # reates NxMx2 tensor 
    l1 = size(points1)[2]
    l2 = size(points2)[2]
    A = [ (points1[k,i] - points2[k,j]) for i=1:l1, j=1:l2, k=1:2]
    return A
end

function apply(func, tensor,param)
    l1,l2,l3 = size(tensor)
    A = [ func(tensor[i,j,:],param) for i=1:l1, j=1:l2]
    return A
end

function linear(r,ϵ) # Linear RBF
    return abs(r)
end

function gaussian(r,ϵ) # guassian RBF
    return exp(-1*r*r*ϵ*ϵ)
end

function Δgaussian(r,ϵ)
    return 2*ϵ*ϵ*exp(-1*r*r*ϵ*ϵ)*(2*r*r*ϵ*ϵ - 1) - 2*ϵ*ϵ*exp(-1*r*r*ϵ*ϵ)
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

function wendland_C2(r::Real,ϵ)
    if r/ϵ >= 0.0 && r/ϵ <=1.0
        return ((1-r/ϵ)^4) * (4r/ϵ+1)
    else 
        return 0.0
    end
end