

using StaticArrays
using Distances
using LinearAlgebra 
using FastHalton
using SparseArrays
using NearestNeighbors
using Symbolics
using Latexify
#using Revise
function abcd(x)
    return x*x
end

abstract type Point  end
abstract type BC end
abstract type Dirichlet <: BC end
abstract type Neumann <: BC end


mutable struct DomainPoint{N} <: Point
    pos::SVector{N,Float64}
end

mutable struct BoundaryPoint{N} <: Point
    pos::SVector{N,Float64}
    normal::SVector{N,Float64}
end

mutable struct BoundaryConditions{T} 
    point_set::Array{BoundaryPoint}
    g::Function 
end

mutable struct Problem
    domain_points::Array{DomainPoint}
    boundary_points::Array{BoundaryPoint}
    bc::Array{BoundaryConditions}
    forcing::Function
    initial_conditions::Function
    tspan
    dt

end

mutable struct Solution
    domain_points::Array{DomainPoint}
    boundary_points::Array{BoundaryPoint}
    sol
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

function point_difference_tensor(points1::Union{Vector{BoundaryPoint},Vector{DomainPoint}},
    points2::Union{Vector{BoundaryPoint},Vector{DomainPoint}}) # reates NxMx2 tensor 
    l1 = length(points1)
    l2 = length(points2)
    A = [ (points1[i].pos[k] - points2[j].pos[k]) for i=1:l1, j=1:l2, k=1:2]
    return A
end

function apply(func, tensor,param)
    l1,l2,l3 = size(tensor)
    A = [Base.@invokelatest func(tensor[i,j,:],param) for i=1:l1, j=1:l2]
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

function generate_vector_function(func::Function,points) 
    N = size(points)[2]
    res = zeros(N)
    x = points[1,:]
    y = points[2,:]
    function aa(time)
        return func.(x,y,Ref(time))
    end
    return aa
end

function generate_vector_function(func_array::Vector{Function},points) 
    N = size(points)[2] 
    l = length(func_array)
    res = zeros(N*l)
    x = points[1,:]
    y = points[2,:]
    function aa(time)
        for i in 1:l
            res[(i-1)*N+1:(i)*N] .= func_array[i].(x,y,Ref(time))
        end
        return res
    end
    return aa
end


function generate_vector_function(func1::Function,func2::Function,points;Mixed=true) 
    N = size(points)[2]
    x = points[1,:]
    y = points[2,:]
    if Mixed == true
        function aa(time)
            res = zeros(2*N) # prealocate resulting vector
            for i in 1:N
                res[2i-1] = func1(x[i],y[i],time)
                res[2i] = func2(x[i],y[i],time)
            end
            return res
        end
        return aa # return function
    else
        function bb(time)
            res = zeros(2*N) # prealocate resulting vector
            for i in 1:N
                res[i] = func1(x[i],y[i],time)
                res[N+i] = func2(x[i],y[i],time)
            end
            return res
        end
        return bb # return function
    end
end

function apply_matrix(func, tensor, param)
    N = size(tensor)[1]
    M = size(tensor)[2]
    res = zeros((N*2,M*2))
    #display(res)
    for i = 1:N
        for j = 1:M
            res[i*2-1:i*2,j*2-1:j*2] = func(tensor[i,j,:],param) #[1 1;2 2]
        end
    end
    return res
end

const ABCD = 10

function max_error(computed_sol,reference_sol,n::Int)
    range = LinRange(0.0,1.2,1000)
    m_array = zeros(length(range))
    for (i,t) in enumerate(range)
        m_array[i] = maximum(abs.(computed_sol(t)[1:n] - reference_sol(t)[1:n]))
    end
    return maximum(m_array)
end 

function construct_kernel_array(matrix_kernel,functionals1,functionals2)
    N1 = length(functionals1)
    N2 = length(functionals2)
    M = Matrix{typeof(matrix_kernel[1,1])}(undef,N1,N2)
    for j = 1:N2
        λⱼ = functionals2[j]
        v = [λⱼ(matrix_kernel[1,:]),λⱼ(matrix_kernel[2,:]),λⱼ(matrix_kernel[3,:])]
        for i = 1:N1
            λᵢ = functionals1[i]
            M[i,j] = λᵢ(v)
        end
    end
    return M 
end

function compile_kernel_array(M)
    @variables x₁ x₂ ϵ;
    N1 = size(M)[1]
    N2 = size(M)[2]
    P = Matrix{Function}(undef,N1,N2)
    for i = 1:N1
        for j = 1:N2
            #display(M[i,j])
            P[i,j] = eval(build_function(M[i,j], [x₁, x₂], ϵ))
        end
    end
    return P
end

function crete_block_point_tensors(p_list1,p_list2)
    N1 = length(p_list1)
    N2 = length(p_list2)
    M = Matrix{Array{Float64, 3}}(undef,N1,N2)
    for i in 1:N1
        for j in 1:N2
            M[i,j] = point_difference_tensor(p_list1[i],p_list2[j])
        end
    end

    return M
end
#point_difference_tensor(Internal_points,Internal_points)
function generate_block_matrices(function_array,tensor_array,param)
    n1,n2 = size(tensor_array)
    if size(tensor_array) != size(function_array)
        return ArgumentError("function array and tensor array size mismatch")
    end
    M = Matrix{Matrix}(undef,n1,n2)
    for i in 1:n1
        for j in 1:n2
            #display(function_array[i,j])
            #display(tensor_array[i,j])
            M[i,j] = apply(function_array[i,j], tensor_array[i,j], param)
        end
    end
    return M
end
function flatten(block_matrix)
    # flattens block matrices into usual matrices 
    n1,n2 = size(block_matrix)
    res = hcat(block_matrix[1,:]...)
    #println(size(res))
    for i in 2:n1
        row  = hcat(block_matrix[i,:]...)
        res = vcat(res,row)
        #println(size(row))
    end
    return res
end

""" Define matrix kernels """


