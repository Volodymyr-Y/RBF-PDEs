
#using SymbolicNumericIntegration
using StaticArrays
using Distances
using LinearAlgebra 
using FastHalton
using SparseArrays
using NearestNeighbors
using Symbolics
using Latexify
import SparseArrays: sparse
using DoubleFloats
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

struct SparseArray2
    row_vec::Array{Int}
    col_vec::Array{Int}
    xvals::Array{Float64}
    yvals::Array{Float64}
    n_rows::Int
    n_columns::Int
end

struct SparseArray
    row_vec::Array{Int}
    col_vec::Array{Int}
    vals::Array{Float64}
    n_rows::Int
    n_columns::Int
end



""" Kernel and test function definitions"""

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

function wendland_C8(r::Real,ϵ)
    if r/ϵ >= 0.0 && r/ϵ <=1.0
        return (1-r*ϵ)^10 * (429*(r*ϵ)^4 + 450*(r*ϵ)^3 + 210*(r*ϵ)^2 + 50*(r*ϵ)+5)
    else 
        return 0.0
    end
end


"""functions to generate points """

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


""" functions for matrix creation"""

function point_difference_tensor(points1,points2) # reates NxMx2 tensor 
    l1 = size(points1)[2]
    l2 = size(points2)[2]
    A = [ (points1[k,i] - points2[k,j]) for i=1:l1, j=1:l2, k=1:2]
    A[A.==0] .= 1e-30
    return A
end

function sparse_point_difference_tensor(points1,points2,r_max)
    #build k-d tree
    small = 1e-100
    n = size(points1)[2] 
    m = size(points2)[2]

    A = SparseArray2([],[],[],[],n,m)
    if n<=m
        tree = KDTree(points2,Euclidean(),leafsize = 10)
        for j in 1:n
            point = points1[:,j]
            idxs = inrange(tree, point, r_max, true)
            N_neighbours = length(idxs)
            a1 = point[1,:] .- points2[1,idxs]
            a2 = point[2,:] .- points2[2,idxs]
            append!(A.row_vec,j*ones(Int64,N_neighbours))
            append!(A.col_vec,idxs)
            append!(A.xvals,replace!(a1,0.0 => small))
            append!(A.yvals,replace!(a2,0.0 => small))
        end
        return A
    else
            tree = KDTree(points1,Euclidean(),leafsize = 10)
            for j in 1:m
                point = points2[:,j]
                idxs = inrange(tree, point, r_max, true)
                N_neighbours = length(idxs)
                #println(points1[1,idxs])
                a1 = points1[1,idxs] .- point[1,:]
                a2 = points1[2,idxs] .- point[2,:]
                append!(A.col_vec,j*ones(Int64,N_neighbours))
                append!(A.row_vec,idxs)
                append!(A.xvals,replace!(a1,0.0 => small))
                append!(A.yvals,replace!(a2,0.0 => small))
            end
            return A
    end
    
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

function apply(func, sm::SparseArray2,param)
    Res = SparseArray(sm.row_vec,sm.col_vec,zeros(size(sm.xvals)),sm.n_rows,sm.n_columns)
    for i in 1:length(sm.xvals)
        Res.vals[i] = func([sm.xvals[i],sm.yvals[i]],param) 
    end
    return Res
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

function construct_kernel_array(matrix_kernel,functionals1,functionals2) # apply functionals to matrix kernel 
    N1 = length(functionals1)
    N2 = length(functionals2)
    M = Matrix{typeof(matrix_kernel[1,1])}(undef,N1,N2)
    
    m = size(matrix_kernel)[1]
    for j = 1:N2
        λⱼ = functionals2[j]
        v = [λⱼ(matrix_kernel[k,:]) for k in 1:m]
        
        #v = [λⱼ(matrix_kernel[1,:]),λⱼ(matrix_kernel[2,:]),λⱼ(matrix_kernel[3,:])]
        for i = 1:N1
            λᵢ = functionals1[i]
            M[i,j] = λᵢ(v)
        end
    end
    return M 
end

function construct_kernel_array(matrix_kernel::Num,functionals1,functionals2) # apply functionals to matrix kernel 
    N1 = length(functionals1)
    N2 = length(functionals2)
    M = Matrix{typeof(matrix_kernel[1,1])}(undef,N1,N2)
    
    for j = 1:N2
        λⱼ = functionals2[j]
        v = λⱼ(matrix_kernel) 
        
        #v = [λⱼ(matrix_kernel[1,:]),λⱼ(matrix_kernel[2,:]),λⱼ(matrix_kernel[3,:])]
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

function crete_block_point_tensors(p_list1::Vector{Matrix{Float64}},p_list2)
    N1 = length(p_list1)
    N2 = length(p_list2)
    M = Matrix{Array{Float64,3}}(undef,N1,N2)
    for i in 1:N1
        for j in 1:N2
            M[i,j] = point_difference_tensor(p_list1[i],p_list2[j])
        end
    end

    return M
end

function crete_block_point_tensors(p_list1::Vector{Matrix{Double64}},p_list2)
    N1 = length(p_list1)
    N2 = length(p_list2)
    M = Matrix{Array{Double64, 3}}(undef,N1,N2)
    for i in 1:N1
        for j in 1:N2
            M[i,j] = point_difference_tensor(p_list1[i],p_list2[j])
        end
    end

    return M
end

function sparse_block_point_tensors(p_list1,p_list2,r_max)
    N1 = length(p_list1)
    N2 = length(p_list2)
    M = Matrix{SparseArray2}(undef,N1,N2)
    for i in 1:N1
        for j in 1:N2
            M[i,j] = sparse_point_difference_tensor(p_list1[i],p_list2[j],r_max)
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
    M = Matrix{}(undef,n1,n2)
    for i in 1:n1
        for j in 1:n2
            #display(function_array[i,j])
            #display(tensor_array[i,j])
            M[i,j] = apply(function_array[i,j], tensor_array[i,j], param)
        end
    end
    return M
end
function generate_block_matrices(function_array,tensor_array::Matrix{SparseArray2},param)
    n1,n2 = size(tensor_array)
    if size(tensor_array) != size(function_array)
        return ArgumentError("function array and tensor array size mismatch")
    end
    M = Matrix{SparseArray}(undef,n1,n2)
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

function flatten(block_matrix::Matrix{SparseArray})
    # flattens block matrices into usual matrices 
    n1,n2 = size(block_matrix)
    row_track = 0
    col_track = 0
    res = SparseArray([],[],[],n1,n2)
    #println(size(res))
    for i in 1:n1
        for j in 1:n2
            #println(block_matrix[i,j].n_rows," ",block_matrix[i,j].n_columns)
            append!(res.col_vec,block_matrix[i,j].col_vec .+ col_track)
            append!(res.row_vec,block_matrix[i,j].row_vec .+ row_track)
            append!(res.vals,block_matrix[i,j].vals)
            col_track = col_track + block_matrix[i,j].n_columns
        end
        col_track = 0
        row_track = row_track + block_matrix[i,1].n_rows
    end
    return res
end


function sparse(M::SparseArray)
    return sparse(M.row_vec,M.col_vec,M.vals)
end


function PU_interpolation_matrix(Points1,Points2,PU_Points,f,ϵ,r_pu)
    #f in the function that acts on the distance tensor,ϵ is the shape parameter, r_pu is the partition radius
    # compute all the trees 
    PU_tree = KDTree(PU_Points,Euclidean(),leafsize = 3)
    Collocation_tree = KDTree(Points2,Euclidean(),leafsize = 3)
    Test_tree  = KDTree(Points1,Euclidean(),leafsize = 3)
    M = zeros(size(Points1)[2],size(Points2)[2]) # later be replaced with sparse matrix
    scaling_vector= zeros(size(Points1)[2])
    for i in 1:size(PU_Points)[2]
        coll_indx = inrange(Collocation_tree, PU_Points[:,i], r_pu, true)
        tst_indx = inrange(Test_tree, PU_Points[:,i], r_pu, true)
        Local_distance_tensor = point_difference_tensor(Points1[:,tst_indx],Points2[:,coll_indx])
        Local_interpolation_matrix = apply(f, Local_distance_tensor,ϵ)
        M[tst_indx,coll_indx] .+= Local_interpolation_matrix
        scaling_vector[tst_indx] .+= 1
        #println(cond(Local_interpolation_matrix))
    end
    #println(scaling_vector)
    return M .* (1 ./scaling_vector)
end

""" fUNCTIONS FOR POLYNOMIALS """

""" generates basis of div free polynomials R^2 -> R^2"""
function generate_2D2_div_free_poly_basis(m)
    lst = []
    for i in 0:m
        for j in 0:i
            p1 = x₁^j * x₂^(i-j)
            aa = -∂₁(p1)
            deg = Symbolics.degree(aa,x₂)
            aa = substitute(aa,x₂ => 1)
            p2 = (aa*x₂^(deg+1))/(deg+1)
            append!(lst,[[p1 , p2]])
        end
    end
    for i in 0:m
        append!(lst,[[0, x₁^i]])
    end
    return lst
end

function generate_2D1_poly_basis(m)
    lst = []
    for i in 0:m
        for j in 0:i
            p = 1.0*x₁^j * x₂^(i-j)
            append!(lst,[p])
        end
    end
    return lst
end


function apply_functionals_to_polynomials(func_lst,poly_lst)
    N_f = length(func_lst)
    N_p = length(poly_lst)
    A = Matrix{Num}(undef,N_f,N_p)
    for i in 1:N_f
        for j in 1:N_p
            #println("asa")
            #println(func_lst[i](poly_lst[j]))
            A[i,j] = func_lst[i](poly_lst[j])
        end
    end
    return A
end

function compile_polynomials(M)
    @variables x₁ x₂;
    N1 = size(M)[1]
    N2 = size(M)[2]
    P = Matrix{Function}(undef,N1,N2)
    for i = 1:N1
        for j = 1:N2
            #display(M[i,j])
            P[i,j] = eval(build_function(M[i,j], x₁, x₂))
        end
    end
    return P
end

function generate_P_matrix(P_list,F_matrix)
    if length(P_list) != size(F_matrix)[1]
        println("troubles")
    end
    L = size(hcat(P_list...))[2]
    N_f,N_poly = size(F_matrix)
    M = Matrix{}(undef,N_f,N_poly)
    for i in 1:N_f
        for j in 1:N_poly
            #display(P_list[i][1,:])
            M[i,j] =  reshape(F_matrix[i,j].(P_list[i][1,:],P_list[i][2,:]),(size(P_list[i])[2],1))
        end
    end

    return flatten(M)
end


function create_peconditioner_diagonal(F_A,F_PA)
    @variables r x₁ x₂
    println(typeof(r))
    N_func, N_poly = size(F_PA)
    F_PA_copy = deepcopy(F_PA)
    F_A_copy = deepcopy(F_A)

    F_A_copy = substitute.(F_A_copy, sqrt(x₁^2+x₂^2) => r)
    F_A_copy = substitute.(F_A_copy, x₁ => r)
    F_A_copy = substitute.(F_A_copy, x₂ => r)
    
    F_PA_copy = substitute.(F_PA_copy, sqrt(x₁^2+x₂^2) => r)
    F_PA_copy = substitute.(F_PA_copy, x₁ => r)
    F_PA_copy = substitute.(F_PA_copy, x₂ => r)
    
    display(F_A_copy)
    deg_array = zeros(N_func)
    for i in 1:N_func
        deg = Symbolics.degree(F_A_copy[i,i],r)
        deg_array[i] = deg/2
    end

    deg_array_poly = zeros(N_poly)
    for i in 1:N_poly
        for j in 1:N_func
            if !isequal(F_PA_copy[j,i],0)
                deg = (Symbolics.degree(F_PA_copy[j,i],r))
                deg_array_poly[i] = deg - deg_array[j]
                break
            end
        end

        """
        if !isequal(F_PA_copy[1,i],0)
            deg = (Symbolics.degree(F_PA_copy[1,i],r))
            deg_array_poly[i] = deg - deg_array[1]
        elseif !isequal(F_PA_copy[2,i],0)
            deg = (Symbolics.degree(F_PA_copy[2,i],r))
            deg_array_poly[i] = deg - deg_array[2]
        elseif !isequal(F_PA_copy[3,i],0)
            deg = (Symbolics.degree(F_PA_copy[3,i],r))
            deg_array_poly[i] = deg - deg_array[3]
        elseif !isequal(F_PA_copy[4,i],0)
            deg = (Symbolics.degree(F_PA_copy[4,i],r))
            deg_array_poly[i] = deg - deg_array[4]
        end
        """
    end
    #println(deg_array)
    return deg_array ,deg_array_poly
end