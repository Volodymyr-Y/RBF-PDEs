include("RBFunctions.jl")
using Symbolics
using BenchmarkTools
function generate_points(N)
    dom,bound = generate_2D_equally_spaced_points(N)
    #display(bound)
    B_points = Array{BoundaryPoint{2}}(undef,size(bound)[2])
    D_points = Array{DomainPoint{2}}(undef,size(dom)[2])
    for i in 1:size(bound)[2]
        B_points[i] = BoundaryPoint{2}(bound[:,i],[0.0,0.0])
    end
    for i in 1:size(dom)[2]
        D_points[i] = DomainPoint{2}(dom[:,i])
    end
    return (D_points,B_points)
end

function generate_problem()
    Domain_set, Boundary_set = generate_points(5)
    top_points = []
    side_points = []
    for point in Boundary_set
        if point.pos[2] == 1.0
            append!(top_points,[point])
        else
            #println("wewe")
            append!(side_points,[point])
        end
    end
    #display(top_points)
    #display(side_points)
    BC_top = BoundaryConditions{Dirichlet}(top_points,(x,t) -> [1.0,0.0])
    BC_sides = BoundaryConditions{Dirichlet}(side_points,(x,t) -> [0.0,0.0])
    forcing(x,t) = [x[1],t]
    in_cond(x) = [0.0,0.0]
    tspan = (0.0,10.0)
    dt = 0.01
    prob = Problem( Domain_set,
                    Boundary_set,
                    [BC_top,BC_sides],
                    forcing,
                    in_cond,
                    tspan,
                    dt)
    return prob
end
function generate_functionals(a::BoundaryConditions{Dirichlet})
        λ3y(x) = x[1]
        λ4y(x) = x[2]
        λ3x(x) = x[1]
        λ4x(x) = x[2]
        return ([λ3x,λ4x],[λ3y,λ4y])
    end


function get_boundary(t,bc::BoundaryConditions{Dirichlet})
    N_p = lastindex(bc.point_set) # number of points in this boundary segmant
    res = zeros(N_p*2)
    for (i,point) in  enumerate(bc.point_set)
        res[i] = bc.g(point.pos,t)[1]
        res[i+N_p] = bc.g(point.pos,t)[2]
    end
   return res
end

function solve_Stokes(problem)
    parameter = 4
    # construct functionals 
    # construct RHS
    @variables ϵ r x₁ x₂ t Δt
    #const nu = 1.0
    #ϕ = 1//945 * ((ϵ*r)^5 +15*(ϵ*r)^3 + 105*(ϵ*r)^2 + 945*(ϵ*r)+ 945)* exp(-ϵ*r)
    ϕ = exp(-r^2*ϵ^2)
    ϕ = substitute(ϕ, r=>sqrt(x₁^2 + x₂^2)) 
    #display(ϕ)  
    Δ(exprs) = expand_derivatives((Differential(x₁)^2)(exprs) + (Differential(x₂)^2)(exprs))
    ∂₁(exprs) = expand_derivatives(Differential(x₁)(exprs))
    ∂₂(exprs) = expand_derivatives(Differential(x₂)(exprs))
    #∂ₜ(exprs) = expand_derivatives(Differential(t)(exprs))

    Φ_div = ([-∂₂(∂₂(ϕ)) ∂₁(∂₂(ϕ)) 0.0 ; ∂₁(∂₂(ϕ)) -∂₁(∂₁(ϕ)) 0.0; 0.0 0.0 ϕ])
    #display(Φ_div)
    
    λ1y(x) = x[1] - Δt*Δ(x[1]) -  Δt*∂₁(x[3]) # Stokes functionals applied to 2nd variable
    λ2y(x) = x[2] - Δt*Δ(x[2]) -  Δt*∂₂(x[3])

    λ1x(x) = x[1] - Δt*Δ(x[1]) +  Δt*∂₁(x[3])
    λ2x(x) = x[2] - Δt*Δ(x[2]) +  Δt*∂₂(x[3]) # Stokes functionals applied to 1st variable

    λu(x) = x[1] # primary_variable_evaluation_functionals
    λv(x) = x[2]
    λp(x) = x[3]
    second_variable_functionals = [λ1y,λ2y]
    first_variable_functionals = [λ1x,λ2x]
    Point_list_for_interpolation = []
    append!(Point_list_for_interpolation,[problem.domain_points,problem.domain_points])
    for bcond in problem.bc
        first_var_bc_functionals, second_var_bc_functionals = generate_functionals(bcond)
        append!(first_variable_functionals,first_var_bc_functionals)
        append!(second_variable_functionals,second_var_bc_functionals)
        append!(Point_list_for_interpolation,[bcond.point_set,bcond.point_set])
    end

    K = construct_kernel_array(Φ_div,first_variable_functionals,second_variable_functionals)
    K = substitute.(K, Δt=>problem.dt)
    K = compile_kernel_array(K)
    R = construct_kernel_array(Φ_div,[λu,λv,λp],second_variable_functionals)
    R = substitute.(R, Δt=>problem.dt)
    R = compile_kernel_array(R)
    #render(latexify(K[1,1]))
    

    #display(R)
    T1 = crete_block_point_tensors(Point_list_for_interpolation,Point_list_for_interpolation)
    T2 = crete_block_point_tensors([problem.domain_points,problem.domain_points,problem.domain_points],Point_list_for_interpolation)
    #display(typeof(T))
    A = generate_block_matrices(K,T1,parameter)
    A = flatten(A)
    B = generate_block_matrices(R,T2,parameter)
    B = flatten(B)
    # construct parts of RHS
    println(length(problem.domain_points))
    println(length(problem.boundary_points))
    println(length(vcat(Point_list_for_interpolation...)))
    N_i = length(problem.domain_points)
    N_b = length(problem.boundary_points)
    function f(t)
        f_array = zeros(2*N_i)
        suma = 0
        for i in 1:N_i
            f_array[i] = problem.forcing(problem.domain_points[i].pos,t)[1]
            f_array[i+N_i] = problem.forcing(problem.domain_points[i].pos,t)[2]
        end
        return f_array
    end
    
    function g(t)
        res = []
        for b_cond in problem.bc
            #println(b_cond.g(1,1))
            res = vcat(res,get_boundary(t,b_cond))
        end
        return res
    end
    
    g(110)
    #display(methods(K[1,1]))
    #FFF = (eval(build_function(Φ_div[1,1],[x₁,x₂],ϵ)))
    #typeof(FFF)
    #methods(FFF)
    #println(Base.@invokelatest FFF([1,1.0],1.0))
    #display(K[1,1]([1.2,1.2],1.9))
    return A,B,f,g
end

prob = generate_problem()
solve_Stokes(prob)

