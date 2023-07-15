using Symbolics
using Latexify
using BenchmarkTools
include("RBFunctions.jl")
using Plots
using LinearAlgebra
using IterativeSolvers
using NearestNeighbors
using DoubleFloats
using Profile

function compile_kernel_array_NS(M,name)
    @variables x₁ x₂ ϵ u1 v1 u2 v2;
    N1 = size(M)[1]
    N2 = size(M)[2]
    #P = Matrix{Expr}(undef,N1,N2)
    for i = 1:N1
        for j = 1:N2
            #display(M[i,j])
            sym =Symbol(name,i,j)
            fff = build_function(M[i,j], x₁, x₂, u1, v1, u2, v2,ϵ)
            eval(Expr(:(=),sym,fff))
        end
    end
end
function compile_polynomials_NS(M,name)
    @variables x₁ x₂ u1 v1;
    N1 = size(M)[1]
    N2 = size(M)[2]
    P = Matrix{Expr}(undef,N1,N2)
    for i = 1:N1
        for j = 1:N2
            #display(M[i,j])
            sym =Symbol(name,i,j)
            fff = build_function(M[i,j], x₁, x₂,u1,v1)
            #P[i,j] = fff
            eval(Expr(:(=),sym,fff))
            
        end
        #tup_sym = Symbol(name,i)
        #eval(Expr(:(=),tup_sym,Expr(:tuple,P[i,:]...)))
    end
end

function apply_NS!(M,p1,p2,u1,v1,u2,v2,f,param)
    d1 = size(p1)[2]
    d2 = size(p2)[2]
    #M = zeros(d1,d2)
    if (d1 != length(u1)) || (d1 !=length(v1)) || (d2 != length(u2)) || (d2 !=length(v2))
        println(d1," ",u1)
        println(d1," ",v1)
        println(d2," ",u2)
        println(d2," ",v1)
        error("velocity and point arrays have  different sizes ")
    end
    a1 = similar(p1[:,1])
    a2 = similar(p1[:,1])
    tol = 1e-60
    for i in 1:d1
        for j in 1:d2
            a1 = p1[1,i] - p2[1,j]
            a2 = p1[2,i] - p2[2,j]
            a1 += tol
            a2 += tol
            #println(a)
            M[i,j] = f(a1,a2,u1[i], v1[i], u2[j], v2[j],param)
            #M[i,j] = f(1.2,1.2,1.2, 1.2, 1.2, 1.2,1.2)
        end
    end
    #return M
end

function apply_P_NS!(M,p,u,v,fp_array...)
    d1 = size(p)[2]
    d2 = length(fp_array)
    if (d1 != length(u)) || (d1 !=length(v))
        error("velocity and point arrays have  different sizes ")
    end
    println("inside apply poly function:")
    println("d1,d2:",d1," ",d2)
    display(p)
    #display(fp_array)
    for i in 1:d1
        for j in 1:d2
            #M[i,j] = F_PA11(p[1,i],p[2,i],u[i],v[i])
            M[i,j] = fp_array[j](p[1,i],p[2,i],u[i],v[i])
        end
    end
    #return res
end


function plot_arrow!(x1,y1,x2,y2)
    color = :blue
    arr_width = 0.2
    vx = x2-x1
    vy = y2 -y1
    l_arr = sqrt(vx*vx+vy*vy)
    pqx= x1+3*(x2-x1)/4
    pqy= y1+3*(y2-y1)/4 # quarterpoint

    vpx = 1
    vpy = -vx/vy
    lp = sqrt(vpx*vpx + vpy*vpy)
    vpx = vpx/lp
    vpy = vpy/lp 

    p1ax = pqx + vpx*l_arr*arr_width
    p1ay = pqy + vpy*l_arr*arr_width

    p2ax = pqx - vpx*l_arr*arr_width
    p2ay = pqy - vpy*l_arr*arr_width

    plot!([x1,x2],[y1,y2],c=color,label = false) # line
    trian = Shape([(p1ax, p1ay),(p2ax,p2ay), (x2,y2), (p1ax, p1ay)])
    plot!(trian, c = color,label = false,linewidth = 0)
end

function visualize_vec_field(Points,u,v)
    scale = 0.05
    vel = sqrt.(u.^2 .+ v .^2)
    mx = maximum(vel)
    #print(size(Points))
    plot()
    for i in 1:size(Points)[2]
        px,py = Points[:,i]
        #print(px,py)
        #plot!([px,px+scale*u[i]/mx],[py,py+scale*v[i]/mx],label = false,arrow = arrow(),)
        plot_arrow!(px,py,px+scale*u[i]/mx,py+scale*v[i]/mx)
    end
    plot!()
end


function fill_boundary!(A,B,i,I_idx,B_idx,I_points,B_points,U,N_poly,param)
    N_points = size(I_points)[2]
    
    N_I = length(I_idx)+1
    N_B = length(B_idx)
    N_L = N_I-1
    N_IL = 2N_I+ 2N_L
    N_ILB = N_IL + 2N_B
    N_tot = N_poly+N_ILB

    zero_u = zeros(N_B)
    N_L = N_I-1
    u_loc_full =  @view U[[i,I_idx...]]
    v_loc_full =  @view U[[i+N_I,(I_idx .+ N_I)...]]
    u_loc      =  @view U[I_idx]
    v_loc      =  @view U[I_idx .+ N_I]
    u_eval = U[i]
    v_eval = U[i+N_points]
    Full_L_I_points =  I_points[:,[i,I_idx...]] .- I_points[:,i:i]
    L_I_points =  I_points[:,I_idx] .- I_points[:,i:i]
    #println(I_idx)
    #display(Full_L_I_points)
    L_B_points =  B_points[:,B_idx] .- I_points[:,i:i]

    L_point_list = (Full_L_I_points,Full_L_I_points,L_I_points,L_I_points,L_B_points,L_B_points)
    u_list = (u_loc_full,u_loc_full,u_loc,u_loc,zero_u,zero_u)
    v_list = (v_loc_full,v_loc_full,v_loc,v_loc,zero_u,zero_u)
    range = (1:N_I , N_I+1:2N_I , 2N_I+1:2N_I+N_L , 2N_I+N_L+1:N_IL, N_IL+1:N_IL+N_B, N_IL+N_B+1:N_ILB)
    #plo = Plots.plot()
    #scatter!(L_I_points[1,:],L_I_points[2,:],label="domain",aspect_ratio = :equal)
    #scatter!(Full_L_I_points[1,:],Full_L_I_points[2,:],label="domain",aspect_ratio = :equal,markersize=1)
    #scatter!(L_B_points[1,:],L_B_points[2,:],label="boundary")
    #display(plo)
    
    
    for j in 1:6
        for i in 1:j
            sym = Symbol(:F_A,i,j)
            #eval(:(func = $sym))
            apply_NS!(view(A,range[i],range[j]),L_point_list[i],L_point_list[j],u_list[i],v_list[i],u_list[j],v_list[j],eval(sym),param)
        end
    end

    for i in 1:6
        sym1 = Symbol(:F_B,1,i)
        sym2 = Symbol(:F_B,2,i)
        apply_NS!(view(B,1:1,range[i]),[0.0 0.0]',L_point_list[i],[u_eval],[v_eval],u_list[i],v_list[i],eval(sym1),param)
        apply_NS!(view(B,2:2,range[i]),[0.0 0.0]',L_point_list[i],[u_eval],[v_eval],u_list[i],v_list[i],eval(sym2),param)
    end

    for i in 1:N_poly
        f1 = eval(Symbol(:F_PB,1,i))
        f2 = eval(Symbol(:F_PB,2,i))
        B[1,N_ILB+i] = f1(0.0,0.0,u_eval,v_eval)
        B[2,N_ILB+i] = f2(0.0,0.0,u_eval,v_eval)
    end

    for j in 1:N_poly
        for i in 1:6
            for k in 1:length(range[i])
                idx = range[i][k]
                f = eval(Symbol(:F_PA,i,j))
                p =  L_point_list[i][:,k]
                u = u_list[i][k]
                v = v_list[i][k]
                A[idx,N_ILB+j] = f(p[1],p[2],u,v)
                #PP[idx,j] = f(p[1],p[2],u,v)
            end
        end
        #print(" ",j)
    end

end

function fill_interiror!(A,B,i,I_idx,I_points,U,N_poly,param)
    N_points = size(I_points)[2]
    
    N_I = length(I_idx)+1
    N_L = N_I-1
    N_IL = 2N_I+ 2N_L
    N_tot = N_poly+N_IL
    N_L = N_I-1
    u_loc_full =  @view U[[i,I_idx...]]
    v_loc_full =  @view U[[i+N_I,(I_idx .+ N_I)...]]
    u_loc      =  @view U[I_idx]
    v_loc      =  @view U[I_idx .+ N_I]
    u_eval = U[i]
    v_eval = U[i+N_points]
    Full_L_I_points =  I_points[:,[i,I_idx...]] .- I_points[:,i:i]
    L_I_points =  I_points[:,I_idx] .- I_points[:,i:i]

    L_point_list = (Full_L_I_points,Full_L_I_points,L_I_points,L_I_points)
    u_list = (u_loc_full,u_loc_full,u_loc,u_loc)
    v_list = (v_loc_full,v_loc_full,v_loc,v_loc)
    range = (1:N_I , N_I+1:2N_I , 2N_I+1:2N_I+N_L , 2N_I+N_L+1:N_IL)
    
    
    for j in 1:4
        for i in 1:j
            sym = Symbol(:F_A,i,j)
            #eval(:(func = $sym))
            apply_NS!(view(A,range[i],range[j]),L_point_list[i],L_point_list[j],u_list[i],v_list[i],u_list[j],v_list[j],eval(sym),param)
        end
    end

    for i in 1:4
        sym1 = Symbol(:F_B,1,i)
        sym2 = Symbol(:F_B,2,i)
        apply_NS!(view(B,1:1,range[i]),[0.0 0.0]',L_point_list[i],[u_eval],[v_eval],u_list[i],v_list[i],eval(sym1),param)
        apply_NS!(view(B,2:2,range[i]),[0.0 0.0]',L_point_list[i],[u_eval],[v_eval],u_list[i],v_list[i],eval(sym2),param)
    end

    for i in 1:N_poly
        f1 = eval(Symbol(:F_PB,1,i))
        f2 = eval(Symbol(:F_PB,2,i))
        B[1,N_IL+i] = f1(0.0,0.0,u_eval,v_eval)
        B[2,N_IL+i] = f2(0.0,0.0,u_eval,v_eval)
    end

    for j in 1:N_poly
        for i in 1:4
            for k in 1:length(range[i])
                idx = range[i][k]
                f = eval(Symbol(:F_PA,i,j))
                p =  L_point_list[i][:,k]
                u = u_list[i][k]
                v = v_list[i][k]
                A[idx,N_IL+j] = f(p[1],p[2],u,v)
                #PP[idx,j] = f(p[1],p[2],u,v)
            end
        end
        #print(" ",j)
    end


end

function solve_linearized_NS(neighbours,N_poly,I_points,B_points,U,f1,f2,gu,gv,param,k)
    N_I = size(I_points)[2]
    N_B = size(B_points)[2]
    N_poly = size(F_PA)[2]
    
    #display(neighbours)
    G = zeros(2*N_I,2*N_I) # global sprse matrix 
    Global_RHS = vcat(f1.(I_points[1,:],I_points[2,:]),f2.(I_points[1,:],I_points[2,:]))    
    cond_num_array = zeros(N_I)
    A = zeros(6*k+N_poly,6*k+N_poly) # prealocate matrix of bigger size 
    B = zeros(2,6*k+N_poly) # prealocate matrix of bigger size 
    C = zeros(2,6*k+N_poly) # prealocate matrix of bigger size 
    L_RHS = zeros(6*k+N_poly) # prealocate matrix of bigger size 
    for i in 1:N_I
        I_idx = neighbours[i][findall(x -> x<=N_I , neighbours[i])][2:end]
        #println(neighbours[i][findall(x -> x<=N_I , neighbours[i])])
        B_idx = neighbours[i][findall(x -> x>N_I , neighbours[i])] .-N_I
        N_I_local = length(I_idx)+1
        N_B_local = length(B_idx)
        N_total_local = 4N_I_local-2 + 2N_B_local + N_poly
        L_RHS[1:2*N_I_local-2] .= vcat(f1.(I_points[1,I_idx],I_points[2,I_idx]),f2.(I_points[1,I_idx],I_points[2,I_idx]))
        #println(I_idx)
        if isempty(B_idx)
            # no bc 
            fill_interiror!(A,B,i,I_idx,I_points,U,N_poly,param)
            status = "interiror"

        else
            # present bc
            fill_boundary!(A,B,i,I_idx,B_idx,I_points,B_points,U,N_poly,param)
            L_RHS[2*N_I_local-2+1:2*N_I_local-2+2N_B_local] = 
            vcat(gu.(B_points[1,B_idx],B_points[2,B_idx]),gv.(B_points[1,B_idx],B_points[2,B_idx]))
            #println("boundary: ",i)
            status = "boundary"

        end
        #A = Symmetric(A,:U)
        A_loc = @view Symmetric(A,:U)[1:N_total_local,1:N_total_local]
        B_loc = @view B[:,1:N_total_local]
        #display(A_loc)
        #display(B_loc)
        #println(cond(A_loc))
        cond_num_array[i] = cond(A_loc)
        C[:,1:N_total_local] .= B_loc*inv(A_loc)
        G[[i,N_I+i],vcat([i],I_idx,[i+N_I], I_idx .+ N_I)] .= C[:,1:2*N_I_local]
        Global_RHS[[i,N_I+i]] .+= -(C[:,1+2*N_I_local:N_total_local] 
        * L_RHS[1:N_total_local-2N_I_local]) 
        #println(N_total_local)
        """
        if i == 314
            println(status)
            println(cond(Symmetric(A,:U)[1:N_total_local-N_poly,1:N_total_local-N_poly]))
            println("cond A: ",cond(A_loc))
            println(L_RHS)
            display(A_loc[1:N_total_local-N_poly,N_total_local-N_poly+1:end])
            display(A_loc[N_total_local-N_poly+1:end,N_total_local-N_poly+1:end])

        end
        """
        fill!(A,0.0)
        fill!(B,0.0)
        fill!(C,0.0)
        fill!(L_RHS,0.0) 

    end
    #display(cond(G))
    G_s = sparse(G)
    sol = G_s\Global_RHS
    #println("max local cond number: ",maximum(cond_num_array))
    #println("min local cond number: ",minimum(cond_num_array))

    return sol
end
