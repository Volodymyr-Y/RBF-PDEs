module SomeModule

export Δ ,∂₁,∂₂,∂ₜ,Φ_div,ΔΦ_div,Φ_curl,Φ,ΔΦ,Φ_normal,ΔΦ_normal


using Symbolics
using Latexify

const nu = 1.0
@variables ϵ r x₁ x₂ t;

#ϕ = 1//945 * ((ϵ*r)^5 +15*(ϵ*r)^3 + 105*(ϵ*r)^2 + 945*(ϵ*r)+ 945)* exp(-ϵ*r)
ϕ = exp(-r^2*ϵ^2)
ϕ = substitute(ϕ, r=>sqrt(x₁^2 + x₂^2)) 
#display(ϕ)  
Δ(exprs) = expand_derivatives((Differential(x₁)^2)(exprs) + (Differential(x₂)^2)(exprs))
∂₁(exprs) = expand_derivatives(Differential(x₁)(exprs))
∂₂(exprs) = expand_derivatives(Differential(x₂)(exprs))
∂ₜ(exprs) = expand_derivatives(Differential(t)(exprs))

Φ_div = ([-∂₂(∂₂(ϕ)) ∂₁(∂₂(ϕ)); ∂₁(∂₂(ϕ)) -∂₁(∂₁(ϕ))])
ΔΦ_div= Δ.([-∂₂(∂₂(ϕ)) ∂₁(∂₂(ϕ)); ∂₁(∂₂(ϕ)) -∂₁(∂₁(ϕ))])
Φ_curl = ([-∂₁(∂₁(ϕ)) -∂₁(∂₂(ϕ)); -∂₁(∂₂(ϕ)) -∂₂(∂₂(ϕ))])
Φ = [-Δ(ϕ) 0 ; 0 -Δ(ϕ)]
ΔΦ = Δ.([-Δ(ϕ) 0 ; 0 -Δ(ϕ)])
Φ_normal = [(ϕ) 0 ; 0 (ϕ)]
ΔΦ_normal = Δ.([(ϕ) 0 ; 0 (ϕ)])

Φ_div =eval(build_function(Φ_div, [x₁, x₂], ϵ)[1])
ΔΦ_div =eval(build_function(ΔΦ_div, [x₁, x₂], ϵ)[1])

Φ_curl =eval(build_function(Φ_curl, [x₁, x₂], ϵ)[1])

Φ =eval(build_function(Φ, [x₁, x₂], ϵ)[1])
ΔΦ =eval(build_function(ΔΦ, [x₁, x₂], ϵ)[1])

Φ_normal =eval(build_function(Φ_normal, [x₁, x₂], ϵ)[1])
ΔΦ_normal =eval(build_function(ΔΦ_normal, [x₁, x₂], ϵ)[1])

# reference solution 

true_u₁ = -x₂*π*sin(π*0.5*(x₁*x₁ + x₂*x₂))*sin(π*t)
true_u₂ =  x₁*π*sin(π*0.5*(x₁*x₁ + x₂*x₂))*sin(π*t)
#println("confirm divergence free property")
#display(∂₁(true_u₁)+∂₂(true_u₂))
true_∂ₜu₁ = ∂ₜ(true_u₁)
true_∂ₜu₂  = ∂ₜ(true_u₂)

true_p = sin(x₁-x₂+t)
true_∂₁p = ∂₁(true_p)
true_∂₂p = ∂₂(true_p)

f₁ = ∂ₜ(true_u₁) - nu*Δ(true_u₁) + true_∂₁p
f₂ = ∂ₜ(true_u₂) - nu*Δ(true_u₂) + true_∂₂p


display(true_∂ₜu₁)
display(true_∂ₜu₂)

f₁ = eval(build_function(f₁,x₁, x₂, t))
f₂ = eval(build_function(f₂,x₁, x₂, t))

true_u₁ = eval(build_function(true_u₁,x₁, x₂, t))
true_u₂ = eval(build_function(true_u₂,x₁, x₂, t))

#true_u₁ = eval(build_function(true_u₁,x₁, x₂, t))
#true_u₂ = eval(build_function(true_u₂,x₁, x₂, t))


true_p = eval(build_function(true_p,x₁, x₂, t))
true_∂₁p = eval(build_function(true_∂₁p,x₁, x₂, t))
true_∂₂p = eval(build_function(true_∂₂p,x₁, x₂, t))

true_∂ₜu₁ = eval(build_function(true_∂ₜu₁,x₁, x₂, t))
true_∂ₜu₂  = eval(build_function(true_∂ₜu₂,x₁, x₂, t))

end