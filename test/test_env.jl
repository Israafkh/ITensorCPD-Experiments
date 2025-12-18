using Pkg
## I don't have this path to add. you could add
# Pkg.develop(url="github.com/kmp5VT/ITensorCPD.jl")
## and tell people to uncomment to add ITensorCPD
#Pkg.develop(path = "$(@__DIR__)/../../ITensorCPD.jl")

using ITensorCPD
using ITensors
using HDF5
using Plots
# using ITensorNetworks
using Profile


using Random
using ITensorCPD: had_contract

# Fit = 1 - || T - \hat{T} || / || T ||
# sqrt(T^2 - 2 * T \hat{T} + \hat{T}^2)
## Update this to give the error in the LS 
function check_fit(als, cpd)
    return check_fit(als, cpd.factors, ITensorCPD.cp_rank(cpd), cpd.λ, 1)
end

function check_fit(als, factors, cprank, λ, fact)
    target = als.target
    inner_prod = (had_contract([target, dag.(factors)...], cprank) * dag(λ))[]
    partial_gram = [fact * dag(prime(fact; tags=tags(cprank))) for fact in factors];
    fact_square = ITensorCPD.norm_factors(partial_gram, λ)
    ref_norm = sqrt(sum(target .^2))
    normResidual =
        sqrt(abs(ref_norm * ref_norm + fact_square - 2 * abs(inner_prod)))
    fit = 1.0 - normResidual / norm(ref_norm)
    println("CPD rank\tMode\tCPD Accuracy")
    println("$(dim(cprank))\t\t$(fact)\t$(fit)")
    return fit
end

# f(A) = || T - [[A, B, C]] ||^2
# f(A) = sqrt(T^2 - 2 * T * [[A, B, C]] - [[A, B, C]]^2)
# df/dA = -T (C \odot B) + A * (C \odot B)^2
function check_grad(target, cpd, grammian, fact, updated_factor)

    factors = cpd.factors
    cpr = ITensorCPD.cp_rank(cpd)
    mttkrp = ITensorCPD.had_contract([target, factors[1:end .!= fact]...], cpr)
    g = itensor(ones(dim(grammian[1])), inds(grammian[1]))

    for x in grammian[1:end .!= fact]
        g = ITensors.hadamard_product!(g, g, x)
    end

    loss_function = norm(-mttkrp + noprime(updated_factor * g))
    println("CPD rank\tMode\tNorm of Gradient")
    println("$(dim(cpr))\t\t$(fact)\t$(loss_function)")
    return loss_function
end

# f(A) = || T - [[A, B, C]]||^2
function check_loss(target, cpd, grammian, fact, updated_factor)
    factors = cpd.factors
    cpr = ITensorCPD.cp_rank(cpd)
    n2 = (target * target)[]
    mttkrp = ITensorCPD.had_contract([target, factors[1:end .!= fact]...], cpr)
    inner = (mttkrp * updated_factor)[]

    g = itensor(ones(dim(grammian[1])), inds(grammian[1]))
    for x in grammian[1:end .!= fact]
        g = ITensors.hadamard_product!(g, g, x)
    end

    cpd2 = (g * (updated_factor * prime(updated_factor, tags=tags(cpr))))[]


    loss_function = n2 - 2 * inner + cpd2
    println("CPD rank\tMode\tLoss")
    println("$(dim(cpr))\t\t$(fact)\t$(loss_function)")
    return loss_function
end

using ITensorCPD: compute_krp, matricize_tensor, solve_ls_problem, row_norm, post_solve
function single_solve(alsRef, cpd, fact)
    cprank = ITensorCPD.cp_rank(cpd)
    target_ind = ind(cpd, fact)
    factors = deepcopy(cpd.factors);
    als = deepcopy(alsRef)

    gram = [i * prime(i, tags=tags(cprank)) for i in cpd.factors]
    if haskey(alsRef.additional_items, :part_grammian)
        als.additional_items[:part_grammian] .= gram
    end

    krp = compute_krp(als.mttkrp_alg, als, factors, cpd, cprank, fact)
    mtkrp = matricize_tensor(als.mttkrp_alg, als, factors, cpd, cprank, fact)
    solution = solve_ls_problem(als.mttkrp_alg, krp, mtkrp, cprank)
    
    factors[fact], λ = row_norm(solution, target_ind)
    post_solve(als.mttkrp_alg, als, factors, λ, cpd, cprank, fact)

    #fit = check_fit(als, factors, cprank, λ, fact)
    fit = check_fit(alsRef, factors, cprank, λ, fact)
    # fit = check_grad(alsRef.target, cpd, gram, fact, solution)
    fit = check_loss(alsRef.target, cpd, gram, fact, solution)
    return ITensorCPD.CPD{ITensor}(factors, λ), als, fit
end

include("colinearity_tensor_generator.jl")

function construct_large_lev_score_cpd(is, rank, nbad_points = 4; rng=RandomDevice())

    rank = rank isa Index ? rank : Index(rank, "CPRank")
    drank = dim(rank)
    nd = length(is)
    factors = Vector{ITensor}()
    for (m,n) in zip(is, 1:nd)
        nbad_point = length(nbad_points) == 1 ? nbad_points : nbad_points[n]
        U = randn(dim(m), drank)
        U[:, 1:nbad_point] .= 0.0
        U[1:nbad_point, :] .= 0.0

        for i in 1:nbad_point
            U[i,i] = 1.0
        end
        U = copy(qr(U).Q)
        if nbad_point == 2
            v = [(1:dim(m))...]
            new_pos = rand(rng, 1:(dim(m)÷3))
            v[new_pos] = 1
            v[1] = new_pos

            new_pos = rand(rng, 2 * (dim(m)÷3)+1:dim(m))
            v[new_pos] = 2
            v[2] = new_pos
            U = U[v,:]
        else
            U = U[randperm(dim(m)), :]
        end
        _, _, V = svd(randn(dim(m),drank))

        ix = m isa Index ? m : Index(m)
        push!(factors, itensor(U[:, 1:drank] * V', ix, rank))
    end

    return ITensorCPD.CPD{ITensor}(factors, itensor(randn(drank), rank))
end

# function cp_score(cp1, cp2)
#     nums = [(x * y)[] / (norm(x) * norm(y) )  for (x,y) in zip(cp1, cp2)]
# end

function compute_lev_score(A::Matrix)
    cols = size(A)[1]
    rows = size(A)[2]
    q, _ = qr(A)
    q = Matrix(q)
    q .*= q
    return [sum(q[i, 1:rows]) for i in 1:cols]
end