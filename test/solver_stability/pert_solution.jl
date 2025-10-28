include("$(@__DIR__)/../test_env.jl")
using ITensorCPD: compute_krp, matricize_tensor, solve_ls_problem, row_norm, post_solve, had_contract

## This looks at the stability of the SEQRCS method for near
## solution problems. We take the true solution (a random CPD) and 
## Perturb one of the factor matrices from its original position by 
## certain levels of noise. Then we optimize the method and see if 
## we can recover the true solution

function single_solve(alsRef, cpd, fact)
    cprank = ITensorCPD.cp_rank(cpd)
    target_ind = ind(cpd, fact)
    factors = deepcopy(cpd.factors);
    als = deepcopy(alsRef)

    krp = compute_krp(als.mttkrp_alg, als, factors, cpd, cprank, fact)
    mtkrp = matricize_tensor(als.mttkrp_alg, als, factors, cpd, cprank, fact)
    solution = solve_ls_problem(als.mttkrp_alg, krp, mtkrp, cprank)
    
    factors[fact], λ = row_norm(solution, target_ind)
    post_solve(als.mttkrp_alg, als, factors, λ, cpd, cprank, fact)

    fit = check_fit(als, factors, cprank, λ, fact)
    gram = [i * prime(i, tags=tags(cprank)) for i in cpd.factors]
    fit = check_loss(alsRef.target, cpd, gram, fact, solution)
    return ITensorCPD.CPD{ITensor}(factors, λ), als, fit
end

## We should try for higher order tensors too
is  = Index.((10,20,30,40));

T, cpd_exact = Colinearity_Tensor(20, 3, (100,100,100), 0.8, nothing, Float64);

check=ITensorCPD.FitCheck(0, 100, norm(T));

#### exact solution
alsNormal = ITensorCPD.compute_als(T, cpd_exact; alg=ITensorCPD.direct(), check);
alsTrue = ITensorCPD.compute_als(T, cpd_exact; alg=ITensorCPD.InvKRP(), check);

#### SEQRCS check
alpha = .01
mode_opt = 1
LevFits = Vector{Vector{Float64}}()
for samps in [50,100,200,500,2000]
   fits = Vector{Float64}()
#    for blocks in 1:50
        cpd_pert = deepcopy(cpd_exact)
        alg = ITensorCPD.BlockLevScoreSampled(samps, 1)

        
        array(cpd_pert.factors[2]) .+= alpha .* randn(size(cpd_pert.factors[2]))
        cpd_pert.factors[2], _ = row_norm(cpd_pert[2], ind(cpd_pert, 2))
        alsBlockLev = ITensorCPD.compute_als(T, cpd_pert; alg, check);
        cpd_pert, als, fit = single_solve(alsBlockLev, cpd_pert, mode_opt);
        push!(fits, fit)
    # end
    push!(LevFits, fits)
end

using Plots
plot(LevFits[2], marker=:circle)
plot!(LevFits[3], marker=:circle)
plot!(LevFits[4], marker=:circle)
plot!(LevFits[5], marker=:circle)


QRFits = Vector{Vector{Float64}}()
for samps in [50,100,200,500,2000]
   fits = Vector{Float64}()
    cpd_pert = deepcopy(cpd_exact)
    array(cpd_pert.factors[2]) .+= alpha .* randn(size(cpd_pert.factors[2]))
    cpd_pert.factors[2], _ = row_norm(cpd_pert[2], ind(cpd_pert, 2))
    _,_, fit = single_solve(alsTrue, cpd_pert, mode_opt);
    _,_, fit = single_solve(alsNormal, cpd_pert, mode_opt);
  #  for blocks in 1:50
        alg = ITensorCPD.SEQRCSPivProjected(1, samps, (1,2,3,4), (10,10,10,10))

        alsSEQRCS = ITensorCPD.compute_als(T, cpd_pert; alg, check);
        _,_, fit = single_solve(alsSEQRCS, cpd_pert, mode_opt);
        
        push!(fits, fit)
    # end
    push!(QRFits, fits)
end

using Plots
plot(QRFits[2], marker=:square)
plot!(QRFits[3], marker=:square)
plot!(QRFits[4], marker=:square)
plot!(QRFits[5], marker=:square)
