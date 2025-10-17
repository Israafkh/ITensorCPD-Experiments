# This considers the exact solution plus some perturbation and determineing how 
# well our code can find a good update (or exact update)

using ITensors, ITensorCPD
using Revise
using ITensorCPD: compute_krp, matricize_tensor, solve_ls_problem, row_norm, post_solve, had_contract

# Fit = || T - \hat{T} || / || T ||
# sqrt(T^2 - 2 * T \hat{T} + \hat{T}^2)
function check_fit(target, factors, cprank, λ, fact)
    inner_prod = (had_contract([target, dag.(factors)...], cprank) * dag(λ))[]
    partial_gram = [fact * dag(prime(fact; tags=tags(cprank))) for fact in factors];
    fact_square = ITensorCPD.norm_factors(partial_gram, λ)
    normResidual =
        sqrt(abs(als.check.ref_norm * als.check.ref_norm + fact_square - 2 * abs(inner_prod)))
    fit = 1.0 - normResidual / norm(als.check.ref_norm)
    println("CPD rank\tMode\tCPD Accuracy")
    println("$(dim(cprank))\t\t$(fact)\t$(fit)")
    return fit
end

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

    fit = check_fit(als.target, factors, cprank, λ, fact)
    return ITensorCPD.CPD{ITensor}(factors, λ), als, fit
end

i, j, k  = Index.((100,110,120));

T = ITensor(Float64, i,j,k);
cpd_exact = ITensorCPD.random_CPD(T, 201);
T = ITensorCPD.reconstruct(cpd_exact);

check=ITensorCPD.FitCheck(0, 100, norm(T));

#### exact solution
alg = ITensorCPD.direct()
als = ITensorCPD.compute_als(T, cpd_exact; alg, check);
# ITensorCPD.optimize(cpd_exact, als; verbose=true);

cpd_exact, als = single_solve(als, cpd_exact, 1);
cpd_exact, als = single_solve(als, cpd_exact, 2);
cpd_exact, als = single_solve(als, cpd_exact, 3);


#### SEQRCS check
alpha = .01
LevFits = Vector{Vector{Float64}}()
for samps in [210]
    fits = Vector{Float64}()
    for blocks in 1:8
        cpd_pert = deepcopy(cpd_exact)
        #alg = ITensorCPD.SEQRCSPivProjected(1, samps, (1,2,3), (10,10,10))
        alg = ITensorCPD.BlockLevScoreSampled(samps, blocks)

        
        array(cpd_pert.factors[2]) .+= alpha .* randn(size(cpd_pert.factors[2]))
        cpd_pert.factors[2], _ = row_norm(cpd_pert[2], ind(cpd_pert, 2))
        @time begin
            als = ITensorCPD.compute_als(T, cpd_pert; alg, check);
            cpd_pert, als, fit = single_solve(als, cpd_pert, 1);
        end
        push!(fits, fit)
    end
    push!(LevFits, fits)
end

using Plots
plot(LevFits[1], marker=:circle)
plot!(LevFits[2], marker=:circle)
plot!(LevFits[3], marker=:circle)
plot!(LevFits[4], marker=:circle)
plot!(LevFits[6], marker=:circle)

QRFits = Vector{Vector{Float64}}()
for samps in [15,45, 60, 100, 200, 300]
    fits = Vector{Float64}()
    
    cpd_pert = deepcopy(cpd_exact)
    alg = ITensorCPD.SEQRCSPivProjected(1, samps, (1,2,3), (10,10,10))

    
    array(cpd_pert.factors[2]) .+= alpha .* randn(size(cpd_pert.factors[2]))
    cpd_pert.factors[2], _ = row_norm(cpd_pert[2], ind(cpd_pert, 2))
    @time begin
        als = ITensorCPD.compute_als(T, cpd_pert; alg, check);
        cpd_pert, als, fit = single_solve(als, cpd_pert, 1);
    end
    for blocks in 1:8
        push!(fits, fit)
    end
    GC.gc()
    push!(QRFits, fits)
end

plot!(QRFits[1], marker=:square)
plot!(QRFits[2], marker=:square)
plot!(QRFits[3], marker=:square)
plot!(QRFits[4], marker=:square)
plot!(QRFits[5], marker=:square)
plot!(QRFits[6], marker=:square)