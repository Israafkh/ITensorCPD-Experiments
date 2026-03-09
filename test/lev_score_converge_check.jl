include("$(@__DIR__)/test_env.jl")
using Plots
using Random
using LinearAlgebra: ColumnNorm, diagm, svdvals
using LinearAlgebra

function proj_angle_compute(cpd, inds,fact)
    cpr = ITensorCPD.cp_rank(cpd)
    krp = ITensorCPD.had_contract([cpd[i] for i in inds], cpr)
    fused = Index(prod([dim(ind(cpd, i)) for i in inds]))
    krp = ITensorCPD.itensor(array(krp), fused, cpr)
    krp_matrix = Matrix(krp,fused, cpr)
    
    Q,_,_=svd(krp_matrix)
    Q1= Matrix(Q)
    Q2 = Q_exact_vect[fact]
    σ = svdvals(Q1' * Q2)
    # sqrt(1-(minimum(σ)^2))
    # sqrt(100-(sum(σ.^2)))/sqrt(100)
    return sqrt(1-(minimum(σ)^2))
end


i,j,k = 90,90,90
samps = 2000
bad = 10
r = 50
cpd_exact = ITensorCPD.random_CPD(zeros(i,j,k), r);
for (x, is) in zip(1:3, [i,j,k])
    array(cpd_exact[x])[1:bad, :] .*= 20;
    array(cpd_exact[x]) .= array(cpd_exact[x])[randperm(is), :]
    tmp, lam = ITensorCPD.row_norm(cpd_exact[x], ind(cpd_exact[x], 1))
    array(cpd_exact[x]) .= tmp
    cpd_exact[] .= lam
end

 r = ITensorCPD.cp_rank(cpd_exact);

# seed = Xoshiro(123)
# rng = RandomDevice()
# cpd_exact = construct_large_lev_score_cpd((i,j,k), r, bad;rng);
# r = ITensorCPD.cp_rank(cpd_exact)

T = had_contract(cpd_exact[1], cpd_exact[2], r) * had_contract(cpd_exact[3], cpd_exact[], r)

### This section computes the leverage score of T and compares to KRP
krp = ITensorCPD.had_contract(cpd_exact[2], cpd_exact[3], r)
levsKRP = (compute_lev_score(reshape(array(krp), (90 * 90, dim(r)))))
levsT = compute_lev_score(reshape(array(T, inds(T)[[2,3,1]]), (90 * 90, 90)), dim(r))

plot(levsKRP, label="Known KRP")
plot!(levsT, label="Target Tensor")
plot!( title="Leverage Score Values\nR=$(dim(r))",
 xlabel="Leverage Score Position",
  ylabel="Value", 
  legend=:topright)
savefig("levs_scores_rank_$(dim(r)).pdf")
####
# als = ITensorCPD.compute_als(T, cpd_exact; alg=ITensorCPD.InvKRP(), check=ITensorCPD.CPAngleCheck(-1, 10,), trunc_tol=1e-10,normal=false);
# cpd_exact = ITensorCPD.optimize(cpd_exact, als; verbose=true);

cpd = ITensorCPD.random_CPD(T, dim(r); rng=RandomDevice());
check=ITensorCPD.FitCheck(1e-5, 100, norm(T))
check_piv = ITensorCPD.CPAngleCheck(1e-8, 20, )

alg = ITensorCPD.direct()
alsNormal = ITensorCPD.compute_als(T, cpd; alg, check=check_piv);
## Choose to update only the factors in the KRP
# int_opt_T, als, _ = single_solve(als, cpd, 2);
# int_opt_T, als, _ = single_solve(als, int_opt_T, 3);
## Or run a small number of LS updates
int_opt_T = ITensorCPD.optimize(cpd,als;verbose=true);
als.target .= T
check_fit(T, int_opt_T)

samps = 1000
alg = ITensorCPD.SEQRCSPivProjected(1, samps, (1,2,3), 40)
alsQR = ITensorCPD.compute_als(T, cpd; alg, check=check_piv, normal=true);
alsQR.target .= T
## Choose to update only the factors in the KRP
# upcpd, upals, _ = single_solve(upals, cpd, 2);
# upcpd, upals, _ = single_solve(upals, upcpd, 3);
## Or run a small number of LS updates
upcpd = ITensorCPD.optimize(cpd, alsQR; verbose=true);

check_fit(T, upcpd)



W1 = ITensorCPD.had_contract(cpd_exact[2], cpd_exact[3], r)
levW1 = compute_lev_score(copy(reshape(array(W1), (90*90,dim(r)))))
W2 = ITensorCPD.had_contract(cpd[2], cpd[3], ITensorCPD.cp_rank(cpd))
levW2 = compute_lev_score(copy(reshape(array(W2), (90*90,dim(r)))))
W3 = ITensorCPD.had_contract(int_opt_T[2], int_opt_T[3], ITensorCPD.cp_rank(int_opt_T))
levW3 = compute_lev_score(copy(reshape(array(W3), (90*90,dim(r)))))
W4 = ITensorCPD.had_contract(upcpd[2], upcpd[3], ITensorCPD.cp_rank(upcpd))
levW4 = compute_lev_score(copy(reshape(array(W4), (90*90,dim(r)))))

angle_true_rand = acos(dot(levW1, levW2) / (norm(levW1) * norm(levW2))) 
angle_true_normal = acos(dot(levW1, levW3) / (norm(levW1) * norm(levW3))) 
angle_true_qr = acos(dot(levW1, levW4) / (norm(levW1) * norm(levW4))) 
println("The angle between the true KRP and the random guess is $(angle_true_rand)")
println("The angle between the true KRP and the normal equations is $(angle_true_normal)")
println("The angle between the true KRP and the QR-random LS is $(angle_true_qr)")

p = scatter(sort(levW1;rev=true)[1:105]; label="True Solution")
scatter!(sort(levW2; rev=true)[1:105]; label="Random Guess")
scatter!(sort(levW3;rev=true)[1:105]; label="Normal Equations")
scatter!(sort(levW4;rev=true)[1:105]; label="QR random")
plot!(title="CP rank = 50, bad = 10", ylabel="Sorted leverage scores", xlabel="Leverage score number")

display(p)
# savefig("Levs.pdf")

p = plot(levW1[400:500], label="True KRP", marker=:circle)
plot!(levW2[400:500], label="Normal Equations")
plot!(levW3[400:500]; marker=:square, label="Normal")
plot!(levW4[400:500]; marker=:diamond, label="QR Random")

display(p)


## Test for angle convergence of leverage scores
cpd = ITensorCPD.random_CPD(T, dim(r); rng=RandomDevice());
alg = ITensorCPD.SEQRCSPivProjected(1, samps, (1,2,3), 40)
alsQR = ITensorCPD.compute_als(T, cpd; alg, check=check_piv, normal=true);
alg = ITensorCPD.direct()
alsNormal = ITensorCPD.compute_als(T, cpd; alg, check=check_piv);

function compute_krp_lev_score(cpd, mode)
    r = ITensorCPD.cp_rank(cpd)
    W = ITensorCPD.had_contract(cpd.factors[1:end .!= mode], r)
    return compute_lev_score(reshape(array(W), (prod(dims(W)[1:end-1]), dim(r))))
end
reflevs1 = compute_krp_lev_score(cpd_exact, 1)
reflevs2 = compute_krp_lev_score(cpd_exact, 2)
reflevs3 = compute_krp_lev_score(cpd_exact, 3)

compute_angle(lev1, lev2) = dot(lev1, lev2) / (norm(lev1) * norm(lev2))
function lev_score_convergence(cpd, als, ref1, ref2, ref3, iter=5)
    levs1 = []
    levs2 = []
    levs3 = []
    upcpd, upals,_ = single_solve(als, cpd, 1)
    push!(levs1, compute_angle(ref1, compute_krp_lev_score(upcpd, 1)))
    push!(levs2, compute_angle(ref2, compute_krp_lev_score(upcpd, 2)))
    push!(levs3, compute_angle(ref3, compute_krp_lev_score(upcpd, 3)))

    upcpd, upals,_ = single_solve(upals, upcpd, 2)
    push!(levs1, compute_angle(ref1, compute_krp_lev_score(upcpd, 1)))
    push!(levs2, compute_angle(ref2, compute_krp_lev_score(upcpd, 2)))
    push!(levs3, compute_angle(ref3, compute_krp_lev_score(upcpd, 3)))

    upcpd, upals,_ = single_solve(upals, upcpd, 3)
    push!(levs1, compute_angle(ref1, compute_krp_lev_score(upcpd, 1)))
    push!(levs2, compute_angle(ref2, compute_krp_lev_score(upcpd, 2)))
    push!(levs3, compute_angle(ref3, compute_krp_lev_score(upcpd, 3)))
    for i in 1:iter-1
        for j in 1:length(cpd)
            upcpd, upals,_ = single_solve(upals, upcpd, j)
            push!(levs1, compute_angle(ref1, compute_krp_lev_score(upcpd, 1)))
            push!(levs2, compute_angle(ref2, compute_krp_lev_score(upcpd, 2)))
            push!(levs3, compute_angle(ref3, compute_krp_lev_score(upcpd, 3)))
        end
    end
    return levs1, levs2, levs3
end

normal1, normal2, normal3 = lev_score_convergence(cpd, alsNormal, reflevs1, reflevs2, reflevs3, 5);
normal1, normal2, normal3 = lev_score_convergence(cpd, alsQR, reflevs1, reflevs2, reflevs3, 5);

plot(normal1)
plot!(normal2)
plot!(normal3)