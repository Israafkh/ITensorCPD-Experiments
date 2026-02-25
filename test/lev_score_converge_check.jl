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
# cpd_exact = ITensorCPD.random_CPD(zeros(i,j,k), r);
# array(cpd_exact[1])[1:bad, :] .*= 100;
# array(cpd_exact[2])[1:bad, :] .*= 100;
# array(cpd_exact[3])[1:bad, :] .*= 100;
# array(cpd_exact[1]) .= array(cpd_exact[1])[randperm(i), :]
# array(cpd_exact[2]) .= array(cpd_exact[2])[randperm(j), :]
# array(cpd_exact[3]) .= array(cpd_exact[3])[randperm(k), :]
#  r = ITensorCPD.cp_rank(cpd_exact);

seed = Xoshiro(123)
rng = RandomDevice()
cpd_exact = construct_large_lev_score_cpd((i,j,k), r, bad;rng);
r = ITensorCPD.cp_rank(cpd_exact)

T = had_contract(cpd_exact[1], cpd_exact[2], r) * had_contract(cpd_exact[3], cpd_exact[], r)

als = ITensorCPD.compute_als(T, cpd_exact; alg=ITensorCPD.InvKRP(), check=ITensorCPD.CPAngleCheck(-1, 10,), trunc_tol=1e-10,normal=false);
cpd_exact = ITensorCPD.optimize(cpd_exact, als; verbose=true);

cpd = ITensorCPD.random_CPD(T, dim(r); rng=RandomDevice());
check=ITensorCPD.FitCheck(1e-5, 100, norm(T))
check_piv = ITensorCPD.CPAngleCheck(1e-8, 500, )
# alg = ITensorCPD.SEQRCSPivProjected(1, 50, (1,2,3), 40)
alg = ITensorCPD.direct()
als = ITensorCPD.compute_als(T, cpd; alg, check=check_piv,normal=false);
int_opt_T = ITensorCPD.optimize(cpd,als;verbose=true);
als.target .= T
check_fit(T, int_opt_T)

samps = 1000
alg = ITensorCPD.SEQRCSPivProjected(1, samps, (1,2,3), 40)
upals = ITensorCPD.compute_als(T, cpd; alg, check=check_piv, normal=true);
upals.target .= T
upcpd = ITensorCPD.optimize(cpd, upals; verbose=true);
check_fit(T, upcpd)

W1 = ITensorCPD.had_contract(cpd_exact[2], cpd_exact[3], r)
levW1 = compute_lev_score(copy(reshape(array(W1), (90*90,dim(r)))))
W2 = ITensorCPD.had_contract(int_opt_T[2], int_opt_T[3], ITensorCPD.cp_rank(int_opt_T))
levW2 = compute_lev_score(copy(reshape(array(W2), (90*90,dim(r)))))
W3 = ITensorCPD.had_contract(upcpd[2], upcpd[3], ITensorCPD.cp_rank(upcpd))
levW3 = compute_lev_score(copy(reshape(array(W3), (90*90,dim(r)))))

angle_true_normal = acos(dot(levW1, levW2) / (norm(levW1) * norm(levW2))) 
angle_true_qr = acos(dot(levW1, levW3) / (norm(levW1) * norm(levW3))) 
println("The angle between the true KRP and the normal equations is $(angle_true_normal)")
println("The angle between the true KRP and the QR-random LS is $(angle_true_qr)")

p = plot(sort(levW1; rev=true)[1:1000]; label="Exact CPD")
plot!(sort(levW2; rev=true)[1:1000]; label="Normal Equation")
plot!(sort(levW3;rev=true)[1:1000]; label="SE-QRCS with 10 samples")
plot!(title="CP rank = 50, bad = 10", ylabel="Sorted leverage scores", xlabel="Leverage score number")

display(p)
# savefig("Levs.pdf")

p = plot(levW1[1:100], label="True KRP", marker=:circle)
plot!(levW2[1:100], label="Normal Equations")
plot!(levW3[1:100]; alpha = 0.8, label="QR Random")

display(p)