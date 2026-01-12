### Calculations for the first table 
include("$(@__DIR__)/../test_env.jl")
using LinearAlgebra, Statistics, DataFrames, CategoricalArrays, StatsPlots
## Generating a random tensor with CPD rank greater than dimension
rng=RandomDevice()
elt = Float64

c = 0.9
v = 0.1
r = 5
lng = 100000
nn = range(0.2, 0.8, length=lng)
A, cp = Colinearity_Tensor(r, (400,400,400), elt(c), [nn[rand(1:lng)] for i in 1:r], elt);
ind_A = inds(A)  
dims = dim.(ind_A);
C = randn(dims...);
N = itensor(C,ind_A...);
T = A+ v*(norm(A)/norm(N))*N
verbose= true
samples = 80
check_piv = ITensorCPD.CPAngleCheck(1e-5, 200)
check_direct = ITensorCPD.FitCheck(1e-5, 200,sqrt(sum(T.^2)) )
nrepeats = 1
rk = 5
r = Index(rk, "CP_rank")
cp_T = ITensorCPD.random_CPD(T,r;rng);

SEQRCS_error_vect = Vector{Float64}()
err_SEQRCS = Vector{Float64}()

opt = ITensorCPD.decompose(T, r; check=check_direct, verbose=true);

verbose=false
for q = 1:nrepeats
    alg = ITensorCPD.SEQRCSPivProjected(1, 80, (1,2,3), (5,))
    # alg = ITensorCPD.LevScoreSampled(80)
    alsQR = ITensorCPD.compute_als(T,cp_T;alg, check =check_piv);
    alsQR.target .= T
    int_opt_T =
    ITensorCPD.optimize(cp_T,alsQR;verbose);
    check_fit(alsQR, int_opt_T)
    push!(SEQRCS_error_vect,check_fit(alsQR, int_opt_T))
end

# m = [
#     (630, 677, 674) #0.5 0.01
#     (666, 690, 704) #0.9 0.01
#     (670, 645, 689) #0.5 0.1
#     (612, 669, 676) #0.9 0.1
# ]
m = [
    (745, 677, 656) # 0.5, 0.01
    (673, 668, 671) # 0.9 0.01
    (703, 710, 653) # 0.5, 0.1
    (715, 728, 693) # 0.9, 0.1
]

mean.(m)