using PyCall, Conda, Pkg
using ITensorCPD, ITensors, LinearAlgebra,ITensors.NDTensors
using LaTeXStrings
# ;source /Users/kpierce/Workspace/HaizhaoYang.github.io/codes/TN-MP-PDE-Solver/env/bin/activate
Pkg.build("PyCall")
include("../test_env.jl")

pyscf = pyimport("pyscf")
scipy = pyimport("scipy")
use_threads_als = true
## Need to pip install pyscf

include("mols.jl")

mol = pyscf.gto.M(; atom=wat10, basis="cc-pvtz", charge=0, spin=0, verbose=3)
auxbasis = "cc-pvtz-ri"
mol.build()
mf = pyscf.scf.RHF(mol).density_fit()
mf.run()
    # NOTE orthogonalize these!
auxmol = pyscf.df.addons.make_auxmol(mol, auxbasis)

## Am I in the correct basis?
ints_3c2e = pyscf.df.incore.aux_e2(mol, auxmol, intor="int3c2e")
ints_2c2e = auxmol.intor("int2c2e")
mf.mo_coeff ## Ordered AO x MO

nao = mol.nao
naux = auxmol.nao
orbitals = mf.mo_coeff
nocc = mol.nelectron ÷ 2
nvirt = nao - nocc

    
# eriT = itensor(eri, Index.(size(eri)))

id = Diagonal(ones(naux))
L = cholesky(Hermitian(ints_2c2e))
invMetric = ints_2c2e \ id

halfMInv = (cholesky(Hermitian(ints_2c2e))).L \ id

#eri = eri[1:nocc, nocc+1:end, 1:nocc, nocc+1:end]
df_int = ints_3c2e[:, :, :]
aoI = Index(naux, "aux")
sz_df = size(df_int)
Bleft = itensor(df_int,  Index(sz_df[1]), Index(sz_df[2]), aoI)
M = itensor(invMetric, aoI, aoI')
Mhalf = itensor(halfMInv, aoI, aoI')

# Mhalf = itensor(halfMInv', aoI, aoI')
# norm(Bleft * Mhalf * (Mhalf * Bright);  - Bleft * M * Bright)

## So the idea is: can we leverage this QR method to quickly construct some MPS
## Then we can naively transform the MPS into a 

# target = Bleft * Mhalf
target = Bleft

α = 1.0
r = Index(Int(floor(α * naux)), "CP rank")
rankdim = dim(r)
cpd = ITensorCPD.random_CPD(target, r;);

linf = nothing

##Run the same thing as above but with pivoted cheaper scheme
# alg = ITensorCPD.SEQRCSPivProjected(1, Int(floor(6 * dim(r))), (1,2,3), (20,20,30))
## PRobably need to preserve symmetry in the shuffling pivots.
linf_qr = []
linf_seqr = []
check=ITensorCPD.CPDiffCheck(1e-10, 50)
#for v in [3,4,5,6,7]
v = 30
pivdim = (Int(floor(v * dim(r))))

@show pivdim
@show pivdim / (nao * naux)
@show pivdim / (nao * nao)
fitsQRTrue = Vector{Float64}()


if !use_threads_als
    BLAS.set_num_threads(1)
end
alg = ITensorCPD.SEQRCSPivProjected(1, pivdim, (1,2,3), (200,));
# algK = ITensorCPD.KSEQRCSPivProjected(1, pivdim,);
# α = 1.0
# r = Index(Int(floor(α * naux)), "CP rank")
# rankdim = dim(r)
# cpd = ITensorCPD.random_CPD(target, r;);
seqrcs_time = @elapsed als = ITensorCPD.compute_als(target, cpd; alg, check, trunc_tol=0);
# # @time alsK = ITensorCPD.compute_als(target, cpd; alg=algK, check, 
#                 prelim_sample_size= 20 * naux, prelim_niter=10, trunc_tol=0,guess_num_levs=Int(floor(4 * naux)));
rng = RandomDevice()
fitsRandTrue = Vector{Vector{Float64}}()
timesQR = Vector{Vector{Float64}}()
fitsQRTrue = Vector{Vector{Float64}}()
timesRand = Vector{Vector{Float64}}()

for lst in [
    fitsRandTrue,
    timesQR,
     fitsQRTrue, timesRand,
     ]
    #  fitsKQRTrue, timesKQR]
    for i in 1:3
    push!(lst, Vector{Float64}())
    end
end
samples = [3, 5, 10, 15, 20]
for α in [Int(α)]
    r = Index(Int(floor(α * naux)), "CP rank")
    cpd = ITensorCPD.random_CPD(target, r; rng);

    for v in samples
        als = ITensorCPD.update_samples(target, als, (Int(floor(v * dim(r)))); reshuffle=true);
        push!(timesQR[α], @elapsed scpdRand = ITensorCPD.optimize(cpd, als; verbose=true));
        push!(fitsQRTrue[α], check_fit(target, scpdRand))
    end
    for v in samples
        @time alsLev = ITensorCPD.compute_als(target, cpd; alg = ITensorCPD.LevScoreSampled((Int(floor(v * dim(r))))), check, normal=true, stop_resample=-1);
        push!(timesRand[α], @elapsed scpdRand = ITensorCPD.optimize(cpd, alsLev; verbose=true));
        push!(fitsRandTrue[α], check_fit(target, scpdRand))
    end
end

## Updated times with parallel + tensor based Sampling
timesQR =[
    [35.487285875, 38.203835, 66.110605542, 86.634159458, 110.877763625],
    [146.994172875, 197.965369625, 342.3873375, 487.638201291, 632.660010916],
    []
]
timesRand = [
 [40.273820375, 53.505516042, 87.204263708, 120.393692459, 154.238111458],
 [159.290998583, 222.74811875, 409.948889208, 622.567260708, 801.439265166],
 [],
]
α = Int(α)
lw = 4

plot_acc = false
if plot_acc
    r = Index(Int(floor(α * naux)), "CP rank")
    ss = samples .* dim(r) #[(Int(floor(v * dim(r)))) for v in samples]
    plot(ss, fitsRandTrue[α], label ="Leverage Score Sampling"; lw)
    plot!(ss, fitsQRTrue[α], label="SE-QRCS Sampling"; lw)
    # plot!(ss, fitsKQRTrue[α], label="KRP SE-QRCS Sampling"; lw)
    name = α == 1 ? L"I_{\mathrm{aux}}" : α == 2 ? L"2 I_{\mathrm{aux}}" : L"3I_{\mathrm{aux}}"
    plot!(title="CPD approximation of "*L"\mathcal{B}" * "\n" * L"R=" * name,
    yrange=[0.2,1],
    yticks=-0.2:0.1:1,
    ## 1
    xticks= α== 3 ?  (0 * 10^4 : 2*10^4: 8*10^4) : (0 * 10^4: 1*10^4: 8*10^4),
    xrange = [0.5* 10^4, 2.5*10^4],
    ## 2
    # xticks= α== 3 ?  (0 * 10^4 : 2*10^4: 8*10^4) : (0 * 10^4: 1*10^4: 8*10^4),
    # xrange = [1* 10^4, 5.5*10^4],
    ## 3
    # xticks= α== 3 ?  (0 * 10^4 : 2*10^4: 8*10^4) : (0 * 10^4: 1*10^4: 8*10^4),
    # xrange = [1.5* 10^4, 8.5*10^4],
    # yscale=:log10
    ylabel="CPD Fit",
    xlabel="Number of Samples",
    xformatter=:scientific,
    legendfontsize=11,
    titlefontsize=16,
    labelfontsize=15,
    tickfontsize=11,
    legend=:bottomright)
    savefig("$(@__DIR__)/../../plots/chemistry/h2o10_chem_rank_$(α).pdf")
end

# seqrcs_time = 81.678826
seqrcs_time = 43.986238209
r = Index(Int(floor(α * naux)), "CP rank")
ss =[(Int(floor(v * dim(r)))) for v in samples]
name = α == 1 ? L"I_{\mathrm{aux}}" : α == 2 ? L"2 I_{\mathrm{aux}}" : L"3I_{\mathrm{aux}}"
plot(ss, timesRand[α], label="Leverage Score Sampling ALS", marker=:circle; lw)
plot!(ss, timesQR[α] .+ seqrcs_time, label="SE-QRCS + ALS", marker=:square; lw)
plot!(ss, ones(length(samples)) .* seqrcs_time, label="Only SE-QRCS"; lw)
plot!(ss, timesQR[α], label="Only ALS"; lw)
plot!(
title="Computational Cost to Decompose " 
* L"\mathcal{B}" * "\n" *L"R =" * name,
ylabel="Time (s)", 
xlabel="Number of Samples",
# yrange=[0,250],
legendcolumns=1,
legendfontsize=8,
titlefontsize=16,
labelfontsize=15,
tickfontsize=11,
legend=:topleft,
xformatter=:scientific,
## 1
xticks= α== 3 ?  (0 * 10^4 : 2*10^4: 8*10^4) : (0 * 10^4: 1*10^4: 8*10^4),
xrange = [0.5* 10^4, 3.2*10^4],
## 2
# xticks= α== 3 ?  (0 * 10^4 : 2*10^4: 8*10^4) : (0 * 10^4: 1*10^4: 8*10^4),
# xrange = [1* 10^4, 6.3*10^4],
## 3
# xticks= α== 3 ?  (0 * 10^4 : 2*10^4: 8*10^4) : (0 * 10^4: 1*10^4: 8*10^4),
# xrange = [1.5* 10^4, 8.5*10^4],
# yscale=:log10
)
savefig("$(@__DIR__)/../../plots/chemistry/h2o10_time_rank_$(α).pdf")
