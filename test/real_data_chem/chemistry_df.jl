using PyCall, Conda, Pkg
using ITensorCPD, ITensors, LinearAlgebra,ITensors.NDTensors
#;source /Users/kpierce/Workspace/HaizhaoYang.github.io/codes/TN-MP-PDE-Solver/env/bin/activate
Pkg.build("PyCall")
include("../test_env.jl")

pyscf = pyimport("pyscf")
scipy = pyimport("scipy")
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
check=ITensorCPD.CPDiffCheck(1e-3, 50)
#for v in [3,4,5,6,7]
v = 30
pivdim = (Int(floor(v * dim(r))))

@show pivdim
@show pivdim / (nao * naux)
@show pivdim / (nao * nao)
fitsQRTrue = Vector{Float64}()

alg = ITensorCPD.SEQRCSPivProjected(1, pivdim, (1,2,3), (200,));
@time als = ITensorCPD.compute_als(target, cpd; alg, check);
rng = RandomDevice()
fitsRandTrue = Vector{Vector{Float64}}()
fitsQRTrue = Vector{Vector{Float64}}()
samples = [5, 10, 15, 20]
for α in [2]
    push!(fitsRandTrue, Vector{Float64}())
    push!(fitsQRTrue, Vector{Float64}())
    r = Index(Int(floor(α * naux)), "CP rank")
    cpd = ITensorCPD.random_CPD(target, r; rng);
    for v in samples
        @time als = ITensorCPD.update_samples(target, als, (Int(floor(v * dim(r)))); reshuffle=true);
        @time scpdRand = ITensorCPD.optimize(cpd, als; verbose=true);
        #push!(fitsQRAppx, check_appx_fit(als, scpdRand))
        push!(fitsQRTrue[α], check_fit(target, scpdRand))
    end
    
    for v in samples
        alsLev = ITensorCPD.compute_als(target, cpd; alg = ITensorCPD.LevScoreSampled((Int(floor(v * dim(r))))), check, normal=true);
        @time scpdRand = ITensorCPD.optimize(cpd, alsLev; verbose=true);
        push!(fitsRandTrue[α], check_fit(target, scpdRand))
    end


end

using LaTeXStrings
α = 3
ss =[(Int(floor(v * dim(r)))) for v in samples]
plot(ss, fitsQRTrue[α], label="SE-QRCS Sampling")
plot!(ss, fitsRandTrue[α], label ="Leverage Score Sampling")
name = α == 1 ? L"I_{\mathrm{aux}}" : α == 2 ? L"2 I_{\mathrm{aux}}" : L"3I_{\mathrm{aux}}"
plot!(title="CPD of "*L"\mathcal{B}" * "\n" * L"R=" * name,
yrange=[0.2,1], yticks=0.2:0.1:1,
ylabel="CPD Fit",
xlabel="Number of Samples",
legend=:bottomright)
savefig("$(@__DIR__)/../plots/h2o10_chem_rank_$(α).pdf")


α = 1.0
r = Index(Int(floor(α * naux)), "CP rank")
rankdim = dim(r)
cpd = ITensorCPD.random_CPD(target, r;);

fitsDiffK = Vector{Vector{Float64}}([Vector{Float64}(), Vector{Float64}(), Vector{Float64}()])

v = 30
for k_sk in [100,200,300,400]
    alg = ITensorCPD.SEQRCSPivProjected(1, pivdim, (1,2,3), (k_sk,));
    @time als = ITensorCPD.compute_als(target, cpd; alg, check);

    als = ITensorCPD.update_samples(target, als, (Int(floor(v/α * dim(r)))); reshuffle=true);
    for α in 1:3
        r = Index(Int(floor(α * naux)), "CP rank")
        cpd = ITensorCPD.random_CPD(target, r;);
        @time scpdRand = ITensorCPD.optimize(cpd, als; verbose=true);
        push!(fitsDiffK[α], check_fit(target, scpdRand))
    end
end
nsamples = Int(floor(v/α * dim(r))) # 126900, α = 1 v = 30
xs = [100, 200, 300, 400]
plot(xs, fitsDiffK[1], label=L"I_{\mathrm{aux}}")
plot!(xs, fitsDiffK[2], label=L"2I_{\mathrm{aux}}")
plot!(xs, fitsDiffK[3], label=L"3I_{\mathrm{aux}}")
name = α == 1 ? L"I_{\mathrm{aux}}" : α == 2 ? L"2 I_{\mathrm{aux}}" : L"3I_{\mathrm{aux}}"
plot!(title="Effect of SE-QRCS CountSketch on \nCPD approximation of " * L"\mathcal{B}",
yrange=[0.85,1.01],
ylabel="CPD Fit",
xlabel="Count Sketch Non-Zeros per Column",
legend=:bottomright)
savefig("$(@__DIR__)/../plots/chemistry/convergence_with_sketch_samples_$(nsamples).pdf")