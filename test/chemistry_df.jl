using PyCall, Conda, Pkg
using ITensorCPD, ITensors, LinearAlgebra,ITensors.NDTensors
#;source /Users/kpierce/Workspace/HaizhaoYang.github.io/codes/TN-MP-PDE-Solver/env/bin/activate
Pkg.build("PyCall")
include("test_env.jl")

pyscf = pyimport("pyscf")
scipy = pyimport("scipy")
## Need to pip install pyscf

include("mols.jl")

#cp_full = []
cp_seqrcs = []
cp_qr = []
cp_lev = []
cp_full = []
#for v in [5]
    se = []
    can = []
    lev = []
    full = []
    #for atom in [wat2, wat3, wat4, wat5, wat6]
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
    
    α = 3.0
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
    fitsQRAppx = Vector{Float64}()
    fitsQRTrue = Vector{Float64}()

    alg = ITensorCPD.SEQRCSPivProjected(1, pivdim, (1,2,3), (200,));
    @time als = ITensorCPD.compute_als(target, cpd; alg, check);
    samples = [5, 10, 15, 20, 25, 30]
    for v in [5, 10, 15, 20, 25, 30]
        @time als = ITensorCPD.update_samples(target, als, (Int(floor(v/α * dim(r)))); reshuffle=true);
        @time scpdRand = ITensorCPD.optimize(cpd, als; verbose=true);
        #push!(fitsQRAppx, check_appx_fit(als, scpdRand))
        push!(fitsQRTrue, check_fit(target, scpdRand))
    end
    
    fitsRandTrue = Vector{Float64}()
    for v in samples
        alsLev = ITensorCPD.compute_als(target, cpd; alg = ITensorCPD.LevScoreSampled((Int(floor(v/α * dim(r))))), check, normal=true);
        @time scpdRand = ITensorCPD.optimize(cpd, alsLev; verbose=true);
        push!(fitsRandTrue, check_fit(target, scpdRand))
    end

    using LaTeXStrings
    ss =[(Int(floor(v/α * dim(r)))) for v in samples]
    plot(ss, fitsQRTrue, label="SE-QRCS Sampling")
    plot!(ss, fitsRandTrue, label ="Leverage Score Sampling")
    name = α == 1 ? L"I_{\mathrm{aux}}" : α == 2 ? L"2 I_{\mathrm{aux}}" : L"3I_{\mathrm{aux}}"
    plot!(title="Decomposing "*L"\mathcal{B}" * "\n" * L"R=" * name,
    yrange=[0.2,1], yticks=0.2:0.1:1,
    ylabel="CPD Fit",
    xlabel="Number of Samples",
    legend=:bottomright)
    savefig("$(@__DIR__)/../plots/chem_rank_$(α).pdf")


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

    end

target = Bleft * Mhalf * prime(Bleft)
using Plots
#plot(cp_full.* 100, label="True CPD")
plot(cp_full.* 100, label="Exact CPD")
plot!(abs.(cp_qr[1].* 100), label="Full QR CPDnpiv = 5 * R")
plot!(abs.(cp_qr[2].* 100), label="Full QR CPDnpiv = 6 * R")
plot!(abs.(cp_qr[3].* 100), label="Full QR CPDnpiv = 7 * R")
plot!(abs.(cp_qr[4].* 100), label="Full QR CPDnpiv = 8 * R")

plot(cp_full.* 100, label="Exact CPD")
#plot!(cp_seqrcs[1].* 100, label="Randomized QR CPD npiv = 5 * R")
#plot!(cp_seqrcs[2].* 100, label="Randomized QR CPD npiv = 6 * R")
plot!(cp_seqrcs[3].* 100, label="Randomized QR CPD npiv = 7 * R")
plot!(cp_seqrcs[4].* 100, label="Randomized QR CPD npiv = 8 * R")
plot!(title="Accuracy of CPD of Gijab\nR=2 * naux", 
xlabel="Number of Water Molecules",
yaxis="L2 Percent Accuracy in Gijab")

plot(cp_full[1].* 100, label="Exact CPD")
# plot!(cp_lev[1] .* 100, label="Leverage Score Sampling Method npiv = 5 * R")
# plot!(cp_lev[2] .* 100, label="Leverage Score Sampling Method npiv = 6 * R")
plot!(cp_lev[3] .* 100, label="Leverage Score Sampling Method npiv = 7 * R")
plot!(cp_lev[4] .* 100, label="Leverage Score Sampling Method npiv = 8 * R")
plot!(title="Accuracy of CPD of Gijab\nR=2 * naux", 
xlabel="Number of Water Molecules",
yaxis="L2 Percent Accuracy in Gijab")
# savefig("cpd-thc.pdf")

auxs = [282, 423, 564,705]
occs = [10,15,20,25]
virs = [106, 159, 212, 265]
ranks = 2 .* auxs
plot((ranks .* 7) ./ (auxs .* occs ))
plot!((ranks .* 7) ./ (auxs .* virs ))
plot!((ranks .* 3.5) ./ (occs .* virs))


### MPS idea
    # mps = MPS(Bleft, inds(Bleft); cutoff=1e-5)
    # prod(linkdims(mps)) - 8 * naux
    # prod(linkdims(mps)) / nvirt
    # prod(linkdims(mps)) / naux
    # 1 - norm(contract(mps) - Bleft) / norm(Bleft)

    # itn = ITensorNetwork([mps...])
    # cpdRand = ITensorCPD.decompose(itn,  nvirt; check=ITensorCPD.FitCheck(1e-5, 100, norm(contract(mps))), verbose=true);
    # contract([mps...]) - Bleft
    # norm(contract(mps) - ITensorCPD.had_contract(cpdRand[1], cpdRand[2], r) * ITensorCPD.had_contract(cpdRand[3], cpdRand[], r)) / norm(contract(mps))
    # BleftAppx = ITensorCPD.had_contract(cpdRand[1], cpdRand[2], r) * ITensorCPD.had_contract(cpdRand[3], cpdRand[], r)
    # BrightAppx = itensor(data(BleftAppx), inds(Bright));
    # norm(Bleft * M * Bright - BleftAppx * M * BrightAppx) / norm(Bleft * M * Bright)

    # [700, 1440, 1920,2375, 2730]
    # [2.482269503546099,3.404255319148936,3.404255319148936,3.368794326241135,3.226950354609929]
    # [0.9969276893868366,0.9969106362083243,0.9969293666035572,0.9968693941509581,0.9968646126337909]


    using ITensors, LinearAlgebra
    a = randn(50, 50)
    b = randn(30,30)

    ua,_,_ = svd(a);
    ub,_,_ = svd(b);
    
    Ua = itensor(ua, Index.(size(ua)))
    Ub = itensor(ub, Index.(size(ub)))

    AB1 = random_itensor(ind(Ua,2), ind(Ub,1))
    AB2 = random_itensor(ind(Ua,2), ind(Ub,1))


    m1 = Ua * hadamard_product(AB1, AB2) * Ub
    m2 = hadamard_product(Ua * AB1 * Ub, Ua * AB2 * Ub)

    array(m1)
    array(m2)