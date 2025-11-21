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
for v in [5]
    se = []
    can = []
    lev = []
    full = []
    for atom in [wat2, wat3, wat4, wat5, wat6]
    mol = pyscf.gto.M(; atom=wat5, basis="cc-pvtz", charge=0, spin=0, verbose=3)
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
    Bleft = noprime(itensor(mf.mo_coeff, ind(Bleft, 2), ind(Bleft, 2)') * (itensor(mf.mo_coeff, ind(Bleft, 1), ind(Bleft,1)') * Bleft))
    Bright = itensor(deepcopy(array(Bleft)), Index(sz_df[1]), Index(sz_df[2]), aoI')
    M = itensor(invMetric, aoI, aoI')

    # Mhalf = itensor(halfMInv', aoI, aoI')
    # norm(Bleft * Mhalf * (Mhalf * Bright);  - Bleft * M * Bright)

    ## So the idea is: can we leverage this QR method to quickly construct some MPS
    ## Then we can naively transform the MPS into a 

    #dferi = Bleft * M * Bright
    
    α = 2.0
    r = Index(Int(floor(α * naux)), "CP rank")
    rankdim = dim(r)
    cpd = ITensorCPD.random_CPD(Bleft, r;);

    dferi_norm = norm(Bleft * M * Bright);
    linf = nothing
    #if v == 5
        alg = ITensorCPD.direct()
        als = ITensorCPD.compute_als(Bleft, cpd; alg, check=ITensorCPD.FitCheck(1e-3, 20, norm(Bleft)));
        optALS = ITensorCPD.optimize(cpd, als; verbose=true);
        
        BleftAppx = ITensorCPD.had_contract(optALS[1], optALS[2], r) * ITensorCPD.had_contract(optALS[3], optALS[], r)
        BrightAppx = itensor(data(BleftAppx), inds(Bright));
        diff = Bleft * M * Bright - BleftAppx * M * BrightAppx
        linf = maximum(diff)
        push!(full, 1 -norm(Bleft * M * Bright - BleftAppx * M * BrightAppx) / dferi_norm)
    end

    ##Run the same thing as above but with pivoted cheaper scheme
    # alg = ITensorCPD.SEQRCSPivProjected(1, Int(floor(6 * dim(r))), (1,2,3), (20,20,30))
    ## PRobably need to preserve symmetry in the shuffling pivots.
    linf_qr = []
    linf_seqr = []
    #for v in [3,4,5,6,7]
    pivdim = (Int(floor(v * dim(r))))

    @show pivdim
    @show pivdim / (nao * naux)
    @show pivdim / (nao * nao)
    @show pivdim / (nvirt * naux)
    @show pivdim / (nocc * nvirt)

    #samples = (pivdim, pivdim, Int(floor(0.9 * nocc * nvirt)))
    samples = (pivdim, pivdim, pivdim)

    alg = ITensorCPD.QRPivProjected(1, samples);
    als = ITensorCPD.compute_als(Bleft, cpd; alg, check=ITensorCPD.CPDiffCheck(1e-3, 50), shuffle_pivots=true);
    cpdRand = ITensorCPD.optimize(cpd, als; verbose=true);

    BleftAppx = ITensorCPD.had_contract(cpdRand[1], cpdRand[2], r) * ITensorCPD.had_contract(cpdRand[3], cpdRand[], r)
    BrightAppx = itensor(data(BleftAppx), inds(Bright));
    1 -norm(Bleft * M * Bright - BleftAppx * M * BrightAppx) / dferi_norm

    push!(linf_qr, maximum(Bleft * M * Bright - BleftAppx * M * BrightAppx))
    push!(can, 1 -norm(Bleft * M * Bright - BleftAppx * M * BrightAppx) / dferi_norm)


    alg = ITensorCPD.SEQRCSPivProjected(1, samples,
             (1,2,3), (100,100,100));
    als = ITensorCPD.compute_als(Bleft, cpd; alg, 
        check=ITensorCPD.CPDiffCheck(5e-5, 1000), shuffle_pivots=true);
    ## This forces the pivots of the first two modes to be the same, hopefully correcting for the symmetry of the problem.
    cpdRand = ITensorCPD.optimize(cpd, als; verbose=true);

    BleftAppx = ITensorCPD.had_contract(cpdRand[1], cpdRand[2], r) * ITensorCPD.had_contract(cpdRand[3], cpdRand[], r)
    BrightAppx = itensor(data(BleftAppx), inds(Bright));

    push!(linf_seqr, maximum(Bleft * M * Bright - BleftAppx * M * BrightAppx))
    push!(se, 1 -norm(Bleft * M * Bright - BleftAppx * M * BrightAppx) / dferi_norm)

    alg = ITensorCPD.LevScoreSampled(samples);
    als = ITensorCPD.compute_als(Bleft, cpd; alg, 
    check=ITensorCPD.CPDiffCheck(1e-3, 50));
    ## This forces the pivots of the first two modes to be the same, hopefully correcting for the symmetry of the problem.

    cpdRand = ITensorCPD.optimize(cpd, als; verbose=true);

    BleftAppx = ITensorCPD.had_contract(cpdRand[1], cpdRand[2], r) * ITensorCPD.had_contract(cpdRand[3], cpdRand[], r)
    BrightAppx = itensor(data(BleftAppx), inds(Bright));

    push!(lev, 1 -norm(Bleft * M * Bright - BleftAppx * M * BrightAppx) / dferi_norm)
    end
    push!(cp_qr, can)
    push!(cp_seqrcs, se)
    push!(cp_lev, lev)
    push!(cp_full, full)
# end

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