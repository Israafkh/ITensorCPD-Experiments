using PyCall, Conda, Pkg
using ITensorCPD, ITensors, LinearAlgebra,ITensors.NDTensors
#;source /Users/kpierce/Workspace/HaizhaoYang.github.io/codes/TN-MP-PDE-Solver/env/bin/activate
Pkg.build("PyCall")

pyscf = pyimport("pyscf")
scipy = pyimport("scipy")
## Need to pip install pyscf

wat3 = [
("O",    1.2361419,   1.0137761,  -0.0612424)
("H",    0.5104418,   0.8944555,   0.5514190)
("H",    1.9926927,   1.1973129,   0.4956931)
("O",   -0.9957202,   0.0160415,   1.2422556)
("H",   -1.4542703,  -0.5669741,   1.8472817)
("H",   -0.9377950,  -0.4817912,   0.4267562)
("O",   -0.2432343,  -1.0198566,  -1.1953808)
("H",    0.4367536,  -0.3759433,  -0.9973297)
("H",   -0.5031835,  -0.8251492,  -2.0957959)
]

atm = [
("O",   -0.185814,  -1.1749469,   0.7662596)
("H",   -0.1285513,  -0.8984365,   1.6808606)
("H",   -0.0582782,  -0.3702550,   0.2638279)
("O",    0.1747051,   1.1050002,  -0.7244430)
("H",   -0.5650842,   1.3134964,  -1.2949455)
("H",    0.9282185,   1.0652990,  -1.3134026)
]
 mol = pyscf.gto.M(; atom=atm, basis="cc-pvtz", charge=0, spin=0, verbose=3)
 auxbasis = "cc-pvtz-ri"
 mol.build()
 mf = pyscf.scf.RHF(mol).density_fit()

    # NOTE orthogonalize these!
auxmol = pyscf.df.addons.make_auxmol(mol, auxbasis)

ints_3c2e = pyscf.df.incore.aux_e2(mol, auxmol, intor="int3c2e")
ints_2c2e = auxmol.intor("int2c2e")
nao = mol.nao
naux = auxmol.nao

eri = mol.intor("int2e")
eriT = itensor(eri, Index.(size(eri)))

id = Diagonal(ones(naux))
invMetric = ints_2c2e \ id

aoI = Index(naux)
Bleft = itensor(ints_3c2e, ind(eriT, 1), ind(eriT, 2), aoI)
Bright = itensor(ints_3c2e, ind(eriT, 3), ind(eriT, 4), aoI')
M = itensor(invMetric, aoI, aoI')

1 -norm(eriT - Bleft * M * Bright) / norm(eriT)
# Now check the error of DF integrals wrt the normal ERIs

orbitals = mf.mo_coeff
nocc = mol.nelectron ÷ 2
nvirt = nao - nocc

α = 3.0
r = Index(Int(α * naux), "CP rank")
cpd = ITensorCPD.random_CPD(Bleft, r);
#alg = ITensorCPD.LevScoreSampled(2 * dim(r))
#alg = ITensorCPD.QRPivProjected(1, Int(3 * dim(r)),)

alg = ITensorCPD.direct()
als = ITensorCPD.compute_als(Bleft, cpd; alg, check=ITensorCPD.FitCheck(1e-3, 20, norm(Bleft)));
optALS = ITensorCPD.optimize(cpd, als; verbose=true);
BleftAppx = ITensorCPD.reconstruct(optALS)
BrightAppx = itensor(data(BleftAppx), inds(Bright))

### Compare the CPD approximated TEI tensor to the DF approximated TEI tensor
dferi = Bleft * M * Bright
1 -norm(eriT - BleftAppx * M * BrightAppx) / norm(eriT)

##Run the same thing as above but with pivoted cheaper scheme
# alg = ITensorCPD.SEQRCSPivProjected(1, Int(floor(6 * dim(r))), (1,2,3), (20,20,30))
alg = ITensorCPD.QRPivProjected(1, Int(floor(5 * dim(r))))
als = ITensorCPD.compute_als(Bleft, cpd; alg, check=ITensorCPD.CPDiffCheck(1e-4, 50));

cpdRand = ITensorCPD.optimize(cpd, als; verbose=true);

BleftAppx = ITensorCPD.reconstruct(cpdRand)
BrightAppx = itensor(data(BleftAppx), inds(Bright))

1 -norm(dferi - BleftAppx * M * BrightAppx) / norm(dferi)
