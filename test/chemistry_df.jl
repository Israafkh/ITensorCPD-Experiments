using PyCall, Conda, Pkg
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
 nao = mol.nao
 orbitals = mf.mo_coeff
naux = auxmol.nao
nocc = mol.nelectron ÷ 2
nvirt = nao - nocc

ovx = ints_3c2e[1:nocc, nocc+1:end, :]
vvx = ints_3c2e[nocc+1:end, nocc+1:end, :]

# ints_3c is the 3-center integral tensor (ij|P), where i and j are the
# indices of AO basis and P is the auxiliary basis
#ints_2c2e = auxmol.intor('int2c2e')

using ITensors, ITensorCPD
#is = Index.(size(ints_3c2e))
is = Index.(size(ovx))
Bov = itensor(ovx, is)
is = Index.(size(vvx))
Bvv = itensor(vvx, is)

α = 5
r = Index(Int(α * size(ints_3c2e)[3]), "CP rank")
cpd = ITensorCPD.random_CPD(Bov, r);
#alg = ITensorCPD.LevScoreSampled(2 * dim(r))
#alg = ITensorCPD.QRPivProjected(1, Int(3 * dim(r)),)
alg = ITensorCPD.direct()
als = ITensorCPD.compute_als(Bov, cpd; alg, check=ITensorCPD.FitCheck(1e-3, 20, norm(Bov)));
cpd_ovx = ITensorCPD.optimize(cpd, als; verbose=true);

fac1 = itensor(array(cpd_ovx.factors[2]), ind(Bvv,1), r)
fac2 = itensor(array(cpd_ovx.factors[2]), ind(Bvv,2), r)
fac3 = itensor(array(cpd_ovx.factors[3]), ind(Bvv,3), r)
cpd_vvx = ITensorCPD.CPD{ITensor}([fac1, fac2, fac3], cpd_ovx.λ)
alg = ITensorCPD.QRPivProjected(1, Int(3 * dim(r)),)
#als = ITensorCPD.compute_als(Bvv, cpd_vvx; alg, check=ITensorCPD.FitCheck(1e-3, 20, norm(Bvv)));
ITensorCPD.optimize(cpd_vvx, als; verbose=true);
