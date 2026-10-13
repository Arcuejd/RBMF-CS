import sys,tempfile,unittest,subprocess
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from scipy import stats
from rbmf_cs.model import RBMFCS,VBConfig
from rbmf_cs.p0_diagnostics import DiagnosticRBMF,FixedPrecision
from rbmf_cs.p0_holdout_gate import require_qualified_release
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_p0_diagnosis import load_development

class P0Tests(unittest.TestCase):
    def setUp(self):
        self.psi=np.linalg.qr(np.random.default_rng(31).normal(size=(8,3)))[0]
        self.low=self.psi@np.array([1.,-.3,.7]);self.ids=np.array([0,2,3,6]);self.y=self.low[self.ids]+.02
        self.cfg=VBConfig(a_gamma=2.,b_gamma=1.,a_delta=2.,b_delta=1.,convergence='factors',tolerance=1e-8,max_iter=3000)

    def test_instrumentation_preserves_original_exactly(self):
        a=RBMFCS(self.psi,self.cfg).fit(self.low,self.y,indices=self.ids)
        b=DiagnosticRBMF(self.psi,self.cfg).fit(self.low,self.y,indices=self.ids)
        assert_allclose(a.mean_alpha_,b.mean_alpha_,atol=0,rtol=0)
        self.assertEqual(a.diagnostics_['history'],b.diagnostics_['history'])

    def test_known_noise_elbo_independent_density_integration(self):
        m=DiagnosticRBMF(self.psi,self.cfg,fixed_tau=25.).fit(self.low,self.y,indices=self.ids)
        c=self.cfg;rng=np.random.default_rng(48);n=80000
        a=rng.multivariate_normal(m.mean_alpha_,m.cov_alpha_,n)
        r=rng.normal(m.rho_mean_,np.sqrt(m.rho_variance_),n)
        logq=stats.multivariate_normal.logpdf(a,m.mean_alpha_,m.cov_alpha_)+stats.norm.logpdf(r,m.rho_mean_,np.sqrt(m.rho_variance_))
        logp=stats.norm.logpdf(r,c.rho0,1/np.sqrt(c.kappa_rho0));samples={}
        for name,f,aa,bb in [('gamma',m.gamma_,c.a_gamma,c.b_gamma),('local',m.local_,c.nu/2,c.nu/2),('delta',m.delta_,c.a_delta,c.b_delta)]:
            shape=np.atleast_1d(f.shape);rate=np.atleast_1d(f.rate);x=rng.gamma(shape,1/rate,size=(n,len(shape)));samples[name]=x
            logq+=stats.gamma.logpdf(x,shape,scale=1/rate).sum(1)
            logp+=stats.gamma.logpdf(x,aa,scale=1/bb).sum(1)
        logp+=stats.norm.logpdf(a,scale=1/np.sqrt(samples['gamma'])).sum(1)
        logp+=stats.norm.logpdf(m.y_,loc=a@m.h_.T,scale=1/np.sqrt(25*samples['local'])).sum(1)
        logp+=stats.norm.logpdf(0,loc=a@m.basis.T-r[:,None]*m.low_,scale=1/np.sqrt(samples['delta'])).sum(1)
        integrand=logp-logq
        self.assertLess(abs(m._elbo()-integrand.mean()),6*integrand.std()/np.sqrt(n))
        self.assertEqual(m.diagnostics_['elbo_decreases'],0)
        # Independent augmented least squares for the synchronized posterior.
        h=m.h_; aug=np.vstack([np.sqrt(25*m.local_.mean)[:,None]*h,np.diag(np.sqrt(m.gamma_.mean)),np.sqrt(m.delta_.mean)*self.psi])
        rhs=np.r_[np.sqrt(25*m.local_.mean)*m.y_,np.zeros(3),np.sqrt(m.delta_.mean)*m.rho_mean_*self.low]
        assert_allclose(m.mean_alpha_,np.linalg.lstsq(aug,rhs,rcond=None)[0],atol=1e-7)
        self.assertAlmostEqual(m.marginal_noise_variance(),c.nu/(c.nu-2)/25)

    def test_reject_zero_or_invalid_known_noise(self):
        for x in [0.,-1.,np.inf,np.nan]:
            with self.assertRaises(ValueError): FixedPrecision(x)

    def test_development_input_rejects_injected_holdout(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);np.savez(p/'aligned_fields.npz',test_hf_model=np.ones((1,2)))
            with self.assertRaisesRegex(ValueError,'test arrays'):load_development(p)

    def test_development_input_rejects_test_dates_disguised_as_validation(self):
        root=Path(__file__).resolve().parents[1];d=load_development(root/'outputs/comparison_validation_v1/inputs')
        d['validation_dates']=np.arange(201501,201513)
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);np.savez(p/'aligned_fields.npz',**d)
            with self.assertRaisesRegex(ValueError,'validation dates'):load_development(p)

    def test_process_guard_rejects_sealed_and_raw_reads(self):
        root=Path(__file__).resolve().parents[1]
        code="""
from pathlib import Path
from rbmf_cs.p0_isolation import install_development_guard
root=Path.cwd();install_development_guard(root,root/'outputs/p0_guard_test')
for p in [root/'outputs/p0_holdout_v1/sealed/temperature_anomaly_50m_candidates.npz',Path('D:/Dataset/anything.nc')]:
    try: p.open('rb')
    except PermissionError: pass
    else: raise AssertionError('forbidden read succeeded')
print('denied both')
"""
        result=subprocess.run([sys.executable,'-c',code],cwd=root,capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('denied both',result.stdout)

    def test_unreleased_holdout_evaluation_gate(self):
        root=Path(__file__).resolve().parents[1]
        with self.assertRaisesRegex(PermissionError,'remains sealed'):
            require_qualified_release(root/'outputs/p0_holdout_v1')

if __name__=='__main__':unittest.main()
