import unittest
import numpy as np
from numpy.testing import assert_allclose
from dataclasses import replace
from rbmf_cs import TrainingTransform, PODBasis, TrainingCovariance, SyntheticConfig, make_synthetic


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.config=SyntheticConfig(nx=6,ny=5,n_train=40,n_test=4,latent_modes=4)
        self.data=make_synthetic(self.config)

    def test_reproducible_synthetic_and_spatial_correlation(self):
        other=make_synthetic(self.config)
        for name in self.data: assert_allclose(self.data[name],other[name],rtol=0,atol=0)
        corr=np.corrcoef(self.data['train_hf'].T)
        nearest=[corr[i,i+1] for i in range(30) if i%6<5]
        self.assertGreater(np.mean(nearest),.8)
        self.assertTrue(np.all(self.data['train_time']<self.data['test_time'][0]))

    def test_lf_independent_of_hf_discrepancy(self):
        changed=make_synthetic(replace(self.config,discrepancy_scale=3.))
        assert_allclose(self.data['test_lf'],changed['test_lf'],atol=0,rtol=0)
        self.assertFalse(np.allclose(self.data['test_hf'],changed['test_hf']))

    def test_pod_orthonormal_and_train_transform(self):
        transform=TrainingTransform.fit(self.data['train_hf'])
        x=transform.transform(self.data['train_hf'])
        assert_allclose(x.mean(axis=0),0,atol=1e-14)
        self.assertAlmostEqual(np.mean(x*x),1.)
        assert_allclose(transform.inverse(x),self.data['train_hf'],atol=1e-14)
        pod=PODBasis.fit(x,6)
        assert_allclose(pod.matrix.T@pod.matrix,np.eye(6),atol=1e-13)
        assert_allclose(x@pod.matrix@pod.matrix.T,x,atol=1e-12)
        self.assertAlmostEqual(pod.energy_fraction,1.)
        assert_allclose(transform.inverse_variance(np.ones(30)),np.full(30,transform.scale**2))

    def test_test_targets_cannot_change_training_artifacts(self):
        before=TrainingTransform.fit(self.data['train_hf'])
        basis=PODBasis.fit(before.transform(self.data['train_hf']),4)
        cov=TrainingCovariance.fit(before.transform(self.data['train_hf']))
        self.data['test_hf'][:]=1e12
        self.data['test_lf'][:]=-1e12
        after=TrainingTransform.fit(self.data['train_hf'])
        assert_allclose(before.mean,after.mean,atol=0,rtol=0)
        assert_allclose(basis.matrix,PODBasis.fit(after.transform(self.data['train_hf']),4).matrix,atol=0,rtol=0)
        assert_allclose(cov.factor,TrainingCovariance.fit(after.transform(self.data['train_hf'])).factor,atol=0,rtol=0)

    def test_factored_covariance_matches_numpy_empirical_covariance(self):
        x=self.data['train_hf']; cov=TrainingCovariance.fit(x); reference=np.cov(x,rowvar=False,ddof=1)
        assert_allclose(cov.diagonal(),np.diag(reference),atol=1e-12)
        assert_allclose(cov.block([1,3],[2,4,7]),reference[np.ix_([1,3],[2,4,7])],atol=1e-12)
        assert_allclose(cov.product(np.arange(30.)),reference@np.arange(30.),atol=1e-10)

    def test_degenerate_training_and_invalid_modes(self):
        with self.assertRaises(ValueError): TrainingTransform.fit(np.ones((3,4)))
        with self.assertRaises(ValueError): PODBasis.fit(np.ones((3,4)),5)
        with self.assertRaises(ValueError): TrainingCovariance.fit(np.full((3,4),np.nan))
        with self.assertRaises(ValueError): SyntheticConfig(temporal_ar=1.)


if __name__=='__main__': unittest.main()
