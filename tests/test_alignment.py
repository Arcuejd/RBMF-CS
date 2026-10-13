import unittest
import numpy as np
from numpy.testing import assert_allclose
from rbmf_cs.alignment import periodic_stencil, sample_stencil, vertical_bracket, training_artifacts


class AlignmentTests(unittest.TestCase):
    def test_periodic_dateline_and_zero_weight_land(self):
        lat=np.array([70.,71.]); lon=np.arange(.5,360,1.)
        a=np.tile(np.arange(360,dtype=float),(2,1))
        a[1]=np.nan
        idx,w=periodic_stencil(lat,lon,[[70.,0.],[70.,360.],[70.,-1.]])
        assert_allclose(sample_stencil(a.ravel(),idx,w),[179.5,179.5,358.5])
        assert_allclose(w.sum(axis=1),1.)
        a[0,359]=np.nan
        self.assertTrue(np.isnan(sample_stencil(a.ravel(),idx,w)).all())

    def test_vertical_and_spatial_extrapolation_rejected(self):
        self.assertEqual(vertical_bracket([10.,40.,60.],50.),(1,2,.5))
        self.assertEqual(vertical_bracket([10.,40.,60.],40.),(1,1,0.))
        with self.assertRaises(ValueError): vertical_bracket([10.,40.,60.],1.)
        with self.assertRaises(ValueError): periodic_stencil([70.,71.],np.arange(360.),[[72.,0.]])

    def test_test_year_cannot_enter_artifact_fit(self):
        dates=np.array([y*100+m for y in [2011,2012] for m in range(1,13)])
        x=np.random.default_rng(12).normal(size=(24,5))
        result=training_artifacts(x,dates)
        for m in range(1,13):
            assert_allclose(result['train_anomaly_c'][dates%100==m].mean(axis=0),0.,atol=1e-15)
        assert_allclose(result['basis'].T@result['basis'],np.eye(result['basis'].shape[1]),atol=1e-12)
        corrupted=dates.copy(); corrupted[-1]=201401
        with self.assertRaisesRegex(ValueError,'reserved'):
            training_artifacts(x,corrupted)

    def test_monthly_offsets_cancel_after_train_only_rebasing(self):
        dates=np.array([y*100+m for y in range(2000,2004) for m in range(1,13)])
        rng=np.random.default_rng(15)
        x=rng.normal(size=(len(dates),6)); fixed=rng.normal(size=(12,6))*20
        first=training_artifacts(x,dates)
        second=training_artifacts(x+fixed[dates%100-1],dates)
        assert_allclose(first['train_model'],second['train_model'],atol=2e-14)
        assert_allclose(first['basis']@first['basis'].T,second['basis']@second['basis'].T,atol=1e-12)


if __name__ == '__main__': unittest.main()
