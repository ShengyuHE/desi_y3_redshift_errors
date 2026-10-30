"""Check covariance normalization and reuse through real lsstypes HDF5 I/O."""
import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clustering import clustering_covariance as cache


class TestCovarianceCache(unittest.TestCase):
    def test_native_matching_rebins_full_covariance(self):
        catalog = dict(version='holi-v3', domain='altmtl', tracer='LRG1', zrange=(0.4, 0.6))
        pole = cache.types.Mesh2SpectrumPole(k=[0.5, 1.5, 2.5, 3.5],
            k_edges=np.array([[0., 1.], [1., 2.], [2., 3.], [3., 4.]]),
            num_raw=np.zeros(4), nmodes=np.array([1., 3., 2., 2.]))
        observable = cache.types.ObservableTree([pole], ells=[0])
        matrix = np.diag([1., 4., 9., 16.]) + 0.5
        covariance = cache.types.CovarianceMatrix(matrix, observable=observable)
        target = observable.select(k=slice(0, None, 2))
        transform = np.array([[0.25, 0.75, 0., 0.], [0., 0., 0.5, 0.5]])
        expected = transform @ matrix @ transform.T
        with tempfile.TemporaryDirectory() as directory:
            covariance.write(cache.get_cov_fn(**catalog, cache_dir=directory).format('mesh2_spectrum_poles'))
            with patch.object(cache, 'read_data_from_fn', side_effect=AssertionError('Must not read mocks')):
                matched = cache.load_plot_covariance(catalog, 'mesh2', target, cache_dir=directory)
                np.testing.assert_allclose(matched.value(), expected)
                errors = cache.get_saved_error_bars(catalog, 'mesh2', target, cache_dir=directory)
                np.testing.assert_allclose(errors, np.sqrt(np.diag(expected))[None, :])

    def test_saved_covariance_matches_bins_and_multipoles_without_mock_reads(self):
        catalog = dict(version='holi-v3', domain='altmtl', tracer='LRG1', zrange=(0.4, 0.6))
        k = np.array([0.025, 0.035, 0.045, 0.055])
        observable = cache.types.ObservableTree([
            cache.types.ObservableLeaf(coords=['k'], k=k, value=np.zeros(len(k)))
            for _ in range(2)], ells=[0, 2])
        generator = np.random.default_rng(42)
        samples = generator.normal(size=(30, 8))
        matrix = np.cov(samples, rowvar=False)
        covariance = cache.types.CovarianceMatrix(matrix, observable=observable,
            attrs={'config_json': json.dumps(dict(corr='mesh2', bins=[0.02, 0.35, 0.01, None]))})
        with tempfile.TemporaryDirectory() as directory:
            fn = cache.get_cov_fn(**catalog, cache_dir=directory).format('mesh2_spectrum_poles')
            covariance.write(fn)
            with patch.object(cache, 'read_data_from_fn', side_effect=AssertionError('Must not read mocks')):
                # Different ordering, scale cuts and tiny catalog-dependent center shifts.
                target = k[[2, 0]] + 1.e-6
                matched = cache.load_plot_covariance(catalog, 'mesh2', target, ells=[2, 0],
                                                     cache_dir=directory)
                indices = [6, 4, 2, 0]
                np.testing.assert_allclose(matched.value(), matrix[np.ix_(indices, indices)])
                errors = cache.get_saved_error_bars(catalog, 'mesh2', target, ells=[2, 0],
                                                   cache_dir=directory)
                np.testing.assert_allclose(errors, np.sqrt(np.diag(matrix)[indices]).reshape(2, 2))
                for invalid in [k + 0.005, [0.6], [k[0], k[0]]]:
                    with self.assertRaisesRegex(ValueError, 'bins do not match'):
                        cache.load_plot_covariance(catalog, 'mesh2', invalid, ells=[0],
                                                   cache_dir=directory)
                with self.assertRaisesRegex(ValueError, 'bin width'):
                    cache.load_plot_covariance(catalog, 'mesh2', k, ells=[0],
                                               bin_type='y3_fs', cache_dir=directory)
                with self.assertRaisesRegex(FileNotFoundError, 'No saved covariance'):
                    cache.load_plot_covariance(dict(catalog, use_dv='verr_nonparam'),
                                               'mesh2', k, ells=[0], cache_dir=directory)

    def test_covariance_matches_sample_scatter_and_cache_avoids_mock_reads(self):
        catalog = dict(version='holi-v3', domain='altmtl', tracer='LRG1', zrange=(0.4, 0.6))
        original = dict(catalog)
        data = np.arange(30, dtype=float).reshape(5, 3, 2)**2

        def filename(**options):
            path = Path(directory) / f'mock{options["mock_id"]}'
            path.mkdir(exist_ok=True)
            (path / 'mesh2_spectrum_poles.h5').touch()
            return str(path / '{}.h5')

        def read(filename, *args, **kwargs):
            index = int(Path(filename).parent.name[4:])
            if kwargs.get('return_observable'):
                return cache.types.ObservableTree([
                    cache.types.ObservableLeaf(coords=['k'], k=[0.05, 0.15], value=v)
                    for v in data[index]], ells=[0, 2, 4]), None
            return (np.array([0.05, 0.15]), None if index == 4 else data[index]), None

        with tempfile.TemporaryDirectory() as directory, \
             patch.object(cache, 'get_measurement_fn', side_effect=filename), \
             patch.object(cache, 'read_data_from_fn', side_effect=read) as reader:
            covariance = cache.get_mock_covariance(catalog, 'mesh2', num=5,
                skip_mock_ids=[1], cache_dir=directory)
            np.testing.assert_allclose(covariance.value(),
                np.cov(data[[0, 2, 3]].reshape(3, -1), rowvar=False, ddof=1))
            self.assertEqual(covariance.attrs['mock_ids'], [0, 2, 3])
            self.assertEqual(covariance.attrs['missing_mock_ids'], [4])
            self.assertEqual(covariance.attrs['nobs'], 3)
            reader.reset_mock()
            reader.side_effect = AssertionError('Cache hit must not read mock measurements')
            error = cache.get_mock_error_bars(catalog, 'mesh2', num=5,
                skip_mock_ids=[1], cache_dir=directory)
            np.testing.assert_allclose(error, data[[0, 2, 3]].std(axis=0, ddof=1))
            reader.assert_not_called()
            self.assertEqual(catalog, original)

    def test_selection_is_checked_without_a_filename_hash(self):
        catalog = dict(version='holi-v3', domain='altmtl', tracer='QSO', zrange=(0.8, 2.1))
        with tempfile.TemporaryDirectory() as directory:
            catalog['base_dir'] = directory
            for dv in [False, 'verr_nonparam']:
                for mock_id in range(4):
                    fn = cache.get_measurement_fn(**catalog, use_dv=dv, mock_id=mock_id)
                    Path(fn.format('mesh2_spectrum_poles')).touch()
            def read(filename, *args, **kwargs):
                if kwargs.get('return_observable'):
                    return cache.types.ObservableTree([
                        cache.types.ObservableLeaf(coords=['k'], k=[0.1, 0.2], value=np.ones(2))
                        for _ in range(3)], ells=[0, 2, 4]), None
                return (np.array([0.1, 0.2]), np.ones((3, 2))), None

            with patch.object(cache, 'read_data_from_fn', side_effect=read) as reader:
                first = cache.get_mock_covariance(catalog, 'mesh2', num=4,
                    skip_mock_ids=[], cache_dir=directory, bin_type='test')
                self.assertEqual(Path(first.attrs['cache_path']).name,
                                 'covariance_mesh2_spectrum_poles_QSO_z0.8-2.1_holi_v3.h5')
                reader.reset_mock()
                # test/y3_sys have the same effective mesh2 binning.
                cache.get_mock_covariance(catalog, 'mesh2', num=4,
                    skip_mock_ids=[], cache_dir=directory, bin_type='y3_sys')
                reader.assert_not_called()
                with self.assertRaisesRegex(ValueError, 'different binning or mock selection'):
                    cache.get_mock_covariance(catalog, 'mesh2', num=4,
                        skip_mock_ids=[1], cache_dir=directory)
                second = cache.get_mock_covariance(dict(catalog, use_dv='verr_nonparam'),
                    'mesh2', num=4, skip_mock_ids=[], cache_dir=directory)
                self.assertEqual(Path(second.attrs['cache_path']).name,
                    'covariance_mesh2_spectrum_poles_QSO_z0.8-2.1_holi_v3+dv_verr_nonparam.h5')
                self.assertTrue(all('verr_nonparam' in call.args[0] for call in reader.call_args_list))


if __name__ == '__main__':
    unittest.main()
