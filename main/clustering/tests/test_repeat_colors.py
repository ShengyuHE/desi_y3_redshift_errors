"""Color repeat selection, histogram inversion, and catalog dispatch checks.

Run in the cosmodesi main environment:
    python -m unittest discover -s main/clustering/tests -p test_repeat_colors.py
"""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import fitsio
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import dv_tools
import cat_tools
from repeat_obs.model_repeats import save_color_repeats_hcdf


class RepeatColorTests(unittest.TestCase):
    def test_direct_selection_and_hcdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            threshold_dir = root / 'thresholds' / 'data-dr2-v2.2'
            threshold_dir.mkdir(parents=True)
            np.savez(threshold_dir / 'QSO_labels.npz', zedges=[.8, 1., 1.2, 2.1],
                     threshold_color=[2., 2., 2.])  # No TARGETID or label arrays needed.
            dtype = [(k, 'f8') for k in ['Z_QF_1', 'Z_QF_2', 'FLUX_W1', 'FLUX_R',
                                        'MW_TRANSMISSION_W1', 'MW_TRANSMISSION_R']]
            d = np.ones(8, dtype=dtype + [('PHOT_MATCH', '?')])
            d['Z_QF_1'] = [.9, .9, .9, .9, .9, .9, np.nan, .9]
            d['Z_QF_2'] = [.9, 1.1, 5., 1.1, 1.1, 1.1, 1., .7]
            d['FLUX_W1'] = [2., 3., 3., 3., 3., -1., 3., 2.]
            d['PHOT_MATCH'][3] = False
            d['FLUX_R'][4] = 0.
            fitsio.write(root / 'QSOrepeats_info.fits', d)
            kwargs = dict(repeat_dir=root, threshold_root=root / 'thresholds')
            blue, bm = dv_tools.get_color_repeats_dv('QSO', .8, 2.1, 'blue', **kwargs)
            red, rm = dv_tools.get_color_repeats_dv('QSO', .8, 2.1, 'red', **kwargs)
            all_dv, am = dv_tools.get_color_repeats_dv('QSO', .8, 2.1, 'all', **kwargs)
            self.assertEqual(len(all_dv), 7)  # Retains missing/invalid photometry.
            self.assertEqual(am['n_all'], len(all_dv))
            self.assertFalse(am['photometry_required'])
            all_fn = save_color_repeats_hcdf('QSO', .8, 2.1, 'all', root, **kwargs)
            with np.load(all_fn, allow_pickle=False) as data:
                self.assertEqual(data['counts'].sum(), 6)
                self.assertAlmostEqual(float(data['zero_fraction']), 1 / 7)
            self.assertEqual(len(blue), 3)  # Equality, negative red-band flux, either-z selection.
            self.assertEqual(len(red), 2)   # Includes discrepant pair with out-of-range mean.
            self.assertEqual(bm['n_eligible'], len(blue) + len(red))
            self.assertEqual(rm['n_eligible'], 5)
            self.assertEqual(np.count_nonzero(blue == 0), 1)
            fn = save_color_repeats_hcdf('QSO', .8, 2.1, 'blue', root, **kwargs)
            with np.load(fn, allow_pickle=False) as data:
                self.assertEqual(data['counts'].sum(), 2)
                self.assertAlmostEqual(float(data['zero_fraction']), 1 / 3)
                self.assertAlmostEqual(float(data['cdf'][-1]), 1.)
                self.assertEqual(json.loads(str(data['metadata_json']))['n_selected'], 3)

    def test_legacy_repeat_hcdf_remains_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'repeat_mode'
            directory.mkdir()
            np.savez(directory / 'HCDF_repeat_QSO_z0.8-2.1_log_abs.npz',
                     grid=[0., 1., 2.], cdf=[0., .5, 1.])
            dv = dv_tools.sample_from_cdf_v1('QSO', .8, 2.1, 101, dv_mode='repeat', cdf_dir=tmp)
            self.assertEqual(len(dv), 101)
            self.assertTrue(np.isfinite(dv).all())
            self.assertTrue((np.abs(dv) <= 100).all())
            np.testing.assert_allclose(np.sort(dv), np.sort(-dv))

    def test_inverse_sampling_and_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = root / 'DA2' / 'loa-v1' / 'repeat_mode'
            directory.mkdir(parents=True)
            for color, edges in [('blue', np.arange(4.)), ('red', np.arange(4.) + 1), ('all', np.arange(4.) + .5)]:
                np.savez(directory / f'HCDF_repeat_QSO_z0.8-2.1_log_abs_{color}.npz',
                         counts=[1, 0, 3], edges=edges, zero_fraction=.2)
            with patch.object(dv_tools, 'REPEAT_DIR', root):
                blue = cat_tools.sample_cutsky_dv('QSO', np.ones(20000), .8, 2.1, 'repeat_blue')
                red = cat_tools.sample_cutsky_dv('QSO', np.ones(20000), .8, 2.1, 'repeat_red')
                all_dv = cat_tools.sample_cutsky_dv('QSO', np.ones(20000), .8, 2.1, 'repeat_all')
                np.testing.assert_allclose(all_dv, np.sqrt(10) * blue, rtol=1e-12)
                np.testing.assert_allclose(red, 10 * blue, rtol=1e-12)
                nonzero = np.abs(blue[blue != 0])
                self.assertTrue(((nonzero <= 10) | (nonzero >= 100)).all())
                self.assertLess(abs(np.mean(blue == 0) - .2), .02)
                self.assertLess(abs(np.mean(nonzero <= 10) - .25), .02)
                np.testing.assert_allclose(np.sort(blue), np.sort(-blue))
                again = dv_tools.sample_from_cdf_v1('QSO', .8, 2.1, len(blue), dv_mode='repeat_blue')
                np.testing.assert_array_equal(blue, again)
                self.assertEqual(len(dv_tools.sample_from_cdf_v1('QSO', .8, 2.1, 3, dv_mode='repeat_red')), 3)
                self.assertEqual(len(dv_tools.sample_from_cdf_v1('QSO', .8, 2.1, 0, dv_mode='repeat_red')), 0)
            for mode in ['repeat_blue', 'repeat_red', 'repeat_all']:
                self.assertEqual(cat_tools.parse_zerr_name(mode), (mode, False))
                path = cat_tools.get_measurement_fn(domain='cutsky', tracer='QSO',
                                                    zrange=(.8, 2.1), use_dv=mode, mkdir=False)
                self.assertIn('+dv_' + mode, path)
                with self.assertRaises(ValueError): cat_tools.parse_zerr_name(mode + '_zevol')


if __name__ == '__main__':
    unittest.main()
