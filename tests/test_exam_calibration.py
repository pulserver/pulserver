"""The names a calibration scan and the series after it agree on."""

import pulserver.recon as recon

NAMES = (recon.B0_MAP, recon.B1_MAP, recon.COIL_SENSITIVITIES)


def test_a_map_one_series_measures_is_read_by_the_next(tmp_path):
    """The point of naming them: two series of an exam mean the same thing."""
    calibration = recon.ExamCache("exam-1", tmp_path)
    calibration[recon.B1_MAP] = [1.0, 0.9, 0.8]
    calibration.close()

    later = recon.ExamCache("exam-1", tmp_path)
    assert later[recon.B1_MAP] == [1.0, 0.9, 0.8]
    later.close()


def test_the_maps_are_told_apart():
    assert len(set(NAMES)) == len(NAMES)


def test_every_name_is_public():
    """A plugin writing one has to be able to name it without reaching inside."""
    for name in ("B0_MAP", "B1_MAP", "COIL_SENSITIVITIES"):
        assert name in recon.__all__
