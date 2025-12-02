"""Test SequenceParams structure."""

from pulserver.parsing import ParamsParser


# Test case to verify correct creation and attribute assignment in ParamsParser
def test_params_parser_initialization():
    params = ParamsParser(
        function_name="design_function_1",
        FOVx=256.0,
        FOVy=256.0,
        Nx=128,
        Ny=128,
        Nslices=32,
        Nechoes=1,
        Nphases=1,
        slice_thickness=1.0,
        slice_spacing=1.5,
        Rplane=2.0,
        R=1.5,
        Rslice=1.0,
        PFfactor=0.75,
        ETL=64,
        TE=30.0,
        TE0=10.0,
        TR=2000.0,
        Tprep=100.0,
        Trecovery=500.0,
        flip=90.0,
        flip2=45.0,
        refoc_flip=180.0,
        freq_dir=1,
        freq_verse=1,
        phase_verse=1,
        bipolar_echoes=0,
        dwell=0.00001,
        raster=0.000004,
        gmax=40.0,
        smax=200.0,
        b0_field=3.0,
    )

    assert params.function_name == "design_function_1"
    assert params.FOVx == 256.0
    assert params.FOVy == 256.0
    assert params.Nx == 128
    assert params.Ny == 128
    # Add more asserts for other fields as necessary


# Test dictionary conversion and field exclusion (asdict)
def test_asdict_method():
    params = ParamsParser(
        function_name="design_function_1",
        FOVx=256.0,
        FOVy=256.0,
        Nx=128,
        Ny=128,
        Nslices=32,
    )

    param_dict = params.asdict()

    assert "function_name" not in param_dict
    assert param_dict["FOVx"] == 256.0
    assert param_dict["FOVy"] == 256.0
    assert param_dict["Nx"] == 128
    assert param_dict["Ny"] == 128
    assert (
        "slice_thickness" not in param_dict
    )  # slice_thickness is None, so it should be excluded
