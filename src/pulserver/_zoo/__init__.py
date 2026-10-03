"""The shipped scanner sequences and reconstructions, and the pairs a console reconstructs them with."""

#: The shipped reconstruction plugin a console reconstructs each shipped
#: scanner-sequence plugin with when a scan names none. The stored design and
#: the reconstruction proxy do not read it.
ZOO_PAIRS: dict[str, str] = {
    "bssfp2d": "pics",
    "bssfp3d": "pics",
    "epi2d": "epi",
    "fse3d": "pics_train",
    "gre2d": "pics",
    "gre3d": "pics",
    "gre_multiecho2d": "pics",
    "gre_multiecho3d": "pics",
    "gre_radial2d": "nufft",
    "gre_spiral2d": "nufft",
    "gre_stack_of_spirals3d": "nufft",
    "gre_stack_of_stars3d": "nufft",
    "mprage3d": "pics_train",
    "se2d": "pics",
    "se3d": "pics",
    "se_radial2d": "nufft",
    "se_spiral2d": "nufft",
    "se_stack_of_spirals3d": "nufft",
    "se_stack_of_stars3d": "nufft",
}
