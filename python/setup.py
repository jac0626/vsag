from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """Force wheel tagging as platform-specific when shipping prebuilt binaries."""

    def has_ext_modules(self):
        return True


setup(
    use_scm_version={
        "root": "..",
        "version_scheme": "release-branch-semver",
        "local_scheme": "no-local-version",
        "write_to": "python/pyvsag/_version.py",
        "fallback_version": "0.0.0",
    },
    distclass=BinaryDistribution,
    zip_safe=False,
)
