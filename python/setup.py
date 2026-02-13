from setuptools import setup
from setuptools.dist import Distribution

try:
    from wheel.bdist_wheel import bdist_wheel
except ImportError:
    bdist_wheel = None


class BinaryDistribution(Distribution):
    """Force wheel tagging as platform-specific when shipping prebuilt binaries."""

    def has_ext_modules(self):
        return True

    def is_pure(self):
        return False


if bdist_wheel is not None:
    class BinaryBdistWheel(bdist_wheel):
        """Mark wheel as platform specific even when binaries are shipped as package data."""

        def finalize_options(self):
            super().finalize_options()
            self.root_is_pure = False
else:
    BinaryBdistWheel = None


setup(
    use_scm_version={
        "root": "..",
        "version_scheme": "release-branch-semver",
        "local_scheme": "no-local-version",
        "write_to": "python/pyvsag/_version.py",
        "fallback_version": "0.0.0",
    },
    distclass=BinaryDistribution,
    cmdclass={"bdist_wheel": BinaryBdistWheel} if BinaryBdistWheel is not None else {},
    zip_safe=False,
)
