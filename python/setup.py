import glob
import os
import shutil
import tempfile
import zipfile

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

        def run(self):
            super().run()
            self._force_platlib_wheels()

        def _force_platlib_wheels(self):
            for wheel_path in glob.glob(os.path.join(self.dist_dir, "*.whl")):
                self._rewrite_wheel_metadata(wheel_path)

        @staticmethod
        def _rewrite_wheel_metadata(wheel_path):
            with tempfile.TemporaryDirectory() as temp_dir:
                with zipfile.ZipFile(wheel_path, "r") as src:
                    src.extractall(temp_dir)

                wheel_metadata_files = glob.glob(
                    os.path.join(temp_dir, "*.dist-info", "WHEEL")
                )
                if not wheel_metadata_files:
                    return

                wheel_metadata_path = wheel_metadata_files[0]
                with open(wheel_metadata_path, "r", encoding="utf-8") as f:
                    wheel_metadata = f.read()

                if "Root-Is-Purelib: true" in wheel_metadata:
                    wheel_metadata = wheel_metadata.replace(
                        "Root-Is-Purelib: true",
                        "Root-Is-Purelib: false",
                    )
                    with open(wheel_metadata_path, "w", encoding="utf-8") as f:
                        f.write(wheel_metadata)

                rewritten_wheel = wheel_path + ".rewrite"
                shutil.make_archive(
                    base_name=rewritten_wheel[:-4],
                    format="zip",
                    root_dir=temp_dir,
                )
                os.replace(rewritten_wheel[:-4] + ".zip", wheel_path)
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
