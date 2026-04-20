import glob
import os
import shutil
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


def _find_layout_root(marker):
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = []
    current = here
    while True:
        candidates.append(current)
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent

    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, marker)):
            return candidate

    raise RuntimeError(f"Unable to find project root containing {marker!r}")


def _source_root():
    return _find_layout_root("CMakeLists.txt")


def _scm_root():
    try:
        return _find_layout_root(".git")
    except RuntimeError:
        return os.path.abspath(os.path.dirname(__file__))


def _read_long_description():
    here = os.path.abspath(os.path.dirname(__file__))
    readme_candidates = [
        os.path.join(here, "README.md"),
        os.path.join(os.path.dirname(here), "README.md"),
    ]

    for readme in readme_candidates:
        if os.path.exists(readme):
            with open(readme, encoding="utf-8") as f:
                return f.read()

    return ""


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = "Debug" if self.debug else "Release"
        mkl_static_link = os.environ.get("MKL_STATIC_LINK", "ON")

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DPython3_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            "-DENABLE_PYBINDS=ON",
            "-DENABLE_TESTS=OFF",
            f"-DMKL_STATIC_LINK={mkl_static_link}",
            "-DCMAKE_CXX_FLAGS_RELEASE=-O3 -DNDEBUG -s",
        ]

        build_args = []
        if self.compiler.compiler_type != "msvc":
            jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", str(os.cpu_count() or 4))
            build_args += ["--", f"-j{jobs}"]

        env = os.environ.copy()
        build_temp = self.build_temp
        os.makedirs(build_temp, exist_ok=True)
        src_dir = _source_root()

        subprocess.check_call(
            ["cmake", src_dir] + cmake_args, cwd=build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=build_temp, env=env
        )

        lib_patterns = ["libvsag.so*", "libvsag.dylib", "*vsag*.dll"]
        for pattern in lib_patterns:
            for lib_path in glob.glob(os.path.join(build_temp, "**", pattern), recursive=True):
                shutil.copy(lib_path, extdir)


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(
    name="pyvsag",
    description="vsag is a vector indexing library used for similarity search",
    author="the vsag project",
    author_email="the.vsag.project@gmail.com",
    license="Apache-2.0",
    url="https://github.com/antgroup/vsag",
    keywords="search, nearest, neighbors",
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    long_description=_read_long_description(),
    long_description_content_type="text/markdown",
    distclass=BinaryDistribution,
    ext_modules=[CMakeExtension("pyvsag._pyvsag")],
    cmdclass={"build_ext": CMakeBuild},
    use_scm_version={
        "root": _scm_root(),
        "write_to": os.path.join(os.path.abspath(os.path.dirname(__file__)), "pyvsag", "_version.py"),
        "version_scheme": "release-branch-semver",
        "local_scheme": "no-local-version",
        "fallback_version": "0.0.0",
    },
    zip_safe=False,
)
