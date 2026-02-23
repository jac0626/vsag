import os
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

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
            build_args += ["--", "-j4"]

        env = os.environ.copy()
        build_temp = self.build_temp
        os.makedirs(build_temp, exist_ok=True)
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        subprocess.check_call(
            ["cmake", src_dir] + cmake_args, cwd=build_temp, env=env
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=build_temp, env=env
        )

        import glob
        import shutil
        for lib in glob.glob(os.path.join(build_temp, "**", "libvsag.so*"), recursive=True) + glob.glob(os.path.join(build_temp, "**", "libvsag.dylib"), recursive=True):
            shutil.copy(lib, extdir)

        # Generate pyi stubs for the extension
        lib_dir = os.path.dirname(extdir)
        stub_env = os.environ.copy()
        stub_env["PYTHONPATH"] = lib_dir + os.pathsep + stub_env.get("PYTHONPATH", "")
        # Add extdir to LD_LIBRARY_PATH so _pyvsag.so can find libvsag.so during import
        stub_env["LD_LIBRARY_PATH"] = extdir + os.pathsep + stub_env.get("LD_LIBRARY_PATH", "")

        # Create a temporary __init__.py so Python treats `extdir` as a module
        init_py_path = os.path.join(extdir, "__init__.py")
        created_init = False
        if not os.path.exists(init_py_path):
            with open(init_py_path, "w") as f:
                f.write("\n")
            created_init = True

        try:
            print("Generating pyi stubs using pybind11-stubgen...")
            subprocess.check_call(
                [sys.executable, "-m", "pybind11_stubgen", "pyvsag._pyvsag", "-o", lib_dir],
                env=stub_env
            )
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to generate .pyi type hints: {e}")
        finally:
            if created_init and os.path.exists(init_py_path):
                os.remove(init_py_path)

class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True

def _read_version():
    here = os.path.dirname(os.path.abspath(__file__))
    _version_file = os.path.join(here, 'pyvsag', '_version.py')
    
    if os.path.exists(_version_file):
        ns = {}
        with open(_version_file) as f:
            exec(f.read(), ns)
        return ns.get('__version__')
    return '0.0.0'

setup(
    name="pyvsag",
    version=_read_version(),
    distclass=BinaryDistribution,
    ext_modules=[CMakeExtension("pyvsag._pyvsag")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
