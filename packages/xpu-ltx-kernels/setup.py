"""Build the XPU kernel extension with icpx (SYCL/ESIMD).

The build is driven by icpx directly, not torch's ``cpp_extension``: torch's
SYCL build path forces ``-fsycl-host-compiler=c++`` and its own header order,
which fights the machine's SYCL runtime layout. Our recipe (proven on Arc Pro
B60 with oneAPI 2026.1 + torch 2.13.0+xpu):

1. Compile each ``.sycl`` source with ``icpx -fsycl -c`` against the SYCL
   headers of the *active* Python environment (``<sys.prefix>/include``), which
   must match the SYCL runtime that torch's XPU wheel bundles.
2. Device-link all objects into one extension with ``icpx -shared -fsycl``
   (this step packages the device image; skipping it makes the extension
   segfault at kernel submission).

Runtime requirements (see AGENTS.md): ``source /opt/intel/oneapi/setvars.sh``,
``SYCL_DEVICE_FILTER=level_zero:gpu``, and a Python env whose torch XPU wheel
shares the same SYCL runtime as the compiler.
"""

import os
import subprocess
import sys
from pathlib import Path

import setuptools
from setuptools.command.build_ext import build_ext

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "csrc"


def _which(*names: str) -> str:
    for n in names:
        p = Path(n)
        if p.is_file():
            return str(p)
        for d in os.environ.get("PATH", "").split(os.pathsep):
            cand = Path(d) / n
            if cand.is_file() and os.access(cand, os.X_OK):
                return str(cand)
    raise RuntimeError(f"compiler not found: {names}")


def _find_sycl_home() -> Path:
    """SYCL_HOME = the pip intel-sycl-rt install that torch's XPU wheel bundles."""
    try:
        import importlib.metadata as md  # noqa: PLC0415

        dist = md.distribution("intel-sycl-rt")
        for f in dist.files:
            if f.name == "libsycl.so":
                p = Path(f.locate()).parent.resolve()
                return p.parent
    except Exception:
        pass
    return Path("/opt/intel/oneapi/compiler/2026.1")


class SyclBuildExt(build_ext):
    """Custom build_ext that compiles SYCL sources with icpx."""

    def run(self) -> None:
        import sysconfig  # noqa: PLC0415
        import torch  # noqa: PLC0415

        self.torch_include = Path(torch.__file__).parent / "include"
        # uv venvs ship no Python headers; resolve via sysconfig so the base
        # interpreter's include dir is used.
        self.python_include = Path(sysconfig.get_path("include"))
        self.sycl_home = Path(os.environ.get("SYCL_HOME", str(_find_sycl_home())))
        self.torch_lib = Path(torch.__file__).parent / "lib"
        super().run()

    def build_extension(self, ext) -> None:  # type: ignore[override]
        icpx = _which("icpx")
        out_dir = Path(self.build_temp)
        out_dir.mkdir(parents=True, exist_ok=True)
        objects = []
        for source in ext.sources:
            src = ROOT / source
            obj = out_dir / (Path(source).stem + ".o")
            self._compile(icpx, src, obj)
            objects.append(obj)
        so = Path(self.get_ext_fullpath(ext.name))
        so.parent.mkdir(parents=True, exist_ok=True)
        self._link(icpx, objects, so)

    def _flags(self) -> list[str]:
        flags = [
            "-O3",
            "-std=c++17",
            "-fsycl",
            "-fsycl-targets=spir64_gen,spir64",
            "-sycl-std=2020",
            "-fPIC",
        ]
        for inc in (self.torch_include, self.torch_include / "torch/csrc/api/include",
                    self.sycl_home / "include", self.sycl_home / "include/sycl",
                    self.python_include):
            flags += ["-isystem", str(inc)]
        return flags

    def _compile(self, icpx: str, src: Path, obj: Path) -> None:
        cmd = [icpx, *self._flags(), f"-DTORCH_EXTENSION_NAME={self.extensions[0].name.split('.')[-1]}",
               "-x", "c++", "-c", str(src), "-o", str(obj)]
        self._run(cmd)

    def _link(self, icpx: str, objects: list[Path], so: Path) -> None:
        libs = ["torch_python", "torch", "torch_cpu", "c10", "torch_xpu", "c10_xpu"]
        cmd = [icpx, "-shared", "-fPIC", "-fsycl", "-o", str(so), *map(str, objects),
               f"-L{self.torch_lib}", *(f"-l{l}" for l in libs),
               f"-L{self.sycl_home}/lib", "-lsycl",
               f"-Wl,-rpath,{self.torch_lib}", f"-Wl,-rpath,{self.sycl_home}/lib"]
        self._run(cmd)

    def _run(self, cmd: list[str]) -> None:
        env = dict(os.environ)
        env.setdefault("SYCL_DEVICE_FILTER", "level_zero:gpu")
        print("  " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True, env=env)


sources = [str(p.relative_to(ROOT)) for p in sorted(SRC.rglob("*.sycl"))]
if not sources:
    raise SystemExit(f"no .sycl sources found under {SRC}")

ext = setuptools.Extension(name="xpu_ltx_kernels._C", sources=sources)

setuptools.setup(
    cmdclass={"build_ext": SyclBuildExt},
    ext_modules=[ext],
)