# SPDX-License-Identifier: Apache-2.0
"""Platform wheel: pure-Python + prebuilt libck_kernels / libtrt_dit_plugins.

The shared libraries are built by CMake in CI and staged into
``python/trt_dit_plugins/lib/`` before ``pip wheel`` runs. The wheel tag is
forced to ``py3-none-<plat>`` so one wheel serves all Python 3 versions.
"""
from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):  # noqa: N802 - setuptools API name
        return True


try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel

    class _PlatWheel(_bdist_wheel):
        def get_tag(self):
            _, _, plat = super().get_tag()
            return ("py3", "none", plat)

    _cmd = {"bdist_wheel": _PlatWheel}
except ImportError:  # setuptools without bdist_wheel: keep default tag
    _cmd = {}

setup(distclass=BinaryDistribution, cmdclass=_cmd)
