"""Checks a linked lignum library: AVX/AVX2 (VEX-encoded) instructions may only appear in the
AVX2 build of the CatBoost kernel (namespace lignum::catboost_kernel::avx2) and in simdjson's own
AVX2 / AVX-512 kernels. Anywhere else they would run on CPUs without AVX2, e.g. an inline function
or template that the linker took from the AVX2 object for every caller.

MSVC compiles the whole AVX2 object with /arch:AVX2 and its libraries carry no symbol names, so
there the object file itself is checked: every function it defines with external linkage must be
in the avx2 namespace (any other one, e.g. an instantiated std::fill, could replace the generic
copy at link time).

Usage: python check_kernel_isa.py <library>
       python check_kernel_isa.py --msvc <dumpbin> <catboost_kernel_avx2.obj>
"""

import re
import subprocess
import sys

ALLOWED = re.compile(r"lignum::catboost_kernel::avx2::|simdjson::(haswell|icelake)::")
# VEX / EVEX instructions start with "v" (vmovdqu, vpaddb, vzeroupper, ...) or use ymm / zmm registers.
VEX = re.compile(r"^\s*[0-9a-f]+:\s+(v[a-z0-9]+)\b|%[yz]mm[0-9]+")
# dumpbin /symbols: "01A 00000000 SECT6  notype ()    External     | ?name@... (undecorated)"
COFF_FUNCTION = re.compile(r"\bSECT[0-9A-F]+\s+notype \(\)\s+External\s+\|\s+(\S+)(.*)$")
MSVC_KERNEL = "@avx2@catboost_kernel@lignum@@"


def check_library(library: str) -> int:
    text = subprocess.run(["objdump", "-d", "-C", "--no-show-raw-insn", library], capture_output=True, text=True, check=True).stdout
    function, bad, kernel = None, {}, 0
    for line in text.splitlines():
        m = re.match(r"^[0-9a-f]+ <(.*)>:$", line)
        if m:
            function = m.group(1)
            continue
        if function and VEX.search(line):
            if ALLOWED.search(function):
                kernel += function.startswith("lignum::catboost_kernel::avx2::") or "catboost_kernel::avx2" in function
            else:
                bad.setdefault(function, line.strip())
    if not kernel:
        print("no AVX2 instruction in lignum::catboost_kernel::avx2: the AVX2 kernel is missing or not vectorised")
        return 1
    for function, line in bad.items():
        print(f"AVX outside the AVX2 kernel: {function}\n    {line}")
    print(f"{kernel} AVX instructions in the AVX2 kernel, {len(bad)} functions with AVX elsewhere")
    return 1 if bad else 0


def check_msvc_object(dumpbin: str, obj: str) -> int:
    text = subprocess.run([dumpbin, "/nologo", "/symbols", obj], capture_output=True, text=True, check=True).stdout
    functions = [m.groups() for m in map(COFF_FUNCTION.search, text.splitlines()) if m]
    kernel = [name for name, _ in functions if MSVC_KERNEL in name]
    bad = [(name, rest.strip()) for name, rest in functions if MSVC_KERNEL not in name]
    if not any(name.startswith("?predict_block@avx2@") for name in kernel):
        print(f"no lignum::catboost_kernel::avx2::predict_block in {obj}: the AVX2 kernel is missing")
        return 1
    for name, undecorated in bad:
        print(f"external function outside the AVX2 kernel: {name} {undecorated}")
    print(f"{len(kernel)} external functions in the AVX2 kernel, {len(bad)} elsewhere")
    return 1 if bad else 0


def main() -> int:
    if sys.argv[1] == "--msvc":
        return check_msvc_object(sys.argv[2], sys.argv[3])
    return check_library(sys.argv[1])


if __name__ == "__main__":
    sys.exit(main())
