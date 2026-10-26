"""Checks a linked lignum library: AVX/AVX2 (VEX-encoded) instructions may only appear in the
AVX2 build of the CatBoost kernel (namespace lignum::catboost_kernel::avx2) and in simdjson's own
AVX2 / AVX-512 kernels. Anywhere else they would run on CPUs without AVX2, e.g. an inline function
or template that the linker took from the AVX2 object for every caller.

MSVC compiles the whole AVX2 object with /arch:AVX2 and its libraries carry no symbol names, so
there the object file itself is checked: no function it defines with external linkage outside the
avx2 namespace (e.g. a standard-library helper, which could replace the generic copy at link time)
may contain VEX instructions.

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


def msvc_vex_functions(dumpbin: str, obj: str) -> dict:
    """Function name -> its first VEX instruction, from `dumpbin /disasm` of an object file."""
    text = subprocess.run([dumpbin, "/nologo", "/disasm", obj], capture_output=True, text=True, check=True).stdout
    function, found = None, {}
    for line in text.splitlines():
        m = re.match(r"^(\S+)(?: \(.*\))?:$", line)
        if m:
            function = m.group(1)
            continue
        # "  0000000000000010: C5 FB 10 04 C8     vmovsd      xmm0,mmword ptr [rax+rcx*8]"
        m = re.match(r"^\s+[0-9A-F]+: (?:[0-9A-F]{2} )+\s+([a-z][a-z0-9]*)\b(.*)$", line)
        if function and m and (m.group(1).startswith("v") or "ymm" in m.group(2)):
            found.setdefault(function, line.strip())
    return found


def check_msvc_object(dumpbin: str, obj: str) -> int:
    text = subprocess.run([dumpbin, "/nologo", "/symbols", obj], capture_output=True, text=True, check=True).stdout
    functions = [m.groups() for m in map(COFF_FUNCTION.search, text.splitlines()) if m]
    kernel = [name for name, _ in functions if MSVC_KERNEL in name]
    if not any(name.startswith("?predict_block@avx2@") for name in kernel):
        print(f"no lignum::catboost_kernel::avx2::predict_block in {obj}: the AVX2 kernel is missing")
        return 1
    # Shared functions without VEX instructions (pointer helpers of the standard library) are
    # harmless whichever copy the linker takes; ones with VEX are not.
    vex = msvc_vex_functions(dumpbin, obj)
    if not vex:
        print(f"no VEX instruction in {obj}: was it compiled with /arch:AVX2?")
        return 1
    shared = [(name, rest.strip()) for name, rest in functions if MSVC_KERNEL not in name]
    bad = [(name, undecorated) for name, undecorated in shared if name in vex]
    for name, undecorated in bad:
        print(f"AVX in a shared function outside the AVX2 kernel: {name} {undecorated}\n    {vex[name]}")
    print(f"{len(vex)} functions with AVX in the object, {len(shared)} shared functions outside the kernel, "
          f"{len(bad)} of them with AVX")
    return 1 if bad else 0


def main() -> int:
    if sys.argv[1] == "--msvc":
        return check_msvc_object(sys.argv[2], sys.argv[3])
    return check_library(sys.argv[1])


if __name__ == "__main__":
    sys.exit(main())
