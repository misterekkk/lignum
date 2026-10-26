"""Checks a linked lignum library: AVX/AVX2 (VEX-encoded) instructions may only appear in the
AVX2 build of the CatBoost kernel (namespace lignum::catboost_kernel::avx2) and in simdjson's own
AVX2 / AVX-512 kernels. Anywhere else they would run on CPUs without AVX2, e.g. an inline function
or template that the linker took from the AVX2 object for every caller.

Usage: python check_kernel_isa.py <library>
"""

import re
import subprocess
import sys

ALLOWED = re.compile(r"lignum::catboost_kernel::avx2::|simdjson::(haswell|icelake)::")
# VEX / EVEX instructions start with "v" (vmovdqu, vpaddb, vzeroupper, ...) or use ymm / zmm registers.
VEX = re.compile(r"^\s*[0-9a-f]+:\s+(v[a-z0-9]+)\b|%[yz]mm[0-9]+")


def main() -> int:
    text = subprocess.run(["objdump", "-d", "-C", "--no-show-raw-insn", sys.argv[1]], capture_output=True, text=True, check=True).stdout
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


if __name__ == "__main__":
    sys.exit(main())
