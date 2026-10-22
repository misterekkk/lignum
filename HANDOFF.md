# Handoff — stan projektu i jak kontynuować

Dla nowej sesji (człowiek lub agent). Kolejność czytania: `CLAUDE.md` (zasady), ten plik,
`ARCHITECTURE.md` (jak działa silnik), `TASKS.md` (roadmapa + pomiary), `git log` (szczegóły).
Stan na 2026-10-22 (późny wieczór), branch `dev/v0.2` (wypchnięty). `main` = stary stan sprzed prac
plus sam plik `.github/workflows/wheels.yml` (żeby GitHub pokazywał „Run workflow”).

## Co to jest

Lignum: silnik predykcji dla modeli LightGBM / XGBoost (rdzeń C++20, Python przez nanobind, C API).
Cele: **wyniki bit w bit identyczne** z biblioteką źródłową, natychmiastowe wczytanie (`.bin` przez
mmap), niskie opóźnienie, wysoka przepustowość. Na PyPI jako **`pylignum`**, import `import lignum`
(nazwa `lignum` na PyPI należy do niezwiązanego loggera JSON).

## Co jest zrobione (szczegóły w `TASKS.md` i `git log`)

- **Poprawność**: semantyka LightGBM / XGBoost (float32, missing None/Zero, strefa zera, kategorie,
  kolejność sumowania). Każdy test Pythona idzie przez 3 kernele: `binned`, `direct`, `compiled`
  (`tests/python/conftest.py`).
- **Interpreter**: kernel z binami (domyślny, `BNode` = 3 poziomy w 64 B) i bezpośredni (`KNode`).
- **Tryb kompilowany** (`src/compiled.*`): drzewa → C → biblioteka współdzielona. Bloki 16 drzew po
  porcjach wierszy ~2 MB, liście w tablicach, wygenerowane binowanie, najczęstsze cechy bloku w
  rejestrach (`LIGNUM_COMPILED_LOCALS`: 16 ARM / 10 x86), pary liści jako select. Biblioteka obok
  modelu: `compile(library=...)`, `load(..., compiled=...)` (~2 ms), odcisk modelu + ABI.
  `compile(fallback=True)`, `autotune(X)` → `compiled_min_rows`. Linux, macOS, **Windows** (cl przez
  vswhere/vcvarsall, MinGW gcc, clang; LoadLibrary).
- **Formaty**: LightGBM `model.txt` i JSON, XGBoost JSON i `.ubj`, `.bin`; `format="auto"` rozpoznaje
  po zawartości. Drzewa liniowe i random forest LightGBM są odrzucane (wcześniej JSON liczył je źle).
- **Arrow** (C Data / C Stream Interface, bez libarrow): `predict()` przyjmuje pyarrow / polars /
  wszystko z `__arrow_c_array__` / `__arrow_c_stream__`; kernel z binami binuje kolumny prosto z
  buforów. C API: `LignumModelPredictArrow`.
- **C API**: wczytanie, predykcja (macierz i Arrow), tryb kompilowany.
- **Wheele** (`.github/workflows/wheels.yml`, cibuildwheel v4.2.1): manylinux_2_28 x86_64/aarch64,
  macOS 15+ arm64/x86_64, Windows x64, sdist; Python 3.10, 3.11 + abi3 dla 3.12+. OpenMP: auditwheel
  (libgomp), delvewheel (vcomp140), macOS: `@rpath` najpierw Homebrew (ta sama kopia co LightGBM /
  XGBoost — dwie kopie = `OMP: Error #15`), potem kopia w `lignum/.dylibs`.
- **Dokumentacja**: strona Starlight (Astro) w `docs/` — przewodniki, koncepcje, benchmarki,
  referencja API generowana z docstringów (`docs/scripts/api_reference.py`). Deploy na GitHub Pages
  z `main` (`.github/workflows/docs.yml`). Logo: propozycja A (drzewo binarne) w `docs/assets/`.

## Wyniki (pełne tabele: `docs/src/content/docs/benchmarks.md`, `benchmarks/results/`)

- Tryb kompilowany vs lleaves (500 drzew, dane rzeczywiste): szybszy na wszystkich 7 modelach na ARM
  i na x86 (Covertype na 1 wątku remis); M4: szybszy wszędzie poza Covertype na 1 wątku (~5%).
  Diabetes (kategorie) 5-6x. Kompilacja 4-12 s vs lleaves 13-280 s. lleaves nie jest bit-exact.
- Pojedynczy wiersz: 2-9 µs skompilowany, 5-13 µs interpreter.
- Arrow: interpreter tak szybki jak z numpy; skompilowany Covertype +2.5%.
- Wczytanie: `model.txt` szybszy niż JSON (Covertype 7 vs 14 ms), `.ubj` 1.7x wolniejszy niż JSON
  (jednorazowe, nie warto optymalizować), `.bin` < 1 ms.

## W toku przy przekazaniu

- **Publikacja na TestPyPI**: ręczny run „Wheels” z `target=testpypi` na `dev/v0.2` (run 39833456450,
  uruchomiony ponownie po tym, jak mój push anulował pierwszy). Sprawdzić wynik w Actions; potem
  instalacja w czystym venv:
  `pip install --pre -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ pylignum`.
- Konfiguracja PyPI jest gotowa: pending publishers na pypi.org i test.pypi.org (`pylignum`,
  `misterekkk/lignum`, `wheels.yml`, środowiska `pypi` / `testpypi`), środowiska w repo (`pypi` z
  wymaganym zatwierdzeniem przez właściciela).

## Następne kroki (ustalone z autorem)

1. **TestPyPI** → sprawdzić instalację na Linux/macOS/Windows → wydanie **0.2.0**: `__version__ =
   "0.2.0"` w `python/lignum/__init__.py` (jedyne źródło wersji), scalić `dev/v0.2` → `main`, run
   „Wheels” z `target=pypi`, zatwierdzić w środowisku `pypi`. Każda wersja może trafić na (Test)PyPI
   tylko raz — testy na `0.2.0.devN`.
2. **README** (główny, `README.md`) do przepisania: jest nieaktualny (stare porównania z Treelite,
  „Quick Local Setup”). Ma być krótki: logo, `pip install pylignum`, przykład, 3-4 punkty, mała
  tabela benchmarków, link do strony. `python/README.md` (opis na PyPI) jest aktualny.
3. **Strona**: przejrzeć treść z autorem, po scaleniu do `main` sprawdzić deploy. Lokalnie:
   `cd docs && npm install && python scripts/api_reference.py && npm run dev`.
4. **Pomiary w chmurze (Hetzner)**: CCX23/CCX33 (dedykowane vCPU, AMD EPYC) — stabilne wyniki x86;
   CAX (Ampere Altra, ARM, współdzielone vCPU) — sprawdzenie ARM. Na świeżym Ubuntu 24.04:
   `curl -LsSf https://raw.githubusercontent.com/misterekkk/lignum/dev/v0.2/benchmarks/cloud.sh | bash`
   (~30-40 min, raporty w `~/lignum-bench-*.tar.gz`, `scp` do domu). Serwer usunąć po testach
   (wyłączony też jest rozliczany). Szczególnie ciekawe: `autotune` na modelu syntetycznym (na EPYC
   9V45 interpreter wygrywał z kodem skompilowanym), p99 pojedynczego wiersza, build `LIGNUM_NATIVE`.
5. Ryzen autora (Windows): na razie bez WSL; tryb kompilowany działa natywnie (VS Build Tools z
   „Desktop development with C++”), lleaves / TL2cgen tylko na Linuksie.

## Otwarte (mniej pilne, pełna lista w `TASKS.md`)

- Covertype na 1 wątku ~5% za lleaves (kształt kodu; `-mllvm -enable-tail-merge=false` daje ~4%,
  tylko clang).
- macOS < 15: własny build libomp z niższym `MACOSX_DEPLOYMENT_TARGET` zamiast Homebrew.
- Wheele musllinux, Windows ARM; drzewa liniowe / random forest LightGBM (gdyby ktoś potrzebował).
- FlatBuffers: decyzja — nie w rdzeniu (aplikacja czyta swój schemat i podaje `double*` / Arrow).

## Pułapki, na które już wpadliśmy

- **Push zmian w `src/`, `python/`, `CMakeLists.txt` uruchamia „Wheels”**. Kolejka jest już
  rozdzielona (push nie anuluje ręcznego runu), ale nie pushować w trakcie publikacji bez potrzeby.
- „Run workflow” jest widoczny tylko dla workflow obecnych na domyślnej gałęzi (`main`).
- Windows: stos 1 MB (rekurencja w parserach musi mieć limity i wykrywanie cykli); `LoadLibrary`
  dopisuje `.dll` do nazwy bez rozszerzenia; pliki testowe zapisywać w trybie binarnym.
- Apple libc++ nie ma `std::from_chars` dla `double` — liczby z tekstu idą przez simdjson.
- macOS: dwie kopie libomp w procesie = crash; wheel bierze najpierw Homebrew.
- Sam M4 mylił przy decyzjach o kernelu — mierzyć też x86 i ARM (`bench-lleaves.yml`,
  `EXPERIMENT_CXX_FLAGS`). Runnery GitHuba są współdzielone: porównywać silniki w obrębie joba.
- ASan na tym Macu się zawiesza — lokalnie `-DLIGNUM_SANITIZERS=undefined`, ASan w CI.
- GitHub API: agent czyta logi / artefakty i uruchamia workflow tokenem z `git credential fill`.

## Jak pracować

```bash
make venv && make bench-deps      # .venv (Python 3.12, uv) + konkurenci do benchmarków
make py-build                     # instalacja edytowalna (po zmianach w C++ zawsze przebudować)
make test                         # C++ (GoogleTest) + Python (każdy test w 3 kernelach)
.venv/bin/python benchmarks/bench.py --engines lignum,lignum-compiled,lleaves \
    --only lightgbm-real-covtype,lightgbm-real-diabetes --trees 500
.venv/bin/python benchmarks/bench_arrow.py
cd docs && npm run build          # strona (najpierw python scripts/api_reference.py)
```
