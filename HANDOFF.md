# Handoff — stan projektu i jak kontynuować

Plik dla nowej sesji (człowiek lub agent). Najpierw przeczytaj `CLAUDE.md` (zasady), potem ten plik,
potem `ARCHITECTURE.md` (jak działa silnik) i `TASKS.md` (roadmapa + wszystkie pomiary).
Stan na 2026-10-22, branch `dev/v0.2` (wypchnięty na GitHub, `main` = stary stan sprzed prac).

## Co to jest

Lignum: silnik predykcji dla modeli LightGBM / XGBoost (rdzeń C++20, Python przez nanobind, C API).
Cele, które ustaliliśmy: **wyniki bit w bit identyczne** z biblioteką źródłową, natychmiastowe
wczytanie modelu (mmap), niskie opóźnienie przy strumieniowaniu wierszy, dobra przepustowość.

## Co zrobiliśmy (chronologicznie, szczegóły w `git log` i `TASKS.md`)

1. **Porządki**: nowy branch od `main`, stary eksperyment z kategoriami zarchiwizowany na
   `dev/new-features`, R usunięty (tag `v0.1-with-R`), układ `include/lignum`, `src`, `python`,
   `tests/{cpp,python}`, jeden root `CMakeLists.txt` + `pyproject.toml`.
2. **Poprawność (bit-exact)**: znalezione i naprawione błędy — XGBoost liczy we float32 (progi,
   akumulacja, base_score, softmax), LightGBM `missing_type` None/Zero, zerowanie |x|<=1e-35,
   `sigmoid:`, `cross_entropy_lambda`, kolejność sumowania drzew. Testy porównują `assert_array_equal`.
3. **Strategie** `serial` / `rows` / `auto` (małe batche bez OpenMP, bloki liczone z liczby wątków),
   przeplatanie 4 drzew dla pojedynczych wierszy.
4. **Multiclass** (LightGBM, XGBoost), **kategorie** LightGBM i XGBoost (semantyka zweryfikowana
   empirycznie), C API + `liblignum`, walidacja plików modelu, fuzz test pod ASan/UBSan w CI.
5. **Kernel z binami** (domyślny): wejście mapowane raz na wiersz na 16-bitowe kody (biny dla cech
   numerycznych — dokładnie, kody kategorii), węzeł `BNode` = 3 poziomy drzewa w 64 B, decyzje
   kategoryczne liczone z góry (tablica kategoria -> bity decyzji; pomysł autora projektu), w drzewie
   jeden bit z L1. Zapisywany w formacie binarnym 0.6 (wczytanie 0.02 ms + walidacja 0.15-0.64 ms).
   Stary kernel (`KNode`) zostaje dla `load(..., binned=False)`; każdy test Pythona idzie przez oba.
6. **Benchmarki**: `benchmarks/bench.py` (wszyscy konkurenci), `bench-lleaves.yml` w CI (x86 + ARM,
   automatycznie przy zmianach w `src/`), zbiory rzeczywiste (`benchmarks/datasets.py`: California,
   Covertype, Adult, Bank, Diabetes) + 20 testów zgodności na nich.
7. **Tryb kompilowany (eksperyment, ostatnia rzecz)** — opis niżej.

## Gdzie jesteśmy wydajnościowo (najważniejsze liczby)

- Wobec natywnego LightGBM/XGBoost i TL2cgen: szybciej praktycznie wszędzie (2-8x).
- Wobec **lleaves** (najsilniejszy konkurent, LLVM, tylko LightGBM, nie bit-exact):
  - małe batche (1 i 100 wierszy, ustawienia domyślne): **wygrywamy 2-6x** wszędzie;
  - kategorie o wysokiej liczności (Diabetes, syntetyczne kategorie): **wygrywamy** także duże batche;
  - **duże batche na typowych danych rzeczywistych (California, Covertype, Adult, Bank): lleaves
    2-4x szybszy** — skoki w kodzie skompilowanym są tam przewidywane prawie bezbłędnie, a interpreter
    nie korzysta z przewidywalności. Wykluczone przyczyny: binowanie, multiclass, kolejność wierszy,
    przeplatanie. To był powód eksperymentu z kompilacją.

## Tryb kompilowany (`src/compiled.*`, `model.compile()`)

Jak działa: z `BinnedForest` generujemy kod C, kompilujemy równolegle systemowym `cc -O2 -fPIC`
(N plików), linkujemy do `.so` (cache po hashu źródła), `dlopen`. Wygenerowana biblioteka:
- `lignum_bin`: binowanie cech numerycznych (stałe progi, rozwinięte wyszukiwanie binarne,
  liniowe zliczanie dla <= 8 progów) — te same kody co `bin_of`;
- `lignum_block_<b>`: blok 16 drzew (`block_trees`) przechodzi przez całą porcję wierszy
  (~2 MB buforów na porcję, `LIGNUM_COMPILED_CHUNK_KB`), zanim ruszy następny blok; liście to
  stałe hex-float dodawane do sum wiersza w kolejności drzew. Kody wiersza wczytywane raz do
  zmiennych lokalnych (`int16`: NaN = -1, bez zawijania), pusty `asm` nie pozwala kompilatorowi
  przenieść odczytów z powrotem do gałęzi; para liści = select zamiast skoku.
Decyzje kategoryczne, wątki i transformacje zostają w Lignum, więc wynik jest bit-exact. Każdy
test Pythona idzie też w wariancie `compiled` (`tests/python/conftest.py`). Tylko Linux/macOS.

Wyniki na M4 (500 drzew, ms, 1 wątek / 10 wątków; kompilacja 6-12 s, lleaves 13-134 s):

| model | interpreter | skompilowany | lleaves |
|---|---:|---:|---:|
| California | 85.3 / 12.4 | **22.6 / 3.7** | 35.7 / 6.4 |
| Covertype | 1674 / 253 | 493 / 74.9 | **388 / 67.2** |
| Adult | 226 / 31.7 | **47.6 / 8.2** | 55.9 / 13.0 |
| Bank | 226 / 33.3 | **51.9 / 8.9** | 61.6 / 10.7 |
| Diabetes | 969 / 136 | **296 / 42.8** | 1508 / 273 |

CI (`bench-lleaves.yml`, commit `d6bce3f`, 4 vCPU, build generic, s: 4 wątki / 1 wątek; pełne
raporty w `benchmarks/results/ci/2026-10-22-gh-*-d6bce3f.md`, 0 niezgodności bit-exact):

| model | x86 interpreter | x86 skompilowany | x86 lleaves | ARM interpreter | ARM skompilowany | ARM lleaves |
|---|---:|---:|---:|---:|---:|---:|
| syntetyczny numeryczny | **0.412 / 0.961** | 0.414 / 1.390 | 0.422 / 1.341 | 0.331 / 1.309 | **0.213 / 0.856** | 0.334 / 1.329 |
| syntetyczny z kategoriami | 0.840 / 2.749 | **0.534 / 1.743** | 0.944 / 2.880 | 0.717 / 2.832 | **0.252 / 1.012** | 0.888 / 3.539 |
| Diabetes | 0.380 / 1.139 | **0.199 / 0.602** | 0.482 / 1.380 | 0.315 / 1.209 | **0.097 / 0.371** | 0.628 / 2.484 |
| Covertype | 0.858 / 2.167 | 0.246 / 0.646 | **0.202 / 0.496** | 0.742 / 2.775 | **0.184 / 0.707** | 0.206 / 0.740 |
| California | 0.037 / 0.108 | **0.015 / 0.043** | 0.018 / 0.050 | 0.034 / 0.134 | **0.008 / 0.033** | 0.015 / 0.058 |
| Adult | 0.090 / 0.288 | **0.028 / 0.097** | 0.034 / 0.115 | 0.080 / 0.316 | **0.016 / 0.068** | 0.025 / 0.099 |
| Bank | 0.098 / 0.281 | **0.037 / 0.078** | 0.039 / 0.081 | 0.088 / 0.331 | **0.022 / 0.069** | 0.027 / 0.087 |

Pojedynczy wiersz (1 wątek): skompilowany 1.9-2.3 µs na x86 / 3.7-4.5 µs na ARM, interpreter 5-8 /
7-11 µs, lleaves 6-15 / 10-62 µs. Wyjątek: syntetyczny numeryczny na x86 (niezależne cechy, skoki
nieprzewidywalne) — bezskokowy interpreter wygrywa na 1 wątku (0.96 vs 1.39 s) i przy 100 wierszach
(169 vs 307 µs). Kryterium wyboru trybu to więc przewidywalność skoków, nie rozmiar batcha.

Kod Covertype (`lignum_block_*`, M4): progi są natychmiastowymi w `cmp w, #imm` (30.9k, zero
odczytów z tablic), liście to stałe 64-bit budowane w kodzie (`mov` + 3x `movk` + `fmov`, 94k
`movk`), progi binowania > 8 cięć w tablicach `.rodata`. Odczyty: 1.4k odczytów kodów wiersza
(było 15.6k) + 6.1k odczytów ze stosu — zmiennych lokalnych jest więcej niż rejestrów (51 cech +
7 sum), więc kompilator je rozlewa; lleaves ma ~4.3k odczytów łącznie.

Co nie pomogło: porównania na surowych `double` (stała 64-bit = 5 instrukcji; California 38 ms
zamiast 23), `-O3`, `-mcpu=native`, bloki 8-64 (szybkość ta sama, kompilacja rośnie z rozmiarem).

**Następne kroki** (w tej kolejności):
1. Ryzen (dedykowana maszyna, `benchmarks/README.md`, `--engines lignum,lignum-compiled,lleaves`),
   także p99 pojedynczego wiersza.
2. Mniej instrukcji w drzewach: liście z tablicy w `.rodata` (`adrp`+`ldr` zamiast 5 instrukcji,
   kod ~35% mniejszy) i mniej rozlewania: bloki układane tak, by drzewa w bloku dzieliły cechy
   (dowolna kolejność między klasami w multiclass jest dozwolona, w obrębie klasy nie), albo
   lokalne tylko dla najczęściej używanych cech bloku.
3. Covertype na x86: IR lleaves tym samym clangiem = 374 ms na M4, więc to kształt kodu;
   `-mllvm -enable-tail-merge=false` daje ~4% (tylko clang; gcc: `-fno-crossjumping`).
4. Automatyczny wybór trybu: przy `compile()` zmierzyć interpreter i kod skompilowany na próbce
   (np. 10k wierszy z danych użytkownika lub modelu) i używać szybszego; osobno dla małych batchy.
5. API: zapis skompilowanej biblioteki obok `.bin` (bez kompilacji przy starcie), Windows (MSVC/clang-cl),
   ostrzeżenie gdy brak kompilatora.

## Co jeszcze otwarte (pełna lista w `TASKS.md`)

- Pomiar na dedykowanej maszynie x86 (Ryzen autora): instrukcja w `benchmarks/README.md`
  (Linux/WSL2; lleaves i TL2cgen nie działają natywnie na Windows). Benchmark raportuje p99.
- Etap 5: Arrow / zero-copy streaming (główny przypadek użycia autora: strumieniowanie z Kafki).
- Etap 6: PyPI (cibuildwheel, bez `-march=native`, dołączanie OpenMP).
- Decyzja strategiczna: tryb kompilowany jako opcja dla przepustowości vs pozycjonowanie jako
  interpreter (opóźnienie, start, dokładność, kategorie).
- Porządki przed wydaniem: README z uczciwymi benchmarkami, scalenie `dev/v0.2` -> `main`.

## Jak pracować (praktycznie)

```bash
make venv && make bench-deps         # .venv (Python 3.12, uv) + konkurenci do benchmarków
make py-build                        # instalacja edytowalna (po zmianach w C++ zawsze przebudować)
make test                            # C++ (GoogleTest) + Python (każdy test dla obu kerneli)
.venv/bin/python benchmarks/bench.py --engines lignum,lignum-compiled,lleaves \
    --only lightgbm-real-covtype,lightgbm-real-diabetes --trees 500
```

- Eksperymenty kernela mierzymy na x86 i ARM: flagi w `EXPERIMENT_CXX_FLAGS` w
  `.github/workflows/bench-lleaves.yml`, push na `dev/v0.2`, wyniki w artefaktach (~5 min).
  Sam M4 nas już kilka razy zmylił.
- Flagi eksperymentalne w kodzie: `LIGNUM_BRANCHY_TAIL/CAT` (domyślne ustalone pomiarami),
  `LIGNUM_BINNED_BRANCHY`, `LIGNUM_BINNED_MIN_ROWS`, `LIGNUM_NATIVE`, `LIGNUM_SANITIZE(RS)`.
- ASan na tym Macu (Apple clang 17, macOS 26) zawiesza się nawet na hello world — lokalnie
  `-DLIGNUM_SANITIZERS=undefined`, ASan działa w CI na Linuksie.
- GitHub API: bez logowania limit 60 zapytań/h; agent czytał logi/artefakty przez token z
  `git credential fill` (tylko odczyt).
