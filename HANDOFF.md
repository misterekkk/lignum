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

Co nie pomogło: porównania na surowych `double` (stała 64-bit = 5 instrukcji; California 38 ms
zamiast 23), `-O3`, `-mcpu=native`, bloki 8-64 (szybkość ta sama, kompilacja rośnie z rozmiarem).

**Następne kroki**:
1. Wyniki z CI (x86 + ARM, `bench-lleaves.yml` mierzy teraz też `lignum-compiled`) i z Ryzena.
2. Covertype (~20% za lleaves): IR lleaves skompilowany tym samym clangiem daje 374 ms, więc to
   kształt naszego kodu. `-mllvm -enable-tail-merge=false` daje ~4% (tylko clang).
3. Próg interpreter/kompilowany: na M4 kompilowany wygrywa już od 1 wiersza (ciepły cache);
   sprawdzić p99 dla pojedynczego wiersza z zimnym kodem na x86.
4. Jeśli się utrzyma: API (zapis skompilowanej biblioteki obok `.bin`), Windows.

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
