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

Jak działa: z `BinnedForest` generujemy kod C (każde drzewo = funkcja z zagnieżdżonymi `if` na
stałych progach; porównania na kodach binów i bitach decyzji), dzielimy na N plików, kompilujemy
równolegle systemowym `cc -O2 -fPIC`, linkujemy do `.so` (cache po hashu źródła), `dlopen`.
Binowanie, wątki i sumowanie liści zostają w Lignum, więc wynik jest bit-exact (test w
`tests/python/test_parity.py::test_compiled_mode_is_bit_identical`). Tylko Linux/macOS.

Wyniki na M4 (500 drzew, duży batch; kompilacja ~3.5-5 s vs lleaves 13-134 s):

| model | interpreter | skompilowany | lleaves |
|---|---:|---:|---:|
| Diabetes | 0.138 s | **0.065 s** | 0.273 s |
| syntetyczny z kategoriami | 0.306 s | **0.191 s** | 0.312 s |
| Adult | 0.032 s | 0.018 s | **0.010 s** |
| Bank | 0.033 s | 0.021 s | **0.011 s** |
| California | 0.012 s | 0.011 s | **0.006 s** |
| Covertype | 0.258 s | 0.229 s | **0.073 s** |
| syntetyczny numeryczny | **0.105 s** | 0.212 s | 0.128 s |

Wniosek: kompilacja daje 1.4-2x na danych rzeczywistych i jest najlepsza przy kategoriach, ale na
Covertype/California/Adult/Bank nadal 2-3x za lleaves. **Następne kroki do sprawdzenia** (w tej
kolejności):
1. Narzut wywołań: dziś każdy wiersz = wywołanie `lignum_eval` -> N funkcji części -> ~500 funkcji
   drzew (najpewniej nie inline'owane) + zapis 500 indeksów liści do bufora i ich odczyt. Spróbować:
   sumowanie liści w wygenerowanym kodzie (stałe double w kolejności drzew — nadal bit-exact),
   `static inline`/jedna funkcja na część, kilka wierszy na wywołanie.
2. Porównania na surowych `double` zamiast kodów binów dla cech numerycznych (jak lleaves: bez
   binowania; NaN: `!(x < t)` gdy NaN idzie w prawo, `x >= t` gdy w lewo — dokładne IEEE).
3. `-O3`, `-march=native`, porównać z tym, co generuje lleaves (`fblocksize`, kolejność row/tree).
4. Jeśli dogonimy: API (`lignum.compile` / zapis skompilowanej biblioteki obok `.bin`), Windows.

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
