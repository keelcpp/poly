# keel::poly

Zero-STL C++20 numerical polynomial algorithms built on [`keel`](https://github.com/keelcpp/keel).

## Features

- **Evaluation**: Horner scalar/batch evaluation (`polyval`, `polyval_batch`, `horner_deriv`) and matrix polynomial evaluation (`polyvalm`)
- **Roots & expansion**: Durand–Kerner simultaneous root finder (`roots`) and root-to-polynomial expansion (`from_roots`, `from_roots_c64`)
- **Arithmetic & calculus**: multiplication (`mul`), Euclidean division (`div`), derivative & quotient derivative (`deriv`, `deriv_ratio`), antiderivative (`integ`)
- **Fitting**: Householder QR Vandermonde polynomial least-squares fitting (`polyfit`)
- **Approximation & rational expansion**: Padé time-delay approximants (`pade_delay`), s-domain (`residue`) and z-domain (`residuez`) partial fraction expansion with simple and repeated pole support
- Zero STL headers in public headers and implementation; full `-fno-exceptions -fno-rtti` support

## Usage

```cpp
#include <keel/poly.h>

keel::f64 p[3] = {1.0, -3.0, 2.0}; // x^2 - 3x + 2
auto r = keel::poly::roots(p);     // {1, 2}
```

## Build & Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`-fno-exceptions -fno-rtti` build:

```bash
cmake -S . -B build-noeh -DCMAKE_BUILD_TYPE=Release -DKEEL_POLY_NO_EXCEPTIONS_RTTI=ON
cmake --build build-noeh -j
ctest --test-dir build-noeh --output-on-failure
```
