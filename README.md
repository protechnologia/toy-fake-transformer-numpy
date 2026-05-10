# toy-fake-transformer-numpy

Minimalny transformer w czystym numpy — bez PyTorcha, bez treningu, z ręcznie ustawionymi wagami.

Model dokańcza sekwencję `Ala ma kota, a kot ma Alę` przewidując kolejny token na podstawie kontekstu.

## Po co to?

Żeby zobaczyć jak transformer działa od środka — bez bibliotek, bez magii, z konkretnymi liczbami w każdym kroku. Każda funkcja ma komentarz z przykładowymi wartościami wejścia i wyjścia.

## Jak uruchomić

```bash
pip install numpy
python fake_transformer.py
```

Oczekiwany output:

```
✓  [Ala] → ma  (oczekiwano: ma)
✓  [Ala ma] → kota  (oczekiwano: kota)
✓  [Ala ma kota] → a  (oczekiwano: a)
✓  [Ala ma kota a] → kot  (oczekiwano: kot)
✓  [Ala ma kota a kot] → ma  (oczekiwano: ma)
✓  [Ala ma kota a kot ma] → Alę  (oczekiwano: Alę)
```

## Architektura

```
token + pozycja → attention (causal) → residual → LM head → next token
```

| Komponent | Wartość | Uwaga |
|---|---|---|
| Słownik | 6 tokenów | Ala, ma, kota, a, kot, Alę |
| Wymiar embeddingu | 8 | 6 token + 2 pozycja |
| Warstwy attention | 1 | Wq = Wk = Wv = I |
| Parametry | ~50 liczb | głównie W_lm (8×6) |

## Dlaczego "fake"?

Attention jest atrapą — macierze `Wq`, `Wk`, `Wv` są jednostkowe (identity). Ponieważ embeddingi tokenów są one-hot i ortogonalne, każdy token po softmaxie patrzy praktycznie tylko na siebie. Żaden kontekst nie przepływa między tokenami.

W efekcie model zachowuje się jak prosta sieć feed-forward:

```
token + pozycja  →  residual(attention)  →  W_lm  →  next token
```

Cała "inteligencja" siedzi w `W_lm` — jedynej macierzy z nauczonymi (analitycznie) wagami.

W prawdziwym transformerze `Wq`, `Wk`, `Wv` są uczone na miliardach tokenów i to właśnie one sprawiają że attention faktycznie wyciąga kontekst z sekwencji.

## Dlaczego "toy"?

Sekwencja ma 7 tokenów, słownik 6 słów, wymiar embeddingu 8. Cały model to kilkadziesiąt liczb.

## Skąd wagi?

Nie ma treningu przez gradient descent. Wagi są ustawione ręcznie lub wyznaczone analitycznie:

- `E` (embeddingi) — macierz jednostkowa, ustawiona ręcznie
- `Wq`, `Wk`, `Wv` — macierze jednostkowe, ustawione ręcznie
- `W_lm` (LM head) — wyznaczony przez `numpy.linalg.lstsq`, czyli rozwiązanie układu równań `X · W_lm = T` gdzie `X` to hidden states a `T` to oczekiwane next tokeny

## Ciekawy szczegół — kodowanie pozycyjne

Token `ma` pojawia się w sekwencji dwa razy — na pozycji 1 (po nim `kota`) i na pozycji 5 (po nim `Alę`). Samo embedding tokenu nie wystarczy żeby je rozróżnić.

Kodowanie pozycyjne sin/cos rozwiązuje to elegancko:

```
ma[pos=1] → dim6= 0.50  dim7= 0.87
ma[pos=5] → dim6= 0.50  dim7=-0.87
```

Ten sam `sin`, różny `cos` — `W_lm` może poprowadzić je w różne strony.
