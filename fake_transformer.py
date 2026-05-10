import numpy as np

# ── Słownik ──────────────────────────────────────────────────────────────────
vocab      = {"Ala": 0, "ma": 1, "kota": 2, "a": 3, "kot": 4, "Alę": 5}
id_to_word = {v: k for k, v in vocab.items()}
V = 6   # rozmiar słownika
D = 8   # wymiar embeddingu: 6 (token) + 2 (pozycja)

sequence = ["Ala", "ma", "kota", "a", "kot", "ma", "Alę"]
tokens   = [vocab[w] for w in sequence]

# ── Wagi (ręcznie) ───────────────────────────────────────────────────────────

# 1. Embeddingi tokenów: jednostkowe — każdy token to osobny ortogonalny kierunek
#    dim6–dim7 są na razie zerowe; wypełnia je kodowanie pozycyjne poniżej.
#    Tokeny są ortogonalne — żaden nie "myli się" z innym.
#
#    E wygląda tak (wiersze = tokeny, kolumny = wymiary):
#
#            d0  d1  d2  d3  d4  d5  d6  d7
#    Ala  [  1   0   0   0   0   0   0   0 ]
#    ma   [  0   1   0   0   0   0   0   0 ]
#    kota [  0   0   1   0   0   0   0   0 ]
#    a    [  0   0   0   1   0   0   0   0 ]
#    kot  [  0   0   0   0   1   0   0   0 ]
#    Alę  [  0   0   0   0   0   1   0   0 ]
E = np.zeros((V, D))
E[:, :V] = np.eye(V)

# 2. Macierze attention: jednostkowe — attention przepuszcza embedding bez zmian.
#    Wq = Wk = Wv = I (macierz 8×8):
#
#    [ 1  0  0  0  0  0  0  0 ]
#    [ 0  1  0  0  0  0  0  0 ]
#    [ 0  0  1  0  0  0  0  0 ]
#    [ 0  0  0  1  0  0  0  0 ]
#    [ 0  0  0  0  1  0  0  0 ]
#    [ 0  0  0  0  0  1  0  0 ]
#    [ 0  0  0  0  0  0  1  0 ]
#    [ 0  0  0  0  0  0  0  1 ]
#
#    UWAGA: to atrapa prawdziwego attention. Ponieważ Wq=Wk=Wv=I oraz embeddingi
#    są one-hot (ortogonalne), każdy token ma najwyższy iloczyn skalarny sam ze sobą
#    i po softmaxie patrzy praktycznie tylko na siebie:
#
#      scores[i,j] = x_i · x_j / √D  →  1/√D dla j==i,  ≈0 dla j≠i
#      attn_out ≈ x_i
#      hidden = x_i + attn_out ≈ 2 * x_i
#
#    Żadna informacja z poprzednich tokenów nie przepływa przez attention.
#    Cała "inteligencja" siedzi w W_lm, który koduje przejścia bezpośrednio.
#    Żeby attention faktycznie działało, Wq i Wk musiałyby być dobrane tak
#    żeby Q jednego tokenu miał wysoki iloczyn skalarny z K innego —
#    co wymaga nietrywialnych wag lub treningu.
Wq = np.eye(D)
Wk = np.eye(D)
Wv = np.eye(D)

# 3. LM head: wyznaczany analitycznie poniżej

# ── Kodowanie pozycyjne ──────────────────────────────────────────────────────
# Sin/cos w wymiarach 6 i 7 — kluczowe dla rozróżnienia "ma" na pozycji 1 i 5.
# Wszystkie inne tokeny mają unikalne dim0–dim5, więc pozycja jest dla nich
# bonusem, a nie koniecznością. Tylko "ma" naprawdę jej potrzebuje:
#
#   ma[1] → dim6= 0.50  dim7= 0.87   (prowadzi do "kota")
#   ma[5] → dim6= 0.50  dim7=-0.87   (prowadzi do "Alę")
#
# Ten sam sin, różny cos — wystarczy żeby W_lm poprowadził je w różne strony.
def pos_enc(pos):
    # Wejście:  pos — liczba całkowita, indeks pozycji tokenu w sekwencji
    # Wyjście:  wektor (D,) — zera wszędzie oprócz dim6 i dim7
    #   pos=0: [0.  0.  0.  0.  0.  0.  0.    1.  ]
    #   pos=1: [0.  0.  0.  0.  0.  0.  0.5   0.87]
    #   pos=5: [0.  0.  0.  0.  0.  0.  0.5  -0.87]  ← różny cos niż pos=1
    #
    # Okres 12 (π/6) dobrany tak, żeby pozycje 0–6 miały unikalne pary (sin, cos).
    p = np.zeros(D)
    p[6] = np.sin(pos * np.pi / 6)
    p[7] = np.cos(pos * np.pi / 6)
    return p

def wejście(tok_idx, pos):
    # Wejście:  tok_idx — indeks tokenu w słowniku, pos — pozycja w sekwencji
    # Wyjście:  wektor (D,) = embedding tokenu + kodowanie pozycyjne
    #   tok=0 ('Ala'), pos=0: [1.   0.   0.   0.   0.   0.   0.    1.  ]
    #   tok=1 ('ma'),  pos=1: [0.   1.   0.   0.   0.   0.   0.5   0.87]
    #   tok=1 ('ma'),  pos=5: [0.   1.   0.   0.   0.   0.   0.5  -0.87]  ← inny niż pos=1
    return E[tok_idx] + pos_enc(pos)

# ── Forward pass ─────────────────────────────────────────────────────────────
def softmax(x):
    # Wejście:  wektor scores (D,), np. [0.35, 0.35, 0.71]
    # Wyjście:  wagi sumujące się do 1,  np. [0.29, 0.29, 0.42]
    #
    # Zamienia surowe scores na wagi — wyższy score dostaje wyższą wagę.
    # Odejmujemy max(x) przed exp() dla stabilności numerycznej
    # (zapobiega overflow przy dużych wartościach).
    e = np.exp(x - np.max(x))
    return e / e.sum()

def attention(x_seq):
    # Wejście:  x_seq — lista n wektorów (D,), po jednym na token w sekwencji
    #   x_seq[0] 'Ala':  [1.   0.   0.   0.   0.   0.   0.   1.  ]
    #   x_seq[1] 'ma':   [0.   1.   0.   0.   0.   0.   0.5  0.87]
    #   x_seq[2] 'kota': [0.   0.   1.   0.   0.   0.   0.87 0.5 ]
    #
    # Wyjście:  lista n wektorów (D,) — dla każdej pozycji ważona suma V
    #   out[0] 'Ala':  [1.   0.   0.   0.   0.   0.   0.   1.  ]  (tylko siebie widzi)
    #   out[1] 'ma':   [0.4  0.6  0.   0.   0.   0.   0.3  0.92]  (lekki wyciek z Ala)
    #   out[2] 'kota': [0.26 0.3  0.44 0.   0.   0.   0.53 0.74]  (wyciek z Ala i ma)
    #
    # Causal self-attention — każda pozycja i może patrzeć tylko wstecz (j <= i),
    # nie w przód. To jest "maska" uniemożliwiająca "podglądanie" przyszłości.
    #
    # Dla każdej pozycji i:
    #   Q = x_i @ Wq                        — czego szukam?
    #   scores[j] = Q · (x_j @ Wk) / √D    — jak bardzo pasuje token j?
    #   weights = softmax(scores)            — ile uwagi poświęcam każdemu j?
    #   out = Σ weights[j] * (x_j @ Wv)     — ważona suma treści V
    #
    # W tym modelu (Wq=Wk=Wv=I, embeddingi one-hot) attention jest atrapą:
    # każda pozycja patrzy głównie na siebie, nie ciągnie kontekstu z poprzednich.
    outputs = []
    for i, xi in enumerate(x_seq):
        Q      = xi @ Wq
        scores = np.array([Q @ (x_seq[j] @ Wk) for j in range(i + 1)]) / np.sqrt(D)
        w      = softmax(scores)
        out    = sum(w[j] * (x_seq[j] @ Wv) for j in range(i + 1))
        outputs.append(out)
    return outputs

def forward(prefix_tokens):
    # Wejście:  lista indeksów tokenów, np. [0] lub [0, 1, 2]
    # Wyjście:  logits (D,) — surowe scores dla każdego tokenu słownika
    #   prefix=['Ala']         → logits=[ 0.  1.  0.  0.  0.  0.] → argmax=1 ('ma')
    #   prefix=['Ala ma kota'] → logits=[ 0.  0.  0.  1.  0.  0.] → argmax=3 ('a')
    #
    # 1. Buduj wektory wejściowe: embedding + pozycja
    # 2. Przepuść przez attention (causal)
    # 3. Residual connection: hidden = wejście + attn_output
    # 4. LM head na ostatnim tokenie: hidden[-1] @ W_lm → logits nad słownikiem
    #    argmax(logits) = przewidywany następny token
    x_seq  = [wejście(t, i) for i, t in enumerate(prefix_tokens)]
    attn   = attention(x_seq)
    # Residual connection — skip connection omijający warstwę attention:
    #
    #   hidden[i] = x_seq[i]        +  attn[i]
    #                ↑                   ↑
    #           "kim jestem"        "co zebrałem
    #        (embedding tokenu       od innych
    #         + pozycja)              tokenów"
    #
    # x_seq[i] niesie oryginalną tożsamość tokenu i zawsze przeżywa.
    # attn[i] dokłada tylko to co attention wyciągnął z kontekstu.
    hidden = [x + a for x, a in zip(x_seq, attn)]
    return hidden[-1] @ W_lm

# ── Wyznaczanie LM head analitycznie ─────────────────────────────────────────
# Budujemy macierz hidden states X (6×8) i targets T (6×6),
# rozwiązujemy X · W_lm = T — zero gradientów, czysta algebra liniowa.
#
# W_lm to jedyna "inteligencja" w tym modelu — zakodowane są w nim wszystkie
# przejścia sekwencji. Cała reszta (embedding, pozycja, attention, residual)
# to tylko przygotowanie wektora wejściowego dla W_lm.
#
# UWAGA: choć mamy architekturę transformera, model zachowuje się jak prosta FFN:
#
#   token + pozycja → x2 (residual) → W_lm → next token
#
# Ponieważ attention jest atrapą (Wq=Wk=Wv=I), żaden kontekst nie przepływa
# między tokenami. Cała magia transformera siedzi w Wq, Wk, Wv nauczonych
# na miliardach tokenów — bez tego to elegancka, ale pusta architektura.
x_seq_full = [wejście(tokens[i], i) for i in range(len(tokens) - 1)]
attn_full  = attention(x_seq_full)
hidden_full = [x + a for x, a in zip(x_seq_full, attn_full)]

X    = np.array(hidden_full)       # (6, 8)
T    = np.eye(V)[tokens[1:]]       # (6, 6) — one-hot następnych tokenów

W_lm, _, _, _ = np.linalg.lstsq(X, T, rcond=None)

# ── Test ─────────────────────────────────────────────────────────────────────
print("Kontekst → przewidywany token\n")
for i in range(1, len(tokens)):
    logits = forward(tokens[:i])
    pred   = id_to_word[np.argmax(logits)]
    ctx    = " ".join(sequence[:i])
    ok     = "✓" if pred == sequence[i] else "✗"
    print(f"  {ok}  [{ctx}] → {pred}  (oczekiwano: {sequence[i]})")

print()
print("Kodowanie pozycyjne dla 'ma' na pozycji 1 i 5:")
print(f"  pos=1: sin={np.sin(1*np.pi/6):.3f}  cos={np.cos(1*np.pi/6):.3f}")
print(f"  pos=5: sin={np.sin(5*np.pi/6):.3f}  cos={np.cos(5*np.pi/6):.3f}")
print("  → ten sam sin, różny cos — stąd model je odróżnia")
