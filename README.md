# Procesamiento-del-Lenguaje-Natural-UBA

## Evaluacion
- [ ] Examen escrito — 5 pts — **24/06**
- [ ] TP (comparativa de tecnicas sobre dataset clasico) — 2 pts — entrega **01/07**
- [ ] Presentacion de paper (individual, ~20 min) — 5 pts — a coordinar

> El TP consiste en tomar un problema clasico (ej: sentiment en Yelp) y resolverlo con multiples tecnicas en orden historico: BoW → fasttext → RNN → BERT finetuneado → LLM. Entregable: una pagina con descripcion + tabla/grafico comparativo.

## Cronograma actualizado (mail 05/05)

| Fecha | Clase | Notas |
|-------|-------|-------|
| 13/05 | Consultas / Repaso general | [meet.google.com/dgn-gjbe-kie](https://meet.google.com/dgn-gjbe-kie) |
| 20/05 | Repaso de todo lo asignado | [clase1](teoria/clase1.md) · [clase2](teoria/clase2.md) · [clase3](teoria/clase3.md) · [clase4](teoria/clase4.md) |
| 27/05 | RNN, LSTMs, Transformers y Atención | [clase4](teoria/clase4.md) · [clase5](teoria/clase5.md) |
| 03/06 | BERT, embeddings contextuales, GPT, T5 | [clase6](teoria/clase6.md) |
| 10/06 | _sin clase — trabajar en el TP_ | |
| 17/06 | Ajuste Fino (LoRA, Adapters). Alineación y Razonamiento | [clase7](teoria/clase7.md) |
| 24/06 | **Examen** | |
| 01/07 | Posible recuperatorio + entrega de TPs | |

## Contenido de estudio (autoestudio durante paro)

- [x] [clase1](teoria/clase1.md) — Pre-procesamiento y Modelos Probabilisticos (Tokenizacion, N-Grams, Smoothing, Perplexity)
- [x] [clase2](teoria/clase2.md) — Vectorizacion y Clasificacion Clasica (BoW, TF-IDF, BM25, Naive Bayes, SVM, LSA)
- [ ] [clase3](teoria/clase3.md) — Word Embeddings estaticos (Word2Vec, GloVe, FastText)
- [ ] [clase4](teoria/clase4.md) — Redes Recurrentes y ELMo (RNN, LSTM, GRU, ELMo)
- [ ] [clase5](teoria/clase5.md) — Transformers y Atencion (Self-attention, Multi-head, Positional encoding)
- [ ] [clase6](teoria/clase6.md) — Pre-entrenamiento y Modelos Modernos (BERT, GPT, T5)
- [ ] [clase7](teoria/clase7.md) — Ajuste Fino y Alineacion (LoRA, Adapters, RLHF, DPO, CoT)

## Datasets de referencia historicos (para el TP)
- **IMDb / Yelp**: clasificacion de sentimiento
- **Reuters-21578**: clasificacion de noticias por categoria
- **20 Newsgroups**: clustering y clasificacion de foros
- **Penn Treebank (WSJ)**: POS tagging y analisis sintactico
- **Brown Corpus**: analisis linguistico general

## Bibliografía
- [Speech and Language Processing - Jurafsky & Martin (SLP3)](https://web.stanford.edu/~jurafsky/slp3/)
- [NLTK Book (practico)](https://www.nltk.org/book/)
