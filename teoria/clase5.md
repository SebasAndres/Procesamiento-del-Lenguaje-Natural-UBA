# Language Models II: Atención y Transformers
03-06

Ref: Vaswani et al. (2017). *Attention is All You Need*. / Devlin et al. (2019). *BERT*. / Yang et al. (2016). *Hierarchical Attention Networks for Document Classification*.

---

## Mecanismo de Atención

Las RNNs seq2seq tienen un **cuello de botella**: el encoder debe comprimir toda la secuencia fuente en un único vector de tamaño fijo. Para secuencias largas, mucha información se pierde.

La atención surge para resolver esto: en lugar de pasar solo el último hidden state al decoder, el decoder puede "mirar" **todos** los hidden states del encoder y ponderar cuáles son más relevantes en cada paso.

### Intuición

No todas las palabras de una oración contribuyen igualmente a cada decisión:

```
"El hotel está buenisimo"  →  Positivo
  0.01  0.03  0.01  0.95
```

El modelo aprende a concentrarse en "buenísimo" para clasificar sentimiento.

### Mecanismo formal (seq2seq con atención)

Sea $h_1, \ldots, h_T$ los hidden states del encoder y $a_i$ el estado actual del decoder en el paso $i$:

**1. Scores** — similitud entre el estado del decoder y cada estado del encoder:

$$s_{ij} = a_i \cdot h_j$$

**2. Pesos de atención** — normalizar con softmax:

$$w_i = \text{softmax}([s_{i1}, s_{i2}, \ldots, s_{iT}])$$

**3. Vector de contexto** — suma ponderada de los valores del encoder:

$$Y_i = w_i \cdot [h_1, h_2, \ldots, h_T] = \sum_j w_{ij} \cdot h_j$$

$Y_i$ es una suma de los hidden states del encoder, ponderada por la similitud que tienen con el estado actual del decoder. Se usa para generar la salida del paso $i$.

### Ventajas

- **Resuelve el cuello de botella**: el decoder accede a toda la secuencia fuente.
- **Ayuda con el vanishing gradient**: conexiones directas desde cada paso del encoder.
- **Interpretabilidad**: los pesos $w_{ij}$ muestran qué palabras fuente influyen en cada palabra generada.
- **Captura dependencias a largo plazo**.

### Variantes

- **Hierarchical Attention**: genera embeddings o representaciones para múltiples inputs (p.ej., palabras dentro de oraciones, oraciones dentro de documentos).
- **Attention as Input Selector**: determina cuál de varios inputs es el más importante para tomar la decisión (no secuencial).

---

## Transformers

Vaswani et al. (2017) — *Attention is All You Need*.

**Idea central**: reemplazar por completo las RNNs con mecanismos de atención. Cada token calcula su representación ponderando al **resto de los tokens** en la secuencia directamente.

```
"Voy a guardar el queso en el [?]"
→ El modelo puede relacionar directamente "queso" con "[?]" sin pasar por tokens intermedios
```

### Ventajas sobre RNNs

| Propiedad | RNN | Transformer |
|-----------|-----|-------------|
| Interacción entre tokens distantes | Indirecta (crece con distancia) | Directa (un paso) |
| Procesamiento | Secuencial | Paralelo |
| Dependencias largas | Difícil (vanishing gradient) | Natural (atención directa) |
| Escalabilidad | Limitada | Alta |

---

## Positional Embeddings

Los Transformers procesan todos los tokens en paralelo → no tienen noción inherente del orden. Sin embargo, el orden importa: "Juan golpeó a Pedro" ≠ "Pedro golpeó a Juan".

**Solución**: sumar una **codificación posicional** al embedding de cada token antes de ingresar al modelo.

**Codificación sinusoidal** (Vaswani et al.):

$$\text{PE}(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d}}\right)$$

$$\text{PE}(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

Donde $pos$ es la posición en la secuencia y $d$ la dimensión del embedding.

**Ventajas:** puede generalizarse a secuencias de cualquier longitud (a diferencia de embeddings posicionales aprendidos con longitud fija).

---

## Self-Attention

En self-attention, cada token atiende a **todos los demás tokens de la misma secuencia** (incluyéndose a sí mismo).

### Queries, Keys y Values

Cada token genera tres vectores mediante proyecciones lineales aprendidas:

$$Q = X \cdot M_q \qquad K = X \cdot M_k \qquad V = X \cdot M_v$$

- **Query (Q)**: "¿qué estoy buscando?"
- **Key (K)**: "¿qué ofrezco para que me busquen?"
- **Value (V)**: "¿qué contenido tengo?"

### Fórmula de Self-Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

El factor $\sqrt{d_k}$ escala los productos punto para evitar que el softmax se sature en regiones de gradiente muy pequeño cuando $d_k$ es grande.

**Proceso por token:**
1. Producto escalar $Q_i \cdot K_j$ para todos los $j$ → scores $[s_{i1}, \ldots, s_{in}]$
2. Softmax → pesos de atención $[w_{i1}, \ldots, w_{in}]$
3. Suma ponderada de values: $Y_i = \sum_j w_{ij} V_j$

### Feed-Forward Layer

Después de self-attention, una red feed-forward aplica transformaciones no lineales posición a posición:

$$\text{FFN}(x) = \max(0, x W_1 + b_1) W_2 + b_2$$

Es necesaria porque la self-attention es lineal en los values; la FFN introduce no-linealidades.

---

## Multiheaded Self-Attention

Un solo mecanismo de atención puede no capturar todos los tipos de relaciones relevantes. Con **multiple heads**, el modelo aprende varias proyecciones $Q, K, V$ en paralelo:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \cdot W^O$$

$$\text{head}_i = \text{Attention}(Q M_q^i,\; K M_k^i,\; V M_v^i)$$

Cada cabeza puede especializarse en distintos patrones (p.ej., una en relaciones sintácticas, otra en co-referencia, etc.).

---

## Residual Connections

Las **conexiones residuales** (skip connections) suman la entrada directamente a la salida de cada subcapa:

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

**Por qué:** en redes muy profundas, el gradiente puede desvanecerse antes de llegar a las capas iniciales. Las conexiones residuales proveen un camino directo para el gradiente.

**Efecto práctico:**
- Entrenamiento más estable y rápido.
- El modelo puede hacer ajustes aditivos a la representación del token, conservando información previa.
- Permite refinamiento incremental: cada capa agrega contexto sin redefinir completamente el token.

---

## Layer Normalization

**Problema:** durante el entrenamiento, las distribuciones de las activaciones pueden cambiar mucho entre capas (*internal covariate shift*), ralentizando la convergencia.

**Solución:** normalizar dentro de cada capa.

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma + \epsilon} + \beta$$

Donde:
- $\mu = \frac{1}{d}\sum_j x_j$, $\sigma = \sqrt{\frac{1}{d}\sum_j (x_j - \mu)^2}$
- $\gamma, \beta$: parámetros **aprendibles** de escala y sesgo

**Beneficios:** convergencia más rápida, estabilidad ante vanishing/exploding gradients.

---

## Encoder

Cada bloque del encoder tiene dos subcapas con residual + LayerNorm:

```
Input Embeddings + Positional Encoding
        ↓
[Multiheaded Self-Attention] → residual → LayerNorm
        ↓
[Feed-Forward Network]       → residual → LayerNorm
        ↓
 (× N bloques)
```

### Hiperparámetros del Encoder

| Parámetro | BERT base | BERT large |
|-----------|-----------|------------|
| Num. Layers | 12 | 24 |
| Hidden Dim | 768 | 1024 |
| Attention Heads | 12 | 16 |
| Max Seq Length | 512 | 512 |

---

## BERT y Preentrenamiento Enmascarado

**BERT** (Devlin et al., 2019) usa solo el **encoder** del Transformer.

### Masked Language Modeling (MLM)

Se enmascaran ~15% de los tokens de la entrada y el modelo debe predecirlos:

```
"El [MASK] está buenísimo" → predice "hotel"
```

Esto fuerza al modelo a aprender representaciones bidireccionales: cada token puede atender a izquierda y derecha.

### Representación del input

- Token especial `[CLS]` al inicio → usado para clasificación de la secuencia completa.
- Token especial `[SEP]` para separar oraciones.
- Suma de: token embedding + segment embedding + positional embedding.

### Ideal para

Tareas de **clasificación de secuencias o tokens** (NER, POS tagging, clasificación de texto, Q&A extractivo).

---

## Transformer: Encoder-Decoder

Para tareas **generativas** (traducción, resumen, Q&A generativo):

```
Fuente  →  [Encoder]  →  Y1...Y14
                              ↓
<START> →  [Decoder]  →  P1, P2, ...
```

El **Decoder** tiene tres subcapas:

1. **Masked Multiheaded Self-Attention**: cada token del decoder solo atiende a tokens anteriores (causal/autoregresivo).
2. **Encoder-Decoder Cross-Attention**: $Q$ viene del decoder, $K$ y $V$ vienen del encoder.
3. **Feed-Forward Network**.

Todo con residual + LayerNorm.

---

## Decoder-Only Transformers

Solo tienen el componente decoder con masked self-attention (no hay encoder).

**Entrenamiento autorregresivo:** predice el próximo token dados todos los anteriores.

**GPT**, **LLaMA**, y la mayoría de los grandes LLMs actuales son decoder-only.

**Tradeoff:** sin encoder explícito → más simple y escalable, pero no tiene representaciones bidireccionales como BERT.

---

## Landscape de Modelos

```
Encoder-Only (BERT, RoBERTa, DeBERTa)
→ Clasificación, NER, extracción

Encoder-Decoder (T5, BART)
→ Traducción, resumen, Q&A generativo

Decoder-Only (GPT, LLaMA)
→ Generación de texto libre, chat, code
```

---

## Reading List

- Vaswani, A. et al. (2017). Attention Is All You Need. *NeurIPS 2017*.
- Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
- Yang, Z. et al. (2016). Hierarchical Attention Networks for Document Classification.
- Liu, Y. et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach.
- Radford, A. et al. (2019). Improving Language Understanding by Generative Pre-Training (GPT).
- He, P. et al. (2020). DeBERTa: Decoding-enhanced BERT with Disentangled Attention.
- HuggingFace Transformers: https://huggingface.co/transformers/
