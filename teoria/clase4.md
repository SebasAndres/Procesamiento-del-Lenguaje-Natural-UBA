# Language Models I: RNN, LSTM, GRU
27-05

Ref: Goodfellow, Bengio & Courville (2016). *Deep Learning*. / Hochreiter & Schmidhuber (1997). *Long short-term memory*. / Cho et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder*. / Peters et al. (2018). *Deep contextualized word representations (ELMo)*.

---

## La necesidad de memoria

Los modelos N-gram no pueden generalizar a palabras similares: si veo "darle de comer al ___" en test pero no en train con "perro", le da probabilidad 0. Los embeddings + redes neuronales ayudan, pero el problema más profundo es la **secuencialidad**: el contexto previo y el orden importan.

- En traducción y análisis de sentimientos, el contexto anterior (y posterior) de una palabra es esencial.
- "Juan golpeó a Pedro" ≠ "Pedro golpeó a Juan" (mismas palabras, distinto significado).

Las RNNs nacen para procesar datos secuenciales manteniendo un estado interno.

---

## Redes Neuronales Recurrentes (RNN)

### Arquitectura

El hidden state $h_t$ se actualiza en cada paso combinando la entrada actual $x_t$ y el estado anterior $h_{t-1}$:

$$h_t = \tanh(W h_{t-1} + U x_t + b_h)$$

$$o_t = V h_t + b_o$$

$$\hat{y}_t = \text{softmax}(o_t)$$

Donde:
- $x_t$: embedding del token en la posición $t$
- $h_t$: estado oculto (la "memoria" de la red)
- $o_t$: output antes de softmax
- $W, U, V$: matrices de pesos **compartidas** en todos los pasos de tiempo

**Ventajas:** en teoría puede usar información de toda la secuencia; el tamaño del modelo no depende del largo del input.

### Usos principales

**Generación de texto:** samplear $\hat{y}_t$ de la distribución de probabilidad y realimentarlo como $x_{t+1}$.

**Clasificación:** tomar la representación del último hidden state, o bien un mean/max element-wise de todos los $h_t$, y pasarla por un clasificador.

### Multi-layer RNNs

Apilar $L$ capas donde la salida de la capa $l$ es la entrada de la capa $l+1$. Permite analizar la información a diferentes niveles de abstracción. El orden de ejecución es: primero todos los pasos de la capa 1, luego capa 2, etc.

---

## Seq2seq: Encoder-Decoder

Para traducción y otras tareas generativas se usa un modelo de dos partes:

1. **Encoder RNN**: procesa la secuencia fuente. El último hidden state resume toda la secuencia (vector de contexto).
2. **Decoder RNN**: modelo de lenguaje condicionado por el vector de contexto del encoder. Genera la secuencia destino token a token con argmax.

```
Vamos a lo de Raúl  →  [Encoder]  →  h_final  →  [Decoder]  →  Let's go to Raúl's
```

**Problema (cuello de botella):** el encoder debe comprimir toda la información de la secuencia fuente en un único vector de tamaño fijo. Para secuencias largas, esto pierde información.

---

## La celda RNR en detalle

Las celdas RNR tienen tres matrices de pesos ($U, W, V$) que se reutilizan en todos los pasos:

$$h_t = \tanh(W h_{t-1} + U x_t)$$

La información fluye paso a paso: cada $h_t$ depende de $h_{t-1}$, que a su vez depende de $h_{t-2}$, etc. Esto permite capturar dependencias arbitrariamente largas **en teoría**.

---

## Entrenamiento: Backpropagation Through Time (BPTT)

Se "desenrolla" la RNN en el tiempo y se calcula el gradiente de la pérdida total $L = \sum_i L_i$ respecto a todos los pesos.

El truco: como $W$ es compartida, el gradiente $\frac{\partial L}{\partial W}$ requiere sumar las contribuciones de todos los pasos. La activación $h_t$ depende de $h_{t-1}$, que también depende de $W$, por lo que el gradiente se propaga hacia atrás a través de todas las conexiones temporales.

---

## Exploding y Vanishing Gradients

Al propagar el gradiente hacia atrás $T$ pasos, aparece un producto de $T$ matrices $W$:

$$\frac{\partial h_T}{\partial h_1} = \prod_{t=2}^{T} W \cdot \text{diag}(\sigma'(h_{t-1}))$$

- Si los valores de $W$ son **grandes**: el gradiente **explota** (actualizaciones gigantes).
- Si los valores de $W$ son **chicos**: el gradiente **desaparece** (no se aprenden dependencias de largo alcance).

### Gradient Clipping (para exploding)

Si $\|\nabla\| > \text{threshold}$, escalar el gradiente:

$$\nabla \leftarrow \frac{\text{threshold}}{\|\nabla\|} \cdot \nabla$$

Esto evita que la optimización "salte el acantilado" en la superficie de pérdida.

### Vanishing gradients

La solución no es clip sino arquitecturas con puertas (LSTM, GRU) que proveen caminos directos para el gradiente.

---

## LSTM (Long Short-Term Memory)

Hochreiter & Schmidhuber (1997). Diseñadas específicamente para dependencias a largo plazo.

**Clave:** introduce el **cell state** $c_t$ (memoria de largo plazo) separado del hidden state $h_t$.

### Las cuatro compuertas

Sea $[h_{t-1}, x_t]$ la concatenación del estado anterior y la entrada actual:

**Forget Gate** — qué olvidar del cell state:
$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$

**Input Gate** — qué nueva información agregar:
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$

**New Cell Content** — contenido candidato:
$$\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$$

**Actualización del Cell State:**
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$

**Output Gate** — qué exponer como hidden state:
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$

**Nuevo Hidden State:**
$$h_t = o_t \odot \tanh(c_t)$$

El cell state $c_t$ fluye con cambios solo multiplicativos ($f_t$) y aditivos ($i_t \odot \tilde{c}_t$), lo que permite que el gradiente fluya sin desvanecerse.

---

## GRU (Gated Recurrent Unit)

Cho et al. (2014). Más simple que LSTM, menos parámetros.

**Reset Gate** — cuánto del estado pasado ignorar:
$$r_t = \sigma(W_r [h_{t-1}, x_t])$$

**Update Gate** — balance entre estado viejo y nuevo:
$$z_t = \sigma(W_z [h_{t-1}, x_t])$$

**Nuevo contenido candidato:**
$$\tilde{h}_t = \tanh(W_h [r_t \odot h_{t-1}, x_t])$$

**Nuevo Hidden State:**
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

### LSTM vs GRU

No hay evidencia fuerte de que uno sea consistentemente mejor que el otro. GRU tiene menos parámetros → más eficiente computacionalmente. Convención: usar LSTM por defecto, probar GRU si se necesita eficiencia o mejora marginal.

---

## RNR Bidireccionales

Dos RNNs en paralelo: una procesa de izquierda a derecha, otra de derecha a izquierda. Los outputs se concatenan:

$$\vec{h}_t = \overrightarrow{\text{RNN}}(x_t, \vec{h}_{t-1}) \qquad \overleftarrow{h}_t = \overleftarrow{\text{RNN}}(x_t, \overleftarrow{h}_{t+1})$$

$$h_t = [\vec{h}_t \;;\; \overleftarrow{h}_t]$$

La representación de la oración se obtiene por mean/max de todos los $h_t$.

**Importante:** no sirve para modelos de lenguaje (el futuro filtra hacia el pasado), pero es ideal para clasificación, etiquetado, NER, etc.

La arquitectura **Bi-LSTM** fue dominante en NLP antes de los Transformers.

---

## ELMo y Embeddings Contextuales

### El problema de los embeddings estáticos

Word2Vec y GloVe asignan **un único vector** a cada palabra sin importar el contexto:
- "Saqué dinero del **banco**" ← institución financiera
- "Me siento en el **banco**" ← mueble

Ambas usan el mismo embedding para "banco".

### ELMo (Embeddings from Language Models)

Peters et al. (2018). Genera representaciones **dinámicas** basadas en el contexto.

**Arquitectura:**
- Input: embeddings basados en caracteres (CNN)
- 2 capas de LSTM bidireccional (4096 dimensiones ocultas, 512 de output)
- Conexiones residuales entre capas

**Propiedades de las capas:**
- Capas inferiores → características **sintácticas** (POS, estructura)
- Capas superiores → características **semánticas** (significado, relaciones)

**Preentrenamiento:** predicción de la siguiente (y anterior) palabra sobre corpus grandes (10B palabras, 10 epochs). El modelo pre-entrenado puede fine-tunearse en tareas específicas.

**Fine-tuning:** adaptar el modelo pre-entrenado a una tarea específica con datos más pequeños. Proceso:
1. Preparar datos específicos de la tarea
2. Adaptar el output según la tarea (clasificación, etiquetado, etc.)
3. Entrenamiento con hyper-parameter tuning

---

## Tareas NLP de referencia

- **SQuAD**: comprensión lectora, extraer respuestas de un pasaje de Wikipedia.
- **SNLI**: pares de oraciones → Entailment / Contradiction / Neutral.
- **SRL** (Semantic Role Labeling): identificar roles semánticos (Agente, Paciente, Instrumento...).
- **NER** (Named Entity Recognition): PER, ORG, LOC, DATE, etc.
- **WSD** (Word Sense Disambiguation): desambiguación de significado de palabras.
- **POS Tagging**: etiquetado de categorías gramaticales.

---

## Reading List

- Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*. MIT Press.
- Hochreiter, S. & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
- Cho, K. et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation.
- Peters, M. et al. (2018). Deep contextualized word representations. *NAACL 2018*.
- Goldberg, Y. (2015). A Primer on Neural Network Models for Natural Language Processing.
- Luong, M. et al. (2015). Stanford Neural Machine Translation Systems for Spoken Language Domains.
- Akbik, A. (2018). Contextual String Embeddings for Sequence Labeling.
- Lample, G. et al. (2016). Neural Architectures for Named Entity Recognition.
- Colah's blog: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
