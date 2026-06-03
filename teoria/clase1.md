# Pre-procesamiento y modelos probabilisticos
18-03

Ref: SLP3 Cap. 2 (Words and Tokens), Cap. 3 (N-gram Language Models), Apendice B (Kneser-Ney Smoothing)

## Tokenizacion

La **tokenizacion** es el proceso de segmentar texto en unidades llamadas **tokens**. Es el primer paso en cualquier pipeline de NLP.

### Conceptos clave
- **Word types**: cantidad de palabras distintas en un corpus (vocabulario $V$, tamanio $|V|$)
- **Word instances**: cantidad total de palabras ($N$ running words)
- **Ley de Herdan/Heaps**: $|V| = kN^\beta$ donde $k$ y $\beta$ son constantes positivas ($0 < \beta < 1$), y $N$ es el numero total de word instances. El vocabulario crece sin limite con mas datos
- **Palabras desconocidas (OOV)**: ningun vocabulario finito cubre todas las palabras posibles

### Subword Tokenization: Byte-Pair Encoding (BPE)
En la practica, los modelos de lenguaje no usan palabras como unidad sino **subwords** que pueden recombinarse para representar palabras nunca vistas.

**BPE** (Sennrich et al., 2016) tiene dos fases:
1. **Trainer**: parte del conjunto de caracteres individuales. Iterativamente encuentra el par de tokens adyacentes mas frecuente, los fusiona en un nuevo token, y repite k veces.
2. **Encoder**: aplica los merges aprendidos (en orden) sobre texto nuevo para tokenizarlo.

Algoritmo:
```
V <- todos los caracteres unicos en C
for i = 1 to k:
    tL, tR <- par adyacente mas frecuente en C
    t_new <- tL + tR
    V <- V + {t_new}
    Reemplazar cada (tL, tR) en C con t_new
return V
```

Ejemplo con corpus `set, new, renew, reset`:
- Vocabulario inicial: {_, e, n, r, s, t, w}
- Merge 1: n+e -> ne (frecuencia 4)
- Merge 2: ne+w -> new (frecuencia 4)
- Merge 3: _+r -> _r (frecuencia 3)
- Y asi sucesivamente...

En la practica BPE se ejecuta sobre bytes UTF-8, produciendo vocabularios de 50K-200K tokens. Variantes modernas incluyen **SuperBPE** y **BoundlessBPE** que permiten merges entre palabras.

**Problema multilingue**: los tokenizadores BPE entrenados mayormente en ingles sobre-segmentan otros idiomas (ej: espaniol usa ~33 tokens donde ingles usa ~18 para oraciones equivalentes).

## Normalizacion (Stemming vs. Lemmatization)

La **normalizacion** busca reducir la variabilidad lexica mapeando formas diferentes a una representacion comun.

### Stemming
- Aplica reglas heuristicas para remover sufijos
- Algoritmo clasico: **Porter Stemmer**
- Ejemplo: "running", "runs", "ran" -> "run" (aproximado)
- Rapido pero impreciso: puede producir stems que no son palabras reales
- No distingue categorias gramaticales

### Lemmatization
- Usa un diccionario/analisis morfologico para encontrar la **forma base** (lema)
- Ejemplo: "better" -> "good", "was" -> "be"
- Mas preciso pero mas lento y requiere recursos linguisticos
- Considera la categoria gramatical (POS) de la palabra

### Morfologia relevante
- **Morfema**: unidad minima con significado (ej: "cats" = "cat" + "-s")
- **Raiz (root)**: morfema central con el significado principal
- **Afijos**: morfemas que agregan significado adicional
  - **Inflexionales**: marcan rol sintactico (-s plural, -ed pasado)
  - **Derivacionales**: cambian la clase gramatical (care -> careful -> carefully)
- **Cliticos**: morfemas que actuan como palabras pero se adjuntan a otras ("I've", "'s")

### Case Folding
- Convertir todo a minusculas: util para IR y muchas tareas
- Problema: pierde informacion (ej: "US" vs "us", "Apple" vs "apple")
- Solucion comun: mantener dos versiones del modelo

## Stopwords

Las **stopwords** son palabras de alta frecuencia (articulos, preposiciones, pronombres) que a menudo se filtran en tareas de IR y clasificacion porque aportan poco contenido semantico.

- Ejemplos tipicos: "the", "a", "is", "in", "of", "and", "to"
- Se relacionan con las **function words** (palabras gramaticales como "a", "of") vs. **content words** (sustantivos, verbos, adjetivos con significado)
- Las function words tienen un conjunto relativamente fijo; las content words crecen indefinidamente
- **Cuando NO filtrar**: en tareas donde el orden importa (language modeling), en busquedas de frases exactas, o cuando la stopword es parte de un nombre propio ("The Who")
- Metodo comun: usar una lista predefinida o filtrar por frecuencia/IDF

## N-Grams

Un **n-grama** es una secuencia contigua de $n$ elementos de un texto. Son la base de los modelos de lenguaje estadisticos.

### Tipos
- **Unigrama** ($n=1$): palabras individuales. $P(w_i)$
- **Bigrama** ($n=2$): pares de palabras. $P(w_i \mid w_{i-1})$
- **Trigrama** ($n=3$): tripletas. $P(w_i \mid w_{i-2}, w_{i-1})$

### Estimacion por Maximum Likelihood (MLE)
Para bigramas:

$$P(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}$$

Donde:
- $w_i$: palabra en la posicion $i$ de la secuencia
- $C(w_{i-1}, w_i)$: conteo de veces que el bigrama $(w_{i-1}, w_i)$ aparece en el corpus
- $C(w_{i-1})$: conteo total de la palabra $w_{i-1}$ en el corpus

### Problema de esparsidad
A medida que $n$ crece, la mayoria de los n-gramas posibles nunca aparecen en el corpus. Un n-grama no visto recibe probabilidad 0, lo cual es problematico porque:
- Una sola probabilidad 0 hace que toda la cadena tenga probabilidad 0
- El modelo no puede generalizar a secuencias nuevas

## Cadenas de Markov

La **asuncion de Markov** simplifica el modelado del lenguaje: la probabilidad de una palabra depende solo de las $n-1$ palabras anteriores, no de todo el historial:

$$P(w_i \mid w_1 \ldots w_{i-1}) \approx P(w_i \mid w_{i-n+1} \ldots w_{i-1})$$

Donde $w_i$ es la palabra actual, $w_1 \ldots w_{i-1}$ es todo el historial previo, y $n$ es el orden del modelo (cantidad de palabras de contexto + 1).

Una **cadena de Markov** es un modelo probabilistico donde las transiciones entre estados dependen unicamente del estado actual. En modelos de lenguaje:
- Los **estados** son palabras (o secuencias de $n-1$ palabras)
- Las **transiciones** siguen las probabilidades de n-gramas estimadas
- Permite **generacion de texto**: muestrear la siguiente palabra segun la distribucion condicionada

Para bigramas (Markov de orden 1):

$$P(w_1, w_2, \ldots, w_n) = \prod_{i=1}^{n} P(w_i \mid w_{i-1})$$

Donde $w_0$ se define como un token especial de inicio de secuencia ($\langle s \rangle$).

## Smoothing (Laplace/Kneser-Ney)

El **smoothing** aborda el problema de esparsidad: muchos n-gramas validos tienen conteo 0 en entrenamiento.

### Laplace (Add-one) Smoothing
Agrega 1 a todos los conteos:

$$P_{\text{Laplace}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + 1}{C(w_{i-1}) + V}$$

Donde:
- $C(w_{i-1}, w_i)$: conteo del bigrama en el corpus
- $C(w_{i-1})$: conteo del contexto (unigrama previo)
- $V$: tamanio del vocabulario (cantidad de word types)
- Se suma 1 al numerador y $V$ al denominador para que la distribucion siga sumando 1

Simple pero redistribuye demasiada masa probabilistica.

### Add-k Smoothing
Generalizacion con $k < 1$:

$$P_{\text{add-k}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) + k}{C(w_{i-1}) + kV}$$

Donde $k$ es un hiperparametro fraccional (tipicamente $k < 1$) y $V$ es el tamanio del vocabulario.

### Backoff
Usa n-gramas de orden inferior cuando los de orden superior no tienen evidencia:
- Si $C(w_{i-2}, w_{i-1}, w_i) > 0$, usar el trigrama
- Si no, usar el bigrama $P(w_i \mid w_{i-1})$
- Si no, usar el unigrama $P(w_i)$

### Interpolacion
Combina multiples ordenes simultaneamente:

$$P(w_i \mid w_{i-2}, w_{i-1}) = \lambda_3 P_3(w_i \mid w_{i-2}, w_{i-1}) + \lambda_2 P_2(w_i \mid w_{i-1}) + \lambda_1 P_1(w_i)$$

Donde:
- $P_3$, $P_2$, $P_1$: probabilidades estimadas por el modelo de trigramas, bigramas y unigramas respectivamente
- $\lambda_1, \lambda_2, \lambda_3$: pesos de interpolacion que suman 1 ($\lambda_1 + \lambda_2 + \lambda_3 = 1$), se aprenden con datos held-out

### Absolute Discounting
Observacion empirica (Church & Gale, 1991): los conteos en un held-out set son aprox. el conteo de entrenamiento menos ~0.75. Entonces se resta un descuento fijo $d$ a cada conteo:

$$P_{\text{AD}}(w_i \mid w_{i-1}) = \frac{C(w_{i-1}, w_i) - d}{\sum_v C(w_{i-1}, v)} + \lambda(w_{i-1}) \cdot P(w_i)$$

Donde:
- $d$: descuento fijo que se resta a cada conteo ($0 \leq d \leq 1$, tipicamente $\approx 0.75$)
- $\sum_v C(w_{i-1}, v)$: total de bigramas que comienzan con $w_{i-1}$ (normalizador)
- $\lambda(w_{i-1})$: peso de interpolacion que distribuye la masa probabilistica ahorrada por el descuento
- $P(w_i)$: probabilidad unigrama usada como backoff

### Kneser-Ney Smoothing
Mejora el absolute discounting con la idea de **probabilidad de continuacion**: en vez de usar $P(w)$ como backoff, usa $P_{\text{CONTINUATION}}(w)$ que mide en cuantos contextos diferentes aparece $w$:

$$P_{\text{CONTINUATION}}(w) = \frac{|\{v : C(v, w) > 0\}|}{|\{(u', w') : C(u', w') > 0\}|}$$

Donde:
- $|\{v : C(v, w) > 0\}|$: cantidad de contextos unicos (palabras $v$) que preceden a $w$ (numerador)
- $|\{(u', w') : C(u', w') > 0\}|$: cantidad total de tipos de bigramas distintos en el corpus (denominador, constante de normalizacion)

Intuicion: "Kong" es muy frecuente pero solo aparece despues de "Hong". "glasses" es menos frecuente pero aparece en muchos contextos -> glasses deberia tener mayor probabilidad de continuacion.

**Interpolated Kneser-Ney** (formula final para bigramas):

$$P_{\text{KN}}(w_i \mid w_{i-1}) = \frac{\max(C(w_{i-1}, w_i) - d,\; 0)}{C(w_{i-1})} + \lambda(w_{i-1}) \cdot P_{\text{CONTINUATION}}(w_i)$$

Donde el primer termino es la probabilidad descontada del bigrama (truncada a 0 si el conteo es menor que $d$), y el segundo termino interpola con la probabilidad de continuacion.

La constante de interpolacion $\lambda$:

$$\lambda(w_{i-1}) = \frac{d}{\sum_v C(w_{i-1}, v)} \cdot |\{w : C(w_{i-1}, w) > 0\}|$$

Donde:
- $\frac{d}{\sum_v C(w_{i-1}, v)}$: descuento normalizado (fraccion de masa descontada por cada bigrama)
- $|\{w : C(w_{i-1}, w) > 0\}|$: cantidad de word types distintos que siguen a $w_{i-1}$ (es decir, cuantas veces se aplico el descuento)

**Formula recursiva general**:

$$P_{\text{KN}}(w_i \mid w_{i-n+1:i-1}) = \frac{\max(c_{\text{KN}}(w_{i-n+1:i}) - d,\; 0)}{\sum_v c_{\text{KN}}(w_{i-n+1:i-1}\; v)} + \lambda(w_{i-n+1:i-1}) \cdot P_{\text{KN}}(w_i \mid w_{i-n+2:i-1})$$

Donde $c_{\text{KN}}$ usa conteos reales para el orden mas alto y conteos de continuacion para ordenes inferiores.

**Modified Kneser-Ney** (Chen & Goodman, 1998): usa tres descuentos diferentes $d_1, d_2, d_{3+}$ segun si el conteo es 1, 2 o $\geq 3$. Es el metodo de smoothing con mejor rendimiento para n-gramas.

## Perplexity

La **perplejidad** mide que tan bien un modelo de lenguaje predice datos de test. Menor perplejidad = mejor modelo.

### Formula

$$\text{Perplexity}(W) = P(w_1, w_2, \ldots, w_N)^{-1/N} = 2^{\;-\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i \mid w_1 \ldots w_{i-1})}$$

Donde:
- $W = w_1, w_2, \ldots, w_N$: secuencia de test de $N$ palabras
- $P(w_i \mid w_1 \ldots w_{i-1})$: probabilidad asignada por el modelo a la palabra $w_i$ dado el contexto previo
- El exponente $-\frac{1}{N}$ normaliza por la longitud de la secuencia (promedio geometrico inverso)

### Interpretacion
- Representa el **factor de ramificacion efectivo**: en promedio, el modelo esta tan confundido como si eligiera uniformemente entre Perplexity opciones en cada paso
- Perplejidad de 100 = el modelo tiene la misma incertidumbre que elegir entre 100 opciones equiprobables
- Es la metrica estandar para evaluar modelos de lenguaje
- Equivalente a $2^{H}$ donde $H$ es la entropia cruzada del modelo sobre los datos de test
- Solo se puede comparar perplexity entre modelos con el **mismo vocabulario**

---

## Referencias

- Jurafsky, D. & Martin, J. H. (2026). *Speech and Language Processing* (3rd ed. draft). Cap. 2: Words and Tokens. https://web.stanford.edu/~jurafsky/slp3/2.pdf
- Jurafsky, D. & Martin, J. H. (2026). *Speech and Language Processing* (3rd ed. draft). Cap. 3: N-gram Language Models. https://web.stanford.edu/~jurafsky/slp3/3.pdf
- Jurafsky, D. & Martin, J. H. (2026). *Speech and Language Processing* (3rd ed. draft). Apendice B: Kneser-Ney Smoothing. https://web.stanford.edu/~jurafsky/slp3/B.pdf
