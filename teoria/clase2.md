# Vectorizacion y clasificacion clasica
25-03

Ref: SLP3 Cap. 4 (Logistic Regression and Text Classification), Cap. 11 (Information Retrieval and RAG), Apendice A (Naive Bayes), Apendice J (PPMI)

## Vector Space Model

El **modelo de espacio vectorial** representa documentos y queries como vectores en un espacio de alta dimensionalidad donde cada dimension corresponde a un termino del vocabulario.

- Los documentos se convierten en representaciones numericas
- Permite medir relevancia/similitud mediante operaciones geometricas
- **Similitud coseno**: mide el angulo entre dos vectores, ignorando la magnitud

$$\cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \cdot \|\mathbf{b}\|}$$

Donde $\mathbf{a}$ y $\mathbf{b}$ son vectores de documentos/queries, $\mathbf{a} \cdot \mathbf{b}$ es el producto punto, y $\|\mathbf{a}\|$ es la norma euclidiana del vector. El resultado esta en $[-1, 1]$: valores cercanos a 1 indican alta similitud.

- Fundamento de la mayoria de sistemas de Information Retrieval clasicos

## Bag of Words (BoW)

**Bag of Words** es la representacion mas simple del Vector Space Model:
- Cada documento es un vector donde cada posicion corresponde a una palabra del vocabulario
- El valor en cada posicion es el **conteo** de esa palabra en el documento
- **Ignora el orden** de las palabras completamente
- El nombre "bolsa" refleja que solo importa que palabras aparecen y cuantas veces

Ejemplo: "the cat sat on the mat"
```
the: 2, cat: 1, sat: 1, on: 1, mat: 1
```

Limitaciones:
- Pierde informacion de orden y contexto
- Vectores muy dispersos (sparse) en vocabularios grandes
- Palabras frecuentes dominan la representacion

## TF-IDF

**TF-IDF** (Term Frequency - Inverse Document Frequency) combina dos medidas estadisticas para ponderar la importancia de un termino en un documento respecto a una coleccion.

### Term Frequency (TF)
Cuantifica la frecuencia de un termino dentro de un documento. Variantes comunes:
- TF raw: conteo directo
- TF logaritmico: $1 + \log(\text{tf})$ si $\text{tf} > 0$, else $0$ (sublineal, evita que alta frecuencia domine)

### Inverse Document Frequency (IDF)
Mide la rareza del termino en toda la coleccion:

$$\text{IDF}(t) = \log\frac{N}{\text{df}(t)}$$

Donde:
- $N$: numero total de documentos en la coleccion
- $\text{df}(t)$: document frequency, cantidad de documentos que contienen el termino $t$

Terminos raros reciben mayor IDF; palabras comunes que aparecen en muchos documentos reciben IDF bajo.

### Score combinado

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \cdot \text{IDF}(t)$$

Donde $t$ es el termino y $d$ es el documento. Produce scores que reflejan tanto la importancia local (dentro del documento) como la distintividad global (en la coleccion).

## BM25

**BM25** (Best Matching 25) es una mejora probabilistica sobre TF-IDF. Es el algoritmo de ranking mas usado en IR clasica.

### Formula

$$\text{BM25}(d, q) = \sum_{q_i \in q} \text{IDF}(q_i) \cdot \frac{\text{TF}(q_i, d) \cdot (k_1 + 1)}{\text{TF}(q_i, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

Donde:
- $d$: documento, $q$: query
- $q_i$: cada termino de la query
- $\text{TF}(q_i, d)$: frecuencia del termino $q_i$ en el documento $d$
- $\text{IDF}(q_i)$: inverse document frequency del termino $q_i$
- $|d|$: longitud del documento $d$ (en palabras)
- $\text{avgdl}$: longitud promedio de documentos en la coleccion

### Parametros clave
- $k_1$ (tipicamente 1.2-2.0): controla la **saturacion** de term frequency. Con $k_1$ alto, mas ocurrencias siguen sumando; con $k_1$ bajo, rapidamente se satura (rendimientos decrecientes).
- $b$ (tipicamente 0.75): controla la **normalizacion por longitud** del documento. $b=0$ no normaliza; $b=1$ normaliza completamente. Corrige la ventaja que tienen documentos largos.
- $\text{avgdl}$: longitud promedio de documentos en la coleccion.

### Mejoras sobre TF-IDF
- **Saturacion sublineal de TF**: mas ocurrencias tienen rendimientos decrecientes (no lineal)
- **Normalizacion explicita de longitud**: evita scores inflados en documentos largos
- Fundamentacion probabilistica (deriva del modelo de relevancia probabilistica)

## Naive Bayes

**Naive Bayes** es un clasificador **generativo** que aprende la distribucion conjunta $P(X, Y)$ y usa el teorema de Bayes para clasificar.

### Teorema de Bayes aplicado

$$P(Y \mid X) = \frac{P(X \mid Y) \cdot P(Y)}{P(X)}$$

Donde:
- $Y$: variable de clase (ej: positivo/negativo)
- $X$: vector de features del documento (las palabras observadas)
- $P(Y \mid X)$: **posterior** -- probabilidad de la clase dado el documento
- $P(X \mid Y)$: **likelihood** -- probabilidad de observar esas features dada la clase
- $P(Y)$: **prior** -- probabilidad a priori de la clase
- $P(X)$: **evidence** -- probabilidad marginal del documento (constante, se ignora al comparar clases)

Para clasificar:

$$\hat{Y} = \arg\max_y P(Y=y \mid X) = \arg\max_y P(X \mid Y=y) \cdot P(Y=y)$$

### Asuncion "naive" (independencia condicional)
Cada feature (palabra) es condicionalmente independiente dada la clase:

$$P(X \mid Y) = \prod_i P(x_i \mid Y)$$

Donde $x_i$ es la $i$-esima feature (palabra) del documento. El producto asume que cada palabra es independiente de las demas, dado que conocemos la clase $Y$. Esto simplifica enormemente el calculo aunque ignora el orden y las dependencias entre palabras.

### Entrenamiento
1. **Prior de clase**: $P(Y=c) = \frac{\text{documentos en clase } c}{\text{total documentos}}$
2. **Likelihood de palabra**: $P(w \mid Y=c) = \frac{\text{count}(w, c)}{\text{total\_words}(c)}$

### Smoothing (Laplace)
Sin smoothing, una palabra no vista en una clase da probabilidad 0, anulando todo el producto. Solucion:

$$P(w \mid Y=c) = \frac{\text{count}(w, c) + 1}{\text{total\_words}(c) + V}$$

Donde:
- $\text{count}(w, c)$: cantidad de veces que la palabra $w$ aparece en documentos de clase $c$
- $\text{total\_words}(c)$: cantidad total de palabras en todos los documentos de clase $c$
- $V$: tamanio del vocabulario (cantidad de word types distintos)

### Clasificacion (en log-space para evitar underflow)

$$\log P(Y=c \mid \text{doc}) = \log P(Y=c) + \sum_i \log P(w_i \mid Y=c)$$

Se selecciona la clase con mayor score.

### Ventajas y limitaciones
- **Ventajas**: rapido de entrenar, funciona bien con pocos datos, buena baseline
- **Limitaciones**: la asuncion de independencia es violada en lenguaje natural; no modela interacciones entre features

## SVM (Support Vector Machines)

Las **Support Vector Machines** son clasificadores **discriminativos** que buscan el hiperplano de separacion optimo entre clases.

### Idea central
- Encontrar el hiperplano que maximiza el **margen** (distancia minima) entre las clases
- Los **vectores de soporte** son los puntos de entrenamiento mas cercanos al hiperplano
- Solo estos puntos determinan la posicion del hiperplano

### Formulacion
Para clasificacion binaria, el hiperplano se define como:

$$\mathbf{w} \cdot \mathbf{x} + b = 0$$

Donde:
- $\mathbf{w}$: vector normal al hiperplano (vector de pesos)
- $\mathbf{x}$: vector de features del punto a clasificar
- $b$: sesgo (bias), desplazamiento del hiperplano respecto al origen

La decision es: $\text{sign}(\mathbf{w} \cdot \mathbf{x} + b)$ (positivo = clase 1, negativo = clase -1).

El objetivo es maximizar el margen $\frac{2}{\|\mathbf{w}\|}$ (donde $\|\mathbf{w}\|$ es la norma del vector de pesos) sujeto a que todos los puntos esten correctamente clasificados.

### Kernel Trick
Para datos no linealmente separables, SVM usa **funciones kernel** que mapean los datos a un espacio de mayor dimension donde si son separables:
- **Kernel lineal**: $K(\mathbf{x}, \mathbf{z}) = \mathbf{x} \cdot \mathbf{z}$
- **Kernel polinomial**: $K(\mathbf{x}, \mathbf{z}) = (\mathbf{x} \cdot \mathbf{z} + c)^d$ donde $c$ es una constante y $d$ el grado del polinomio
- **Kernel RBF (Gaussiano)**: $K(\mathbf{x}, \mathbf{z}) = \exp(-\gamma \|\mathbf{x} - \mathbf{z}\|^2)$ donde $\gamma > 0$ controla el ancho de la campana gaussiana

En todos los casos, $K(\mathbf{x}, \mathbf{z})$ mide la similitud entre dos puntos $\mathbf{x}$ y $\mathbf{z}$ en un espacio de mayor dimension sin necesidad de calcular la transformacion explicitamente.

### SVM para texto
- Funciona muy bien con datos de alta dimension y sparse (tipico de texto)
- El kernel lineal suele ser suficiente para clasificacion de texto
- Historicamente fue el metodo mas exitoso para clasificacion de texto antes de deep learning

## Feature Engineering de texto

El **feature engineering** es el proceso de disenar manualmente las representaciones de entrada para el clasificador.

### Tipos de features
- **Bag of Words**: conteo o presencia de palabras
- **N-gramas de caracteres**: capturan morfologia y son robustos a errores ortograficos
- **Features manuales**: conteo de palabras positivas/negativas, presencia de negacion, longitud del documento, signos de puntuacion, etc.
- **Feature interactions**: combinaciones de features primitivos (ej: "palabra es 'St.' AND palabra anterior es mayuscula")

### Disenar vs. aprender features
- **Clasico**: features disenados a mano, requiere expertise del dominio
- **Moderno (representation learning)**: se aprenden automaticamente del input (embeddings, deep learning)

### Escalado de features
Importante para modelos como logistic regression y SVM:
- **Estandarizacion (z-score)**: $x' = \frac{x - \mu}{\sigma}$ donde $\mu$ es la media y $\sigma$ la desviacion estandar de la feature en el dataset
- **Normalizacion min-max**: $x' = \frac{x - \min(x)}{\max(x) - \min(x)}$ donde $\min(x)$ y $\max(x)$ son los valores minimo y maximo de la feature
- **Log-transform**: para conteos con distribucion Zipfiana

## Latent Semantic Analysis (LSA)

**Latent Semantic Analysis** (tambien llamado Latent Semantic Indexing, LSI) usa descomposicion matricial para descubrir relaciones semanticas latentes entre terminos y documentos.

### Idea central
1. Construir una **matriz termino-documento** (o termino-termino) ponderada por TF-IDF o PPMI
2. Aplicar **SVD (Singular Value Decomposition)** para reducir dimensionalidad

### SVD (Descomposicion en Valores Singulares)

$$A = U \Sigma V^T$$

Donde:
- $A$: matriz original (terminos x documentos), dimension $[m \times n]$, con $m$ = cantidad de terminos y $n$ = cantidad de documentos
- $U$: matriz de vectores singulares izquierdos $[m \times r]$ -- cada fila es la representacion de un termino en el espacio latente
- $\Sigma$: matriz diagonal de valores singulares $[r \times r]$ -- los valores $\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_r$ indican la importancia de cada dimension latente
- $V^T$: matriz de vectores singulares derechos $[r \times n]$ -- cada columna es la representacion de un documento
- $r$: rango de la matriz $A$ (cantidad de dimensiones latentes)

### Truncamiento
Se conservan solo los $k$ valores singulares mas grandes ($k \ll r$):

$$A_k = U_k \Sigma_k V_k^T$$

Donde $k \ll r$ es la cantidad de dimensiones retenidas. $U_k$, $\Sigma_k$ y $V_k^T$ son las versiones truncadas conservando solo las $k$ dimensiones con valores singulares mas grandes. Esto produce representaciones **densas** de baja dimension que capturan las relaciones semanticas principales.

### PPMI (Positive Pointwise Mutual Information)
Una alternativa a TF-IDF para ponderar matrices de co-ocurrencia termino-termino:

$$\text{PMI}(w, c) = \log_2 \frac{P(w, c)}{P(w) \cdot P(c)}$$

$$\text{PPMI}(w, c) = \max(\text{PMI}(w, c),\; 0)$$

Donde:
- $w$: palabra objetivo (target word)
- $c$: palabra de contexto (context word)
- $P(w, c)$: probabilidad conjunta de que $w$ y $c$ co-ocurran
- $P(w)$: probabilidad marginal de $w$
- $P(c)$: probabilidad marginal de $c$
- Si $\text{PMI} > 0$: las palabras co-ocurren mas de lo esperado por azar
- Si $\text{PMI} < 0$: co-ocurren menos de lo esperado (PPMI lo reemplaza por 0)

- PMI mide cuanto mas co-ocurren dos palabras de lo esperado por azar
- PPMI reemplaza valores negativos (asociaciones menores a las esperadas) por 0, ya que los negativos son poco confiables
- Sesgo: PMI sobrevalora eventos infrecuentes. Solucion: usar $P_\alpha(c)$ con $\alpha = 0.75$

### Beneficios de LSA
- **Sinonimia**: terminos que no co-ocurren directamente pero aparecen en contextos similares tendran representaciones cercanas
- **Polisemia**: reduce parcialmente el efecto de palabras con multiples significados
- Vectores densos de baja dimension son mas eficientes que los sparse originales
- Precursor conceptual de los word embeddings modernos

## Logistic Regression (referencia complementaria)

Aunque no es parte del titulo de la clase, la **regresion logistica** es fundamental como clasificador discriminativo de texto.

### Componentes
1. **Representacion**: vector de features $\mathbf{x} = [x_1, \ldots, x_n]$
2. **Funcion de clasificacion**: sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$, donde $z = \mathbf{w} \cdot \mathbf{x} + b$ ($\mathbf{w}$: pesos, $b$: bias). Mapea cualquier valor real al rango $(0, 1)$
3. **Funcion de perdida**: cross-entropy $L_{\text{CE}} = -[y \log \hat{y} + (1-y) \log(1 - \hat{y})]$ donde $y \in \{0, 1\}$ es la etiqueta real y $\hat{y}$ es la probabilidad predicha por el modelo
4. **Optimizacion**: Stochastic Gradient Descent (SGD)

### Multinomial (Softmax) Logistic Regression
Para $K$ clases: $\hat{\mathbf{y}} = \text{softmax}(\mathbf{W}\mathbf{x} + \mathbf{b})$

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

Donde $z_i = \mathbf{w}_i \cdot \mathbf{x} + b_i$ es el logit (score sin normalizar) para la clase $i$, y $K$ es el numero total de clases. La funcion normaliza los logits en una distribucion de probabilidad que suma 1.

### Generativo vs. Discriminativo
- **Naive Bayes** (generativo): modela $P(X \mid Y)$, asume independencia, funciona bien con pocos datos
- **Logistic Regression** (discriminativo): modela $P(Y \mid X)$ directamente, no asume independencia, generalmente mejor con muchos datos

---

## Referencias

- Jurafsky, D. & Martin, J. H. (2026). *Speech and Language Processing* (3rd ed. draft). Cap. 4: Logistic Regression and Text Classification. https://web.stanford.edu/~jurafsky/slp3/4.pdf
- Jurafsky, D. & Martin, J. H. (2026). *Speech and Language Processing* (3rd ed. draft). Cap. 11: Information Retrieval and Retrieval-Augmented Generation. https://web.stanford.edu/~jurafsky/slp3/11.pdf
- Jurafsky, D. & Martin, J. H. (2026). *Speech and Language Processing* (3rd ed. draft). Apendice A: Naive Bayes Classification. https://web.stanford.edu/~jurafsky/slp3/A.pdf
- Jurafsky, D. & Martin, J. H. (2026). *Speech and Language Processing* (3rd ed. draft). Apendice J: PPMI. https://web.stanford.edu/~jurafsky/slp3/J.pdf
