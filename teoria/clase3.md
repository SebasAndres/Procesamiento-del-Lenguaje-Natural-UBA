# Word Embeddings Estaticos
01-04

Ref: SLP3 Cap. 6 (Vector Semantics and Embeddings)

## Motivacion: de vectores sparse a densos

Las representaciones de clase2 (BoW, TF-IDF) producen vectores **sparse** de dimension $|V|$ (decenas de miles). Los **word embeddings** son vectores **densos** de baja dimension (100-1000 dims) que capturan significado semantico.

**Hipotesis distribucional** (Firth, 1957): "a word is characterized by the company it keeps". Palabras que aparecen en contextos similares tienen significados similares.

Diferencia clave con LSA (clase2):
- LSA: factorizacion de matriz global termino-documento
- Word2Vec/GloVe: entrenados con ventanas de contexto locales o co-ocurrencias globales, producen representaciones mas ricas semanticamente

## Word2Vec (Mikolov et al., 2013)

Word2Vec tiene dos arquitecturas que aprenden embeddings a partir de texto sin etiquetar.

### Skip-gram

Dado una palabra objetivo $w_t$, predice las palabras de contexto en una ventana de tamanio $\pm c$:

$$\text{objetivo: maximizar } \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c,\; j \neq 0} \log P(w_{t+j} \mid w_t)$$

Donde $T$ es la longitud del corpus, $c$ es el tamanio de la ventana de contexto, y $w_{t+j}$ son las palabras de contexto alrededor de la palabra objetivo $w_t$.

La probabilidad se calcula con softmax sobre dos matrices de embeddings:

$$P(w_O \mid w_I) = \frac{\exp(\mathbf{v}_{w_O}^{\prime\top} \mathbf{v}_{w_I})}{\sum_{w=1}^{V} \exp(\mathbf{v}_w^{\prime\top} \mathbf{v}_{w_I})}$$

Donde:
- $\mathbf{v}_{w_I}$: embedding de la palabra **input** $w_I$ (matriz $W$)
- $\mathbf{v}_{w_O}^{\prime}$: embedding de la palabra **output** $w_O$ (matriz $W'$)
- Al final se usa $W$ como los embeddings del modelo

### CBOW (Continuous Bag of Words)

Inverso de skip-gram: dadas las palabras de contexto, predice la palabra central. Mas rapido de entrenar, util para vocabularios grandes.

### Negative Sampling

El denominador del softmax es costoso (suma sobre todo $V$). **Negative sampling** simplifica el objetivo: en vez de predecir sobre todo el vocabulario, distingue la palabra de contexto real de $k$ palabras negativas muestreadas aleatoriamente:

$$\log \sigma(\mathbf{v}_{w_O}^{\prime\top} \mathbf{v}_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} \left[\log \sigma(-\mathbf{v}_{w_i}^{\prime\top} \mathbf{v}_{w_I})\right]$$

Donde $\sigma$ es la funcion sigmoide, $k$ es el numero de muestras negativas (tipicamente 5-20), y $P_n(w) \propto f(w)^{3/4}$ es una distribucion de muestreo que suaviza las frecuencias para que las palabras raras sean mas representadas.

### Propiedades semanticas: analogias vectoriales

Los embeddings capturan relaciones por aritmetica vectorial:

$$\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}$$

Esto emerge del entrenamiento, no esta programado. Funciona para relaciones sinonimia, antonimia, relaciones pais-capital, genero, tiempo verbal, etc.

Para buscar la palabra analogia:

$$\hat{w} = \arg\max_{w} \cos(\mathbf{v}_w,\; \mathbf{v}_b - \mathbf{v}_a + \mathbf{v}_{a'})$$

## GloVe (Pennington et al., 2014)

**GloVe** (Global Vectors) combina las ventajas de los metodos de matrix factorization (capturan estadisticas globales) y los modelos de ventana local (como Word2Vec).

### Idea central

Construir una matriz de co-ocurrencia global $X$ donde $X_{ij}$ = numero de veces que la palabra $j$ aparece en el contexto de la palabra $i$. Luego factorizarla con un objetivo especifico:

$$J = \sum_{i,j=1}^{V} f(X_{ij}) \left( \mathbf{v}_i^{\top} \mathbf{v}_j + b_i + b_j - \log X_{ij} \right)^2$$

Donde:
- $\mathbf{v}_i, \mathbf{v}_j$: vectores de embeddings de las palabras $i$ y $j$
- $b_i, b_j$: terminos de sesgo (bias) por palabra
- $f(X_{ij})$: funcion de peso que reduce la importancia de co-ocurrencias muy frecuentes:

$$f(x) = \begin{cases} (x/x_{\max})^\alpha & \text{si } x < x_{\max} \\ 1 & \text{en caso contrario} \end{cases}$$

Con $x_{\max} = 100$ y $\alpha = 3/4$ tipicamente. El embedding final de una palabra es $\mathbf{v}_i + \mathbf{v}_i^{\prime}$ (suma de ambas matrices).

### Ventaja sobre Word2Vec

GloVe entrena directamente sobre la matriz de co-ocurrencia (estadisticas globales), lo que suele ser mas eficiente y producir mejores embeddings en benchmarks de similitud y analogias.

## FastText (Bojanowski et al., 2017)

**FastText** extiende Word2Vec usando **subword embeddings**: cada palabra es la suma de los embeddings de sus n-gramas de caracteres.

Para la palabra "where", con $n=3$: `<wh`, `whe`, `her`, `ere`, `re>`, y el token especial `<where>`.

El embedding de una palabra es:

$$\mathbf{v}_w = \frac{1}{|G_w|} \sum_{g \in G_w} \mathbf{z}_g$$

Donde $G_w$ es el conjunto de n-gramas de caracteres de la palabra $w$ (incluyendo la palabra completa) y $\mathbf{z}_g$ es el vector del n-grama $g$.

### Ventajas
- **Maneja OOV**: una palabra nunca vista puede tener un embedding si comparte n-gramas con palabras conocidas
- **Morfologia**: mejora en idiomas morfologicamente ricos (aleman, finlandés, espanol) donde variantes de una misma raiz son frecuentes
- **Robustez a errores**: palabras con typos tienen embeddings razonables

## Evaluacion de embeddings

### Evaluacion intrinseca
- **Similitud de palabras**: correlacion con juicios humanos en datasets como WordSim-353, SimLex-999
- **Analogias**: accuracy en el benchmark de Mikolov (Google Analogy Test, ~19K pares)
- **Clustering**: coherencia de los clusters formados

### Evaluacion extrinseca
Desempeno en tareas downstream: NER, clasificacion de sentimiento, parsing. Es la metrica mas importante pero mas costosa.

### Limitacion fundamental: embeddings estaticos

Todos estos modelos producen **un unico vector por palabra** sin importar el contexto. El problema:

- "bank" en "river bank" vs. "savings bank" → mismo vector
- "play" en "I play guitar" vs. "a play by Shakespeare" → mismo vector

Esto motiva los **embeddings contextuales** (ELMo, BERT) que se veran en clases 4 y 6.

## Sesgo en word embeddings

Los embeddings aprenden los sesgos presentes en los datos de entrenamiento:

$$\mathbf{v}_{\text{doctor}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{nurse}}$$

Esto refleja sesgos de genero del corpus. Tecnicas de debiasing (Bolukbasi et al., 2016) buscan remover estas asociaciones proyectando embeddings fuera del subespacio de genero.

---

## Referencias

- Jurafsky, D. & Martin, J. H. (2026). *Speech and Language Processing* (3rd ed. draft). Cap. 6: Vector Semantics and Embeddings. https://web.stanford.edu/~jurafsky/slp3/6.pdf
- Mikolov, T. et al. (2013). Efficient Estimation of Word Representations in Vector Space. arXiv:1301.3781
- Pennington, J., Socher, R., & Manning, C. (2014). GloVe: Global Vectors for Word Representation. EMNLP.
- Bojanowski, P. et al. (2017). Enriching Word Vectors with Subword Information. TACL.
