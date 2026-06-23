# Pretraining y Modelos Generativos
10-06

Ref: Devlin et al. (2019). *BERT*. / Clark et al. (2020). *ELECTRA*. / Hu et al. (2021). *LoRA*. / Raffel et al. (2020). *T5*. / Radford et al. (2019). *GPT-2*. / Brown et al. (2020). *GPT-3*.

---

## Pretraining y Transfer Learning

El **preentrenamiento** consiste en entrenar un modelo en un corpus de texto grande en una **tarea general**, antes de adaptarlo a una tarea específica. Esto ayuda al modelo a incorporar estructuras semánticas, sintácticas y conocimiento del mundo.

**Beneficios:**
- Acelera el entrenamiento para la tarea objetivo.
- Reduce drásticamente la cantidad de datos necesarios para fine-tuning (hasta 0 en modelos grandes, vía zero-shot/few-shot).

**Evolución:**
1. Primero se preentrenaban solo los **embeddings** (Word2Vec, GloVe) y el resto de la red se inicializaba al azar.
2. Luego se empezó a preentrenar **todo el modelo** (ELMo, BERT, GPT...).

```
Texto general → [LLM Preentrenado] → Texto en Inglés / Texto en Español / Texto Especializado → [Modelo Especializado]
```

### Tipos de preentrenamiento

| Tipo | Descripción | Ejemplo |
|------|-------------|---------|
| **MLM** (Masked Language Modeling) | Predice palabras enmascaradas en una oración | BERT |
| **NWP** (Next Word Prediction) | Predice la siguiente palabra en una secuencia | GPT |
| **NSP** (Next Sentence Prediction) | Predice si una oración sigue a otra | BERT |
| **Discriminativo** | Aprende a distinguir tokens reales de generados (reemplazos) | ELECTRA |

### Qué aprenden los modelos durante el preentrenamiento

A fuerza de predecir tokens faltantes sobre corpus masivos, los modelos terminan capturando: relaciones semánticas ("Vivo en Buenos \_, Argentina" → Aires), conocimiento de mundo y hechos (capitales, fechas históricas), conocimiento numérico/aritmético básico, analogías, y hasta sentimiento/sintaxis. Esto emerge sin supervisión explícita — solo de la tarea de predicción de tokens.

### Por qué transfer learning funciona (intuición de SGD)

- Hay mucho dato textual "gratuito" (no etiquetado) disponible para preentrenar.
- El modelo preentrenado inicia el fine-tuning en un punto del espacio de parámetros ya "cerca" de un óptimo razonable, en lugar de un punto aleatorio.
- Esto reduce la exploración aleatoria necesaria y permite ajustes menores para adaptarse a la tarea nueva, ahorrando tiempo y datos.

---

## Modelo de Lenguaje Enmascarado (MLM)

Modelos **bidireccionales** (BERT, RoBERTa, DeBERTa) entrenados para predecir palabras ocultas usando contexto de ambos lados.

**Proceso de entrenamiento (BERT):**
- Se selecciona aleatoriamente ~15% de los tokens de la entrada.
- De esos tokens seleccionados:
  - 80% de las veces se reemplazan por `[MASK]`.
  - 10% de las veces se reemplazan por un token aleatorio.
  - 10% de las veces se dejan sin cambios (pero igual se predicen).
- La pérdida se computa **solo** sobre los tokens seleccionados, no sobre toda la secuencia.

**Datos de preentrenamiento de BERT:** BooksCorpus (800M palabras) + English Wikipedia (2,500M palabras).

**Variantes exploradas:** no usar NSP y entrenar más tiempo (RoBERTa), o enmascarar tokens contiguos (span masking) en vez de tokens aislados.

**Ventajas de MLM:**
- Captura contexto bidireccional verdadero.
- Sirve para múltiples tareas sin cambios arquitectónicos (clasificación, NER, etc.).
- **Limitación:** no se puede usar directamente para generar texto, porque depende de ver ambos lados del token enmascarado.

### ELECTRA: preentrenamiento discriminativo

En vez de predecir tokens enmascarados (generativo), ELECTRA entrena un **discriminador** que debe decidir, para cada token de la entrada, si es el original o fue **reemplazado** por un generador pequeño. Esto aprovecha la señal de **todos** los tokens de la secuencia (no solo el 15% enmascarado), haciendo el preentrenamiento mucho más eficiente en cómputo.

---

## Adapters y LoRA

### Limitaciones del fine-tuning completo

- Recursos computacionales y memoria significativos.
- Mayor riesgo de overfitting en datasets pequeños específicos de tarea.
- Tiempos de entrenamiento largos en modelos grandes.
- Requiere un modelo completo separado por cada tarea (no hay aprendizaje multi-tarea con un solo modelo).

### Adapters

Módulos entrenables **pequeños** insertados dentro de las capas de un modelo preentrenado. Solo se entrenan los adapters; el resto de los pesos queda congelado.

**Ventajas:** menos recursos computacionales, preserva el conocimiento preentrenado, permite multi-tarea reutilizando el mismo modelo base con distintos adapters.

### LoRA (Low-Rank Adaptation)

Hu et al. (2021). Diseñado para fine-tuning eficiente de LLMs grandes.

**Idea:** mantener los pesos preentrenados $W$ congelados, e introducir una actualización de **bajo rango**:

$$W' = W + \Delta W = W + BA$$

donde $A \in \mathbb{R}^{r \times d}$ y $B \in \mathbb{R}^{d \times r}$, con $r \ll d$ (rango bajo). Solo $A$ y $B$ son entrenables.

**Proceso:**
1. Congelar los parámetros preentrenados.
2. Introducir las matrices de bajo rango $A, B$ en cada capa del Transformer.
3. **Forward:** combinar pesos originales + actualización de bajo rango.
4. **Backward:** calcular gradientes solo para $A$ y $B$, y actualizarlas.

**Ventajas:** reduce drásticamente los parámetros entrenables (menor riesgo de overfitting), reduce tiempos de entrenamiento, y facilita cambiar de tarea rápidamente (solo se intercambian las matrices LoRA, no todo el modelo).

---

## Limitaciones de los Encoders para Generación

Los encoders (BERT-like) procesan tokens **bidireccionalmente y en simultáneo**: cada token puede ver todo el resto de la secuencia. Esto es excelente para comprensión, pero **incompatible con generación secuencial**, que requiere generar token por token usando solo el contexto previo (naturaleza autorregresiva). Los decoders, en cambio, procesan de forma autorregresiva, paso a paso.

---

## Encoder-Decoder: BART y T5

| Modelo | Estrategia de preentrenamiento |
|--------|--------------------------------|
| **BART** | Ofusca el texto de entrada en el encoder (token masking, sentence permutation, document rotation, token deletion) y el decoder debe reconstruir el texto original |
| **T5** | "Span corruption": se corrompen tramos (spans) del texto y el objetivo es reconstruir el input original; preentrenado en C4 (Colossal Clean Crawled Corpus) |

Ambos enmarcan el preentrenamiento como una tarea de **denoising**: corromper el input y entrenar al modelo para reconstruirlo, en vez de simplemente predecir la siguiente palabra (LM puro). Una vez preentrenados, pueden fine-tunearse para múltiples tareas downstream (traducción, resumen, Q&A) gracias a su arquitectura encoder-decoder con atención cruzada.

---

## Decoder-Only Models y GPT

Los modelos solo-decodificador están enfocados en **generar texto** dado el contexto previo. Son ideales cuando el output es una secuencia. El objetivo de preentrenamiento es simplemente predecir la próxima palabra dada la historia.

### GPT

Modelo **unidireccional** (solo mira el pasado) basado en la arquitectura Transformer, usando solo la fase de decodificación (sin encoder explícito). Trata la entrada como un prefijo condicional para generar texto directamente, sin necesitar tratamiento especial del prompt.

**Preentrenamiento:** predicción de la siguiente palabra.

```
Input:  El elefante es rosa .
Target:    elefante es rosa . <end>
```

**Fine-tuning (GPT-1):** se adapta el modelo preentrenado a tareas específicas con una capa de salida adicional sobre la representación del modelo.

**Generación de texto:** se genera palabra por palabra usando el conocimiento previo y el contexto, seleccionando cada token según probabilidades. Durante el entrenamiento se usan secuencias "perfectas" (teacher forcing), pero durante la generación los errores se propagan (exposure bias).

### GPT-2: *Language Models are Unsupervised Multitask Learners*

Radford et al. (2019).

- Modelo mucho más grande que GPT-1: 1.5B parámetros (vs 117M).
- Secuencias más largas (1024 vs 512 tokens).
- Entrenado en más datos y mejor curados.
- Sigue preentrenado con predicción de la siguiente palabra.
- Foco en **zero-shot**: resolver tareas sin ningún fine-tuning, solo con el prompt adecuado.

### GPT-3: *Language Models are Few-Shot Learners*

Brown et al. (2020). 175B parámetros — modelos órdenes de magnitud más grandes y costosos.

**In-Context Learning:** en vez de adaptar el modelo a la tarea (fine-tuning), se adapta la **tarea al modelo**: el modelo se usa out-of-the-box, con un input mínimo (instrucciones y/o ejemplos en el propio prompt). La interacción es intuitiva, en lenguaje natural — esto marca el comienzo del "fin del fine-tuning" para muchas tareas.

- Más ejemplos en el prompt (few-shot) → mejor resultado.
- Modelos más grandes aprenden más rápido con menos ejemplos.
- El rendimiento todavía no satura al escalar tamaño de modelo y datos.

---

## Capacidades Emergentes

Habilidades **no programadas explícitamente** que el modelo adquiere y generaliza automáticamente a tareas nuevas: programar, traducir, generar contenido, entender datos estructurados, resumir, responder preguntas, extraer información, seguir instrucciones, etc.

**Observación clave:** las habilidades emergen de manera **impredecible** y no se pueden extrapolar linealmente desde modelos más pequeños.
- GPT-3 (175B parámetros) mostró habilidades limitadas en el benchmark WiC con few-shot.
- Escalar a PaLM (540B parámetros) mejoró significativamente el rendimiento, sin cambios arquitectónicos.

**Hipótesis sobre el origen de la emergencia:**
- Escalas mayores permiten mejor memorización y manejo de tareas complejas.
- El razonamiento multi-paso puede requerir cierta profundidad (número de capas) mínima.
- Modelos con más parámetros almacenan conocimiento del mundo de forma comprimida y eficaz.

La emergencia sigue siendo, en gran parte, un fenómeno **no explicado** teóricamente.

---

## Alucinaciones en LLMs

Generación de texto que es semántica o sintácticamente plausible pero **factualmente incorrecto**, lo cual es difícil de detectar a simple vista porque suena fluido y coherente.

### Tipos

- **Alucinación intrínseca:** el contenido generado contradice directamente la fuente (p.ej., fechas o eventos mal representados).
- **Alucinación extrínseca:** el contenido no puede ser verificado ni contradicho por la fuente (detalles agregados que no están ni a favor ni en contra de lo dado).

### Fidelidad vs. Facticidad

- **Fidelidad:** adherencia al contenido fuente (minimizar alucinación respecto al input).
- **Facticidad:** alineación con hechos del mundo real — puede diferir de la fidelidad según cómo se defina "hecho" (conocimiento del mundo vs. contenido de la fuente).

### Causas

- **Datos:** errores en los datos de entrenamiento; tareas de NLG que fomentan la divergencia (p.ej. creatividad) por diseño.
- **Entrenamiento/modelado:** representaciones imperfectas, errores de decodificación, sesgo de exposición (exposure bias), conocimiento fijo e inherente del modelo (no se actualiza tras el entrenamiento).

### Mitigación

- Mejorar la calidad de los datos de entrenamiento.
- Mejorar los modelos (arquitectura, escala).
- Reinforcement Learning con reward models (RLHF).
- Mecanismos de atención con sesgo condicionado a la fuente.
- **RAG** (Retrieval-Augmented Generation): incorporar fuentes externas verificables.

---

## RAG (Retrieval-Augmented Generation)

```
Input → [Motor de Búsqueda Interno] → Información Externa → [LLM] → Output
```

Combina el poder de generación del modelo con búsqueda en una colección específica de documentos:
1. Búsqueda de documentos relevantes para el input.
2. Generación de la respuesta condicionada a esos documentos.

**Beneficios:** ampliar el contexto con información externa mejora la factualidad de las respuestas. Permite customizar el modelo para usuarios/empresas dándole acceso a documentos internos o personales, sin reentrenar el modelo.

---

## Costo de los LLMs

Los LLMs implican costos significativos de **entrenamiento, hosting e inferencia**, con requerimientos sustanciales de infraestructura (GPUs) y un impacto ambiental no trivial (consumo de energía, emisiones de carbono).

### Destilación (Knowledge Distillation)

**Objetivo:** reducir tamaño y complejidad del modelo, facilitando y acelerando su despliegue.

Técnica para transferir conocimiento de un modelo grande (**maestro**) a uno más pequeño (**estudiante**): el estudiante aprende a imitar las predicciones (o comportamientos) del maestro.

**Destilación simple basada en datos:**
1. Generar un dataset usando las predicciones del modelo maestro.
2. Hacer fine-tuning (instruction tuning) del modelo estudiante preentrenado sobre ese dataset.

### Early Exit

Idea: los hidden states durante la generación tienden a **saturarse** en capas intermedias — seguir procesando en capas superiores puede ser innecesario para ciertos tokens ("overthinking").

**Implementación:** medir métricas de saturación (qué tan lejos está un hidden state del siguiente) o entrenar un clasificador que decida en qué capa "salir" (basado en la confianza del softmax sobre qué token elegir).

**Problemas de Early Exit:**
- **Batching:** hay que esperar a que el último token termine de procesar todas las capas.
- **KV Caching:** si un token sale antes pero otro después, hay que recalcular cachés.
- **Eficiencia:** un token que sale antes no atiende a las capas superiores de los tokens previos.
- **Incertidumbre de costo:** el peor caso sigue siendo el costo de la red completa.

**Skipping** (alternativa a Early Exit): cada posición sale exactamente en la misma capa para todos los tokens (política estática), lo que resuelve el problema de batching pero sacrifica flexibilidad — da un costo predecible.

---

## Toxicidad y Sesgo en LLMs

- **Toxicidad:** lenguaje grosero, irrespetuoso, irrazonable u ofensivo, incluyendo sutilezas de lenguaje dañino.
- **Sesgo:** más allá de la discriminación explícita, incluye preferencias distribucionales sutiles que impactan en la equidad sin ser inmediatamente evidentes. Incluye sesgo de clasificación (falsos positivos) e impacto dispar entre idiomas.

### Fuentes y causas

- **Datos de entrenamiento:** los corpus web a gran escala contienen altos niveles de toxicidad, reflejando sesgos del internet en general.
- **Arquitectura:** las redes pueden aprender y amplificar sesgos sociales inadvertidamente.
- **Input del usuario:** las interacciones pueden provocar respuestas sesgadas o tóxicas (necesidad de robustez ante manipulación).

### Mitigación

- **Prácticas de datos:** mejores métodos de limpieza, evitando sobrefiltración que margine grupos o pierda cobertura.
- **Métodos de evaluación:** técnicas que distingan sesgos abiertos y encubiertos, balanceando seguridad con representación de voces diversas.
- **Entrenamiento adaptativo al dominio:** ajustar el balance entre calidad del modelo y sesgo social — p.ej., vía RLHF.
- **Compromiso crítico del usuario:** educar sobre capacidades y riesgos para un uso responsable.

---

## Reading List

- Clark, K. et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators.
- Houlsby, N. et al. (2019). Parameter-Efficient Transfer Learning for NLP.
- Hu, E. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
- Raffel, C. et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5).
- Chowdhery, A. et al. (2022). PaLM: Scaling Language Modeling with Pathways.
- Google (2023). PaLM 2 Technical Report.
- Radford, A. et al. (2019). Language Models are Unsupervised Multitask Learners (GPT-2).
- Brown, T. et al. (2020). Language Models are Few-Shot Learners (GPT-3).
