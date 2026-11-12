# Diseño: Mini sistema de Retail Video Analytics (multi-cámara)

**Fecha:** 2026-11-12
**Autor:** Juan Diego Ossa (con apoyo de Claude)
**Horizonte:** 1 semana, dedicación de días completos

## Intención y por qué

Juan Diego se presenta a un puesto de Computer Vision enfocado en retail analytics a
gran escala (300+ tiendas, 6000+ cámaras): multi-camera tracking, Person Re-ID,
conteo de visitantes únicos, detección de interacción cliente-producto y
optimización de inferencia en GPU.

Su CV (ver proyectos en Tecondor y Capiro) prueba dominio productivo de detección,
clasificación y conteo de objetos con YOLOv8/OpenVINO, pipelines de datos
offline-first (SQLite + S3) y backends en FastAPI/Flask/NestJS. Lo que **no** aparece
en ningún proyecto anterior es el núcleo específico de esta oferta: seguimiento de
una *identidad* (no un objeto estático), asociación de esa identidad entre cámaras
distintas, y heurísticas de interacción persona-producto.

Este proyecto existe para cerrar ese hueco concreto con algo que pueda mostrar y
defender técnicamente en la entrevista — no para repetir lo que el CV ya demuestra.

## Goals

1. Pipeline de detección + tracking multi-objeto por cámara (YOLOv8 + ByteTrack o
   BoT-SORT).
2. Re-ID de personas entre 2 cámaras usando embeddings de un backbone preentrenado,
   con deduplicación de identidades (conteo de visitantes únicos).
3. Medición de dwell time por zona/cámara.
4. Backend FastAPI + SQLite exponiendo esas métricas (reutiliza su stack conocido).
5. Benchmark de optimización GPU: PyTorch vs TensorRT (FPS, VRAM) en single-stream y
   dual-stream concurrente, usando la RTX disponible.
6. Documento final que responda, con evidencia del propio proyecto, las 4 preguntas
   filtro de la oferta (conteo de únicos, pick-up detection, identidad persistente
   entre cámaras, costos de procesar miles de streams).

### Stretch goal (solo si el tiempo alcanza después de 1-5)

7. Detección de interacción mano-producto: YOLOv8-pose + heurística espacial/temporal
   sobre una mesa de prueba con objetos.

## Non-goals

- Dashboard/frontend con UI pulida — con un endpoint y/o un script que grafique
  resultados basta.
- Entrenar un modelo de detección o de Re-ID desde cero.
- Arquitectura cloud/distribuida real, autenticación, multi-tenancy.
- NVIDIA DeepStream — demasiada fricción de setup en Windows para el tiempo
  disponible; se documenta como ruta de producción pero no se implementa.
- Procesar literalmente miles de streams — se demuestra el patrón con 2 cámaras y se
  documenta por escrito cómo escalaría (batching, sharding por GPU, edge vs cloud).

## Anti-goals (qué haría fallar el proyecto aunque "funcione")

- Que el Re-ID sea una caja negra que Juan Diego no pueda explicar en la entrevista
  (por qué ese backbone, qué falla, qué métrica lo valida).
- Que el conteo de únicos funcione solo por casualidad con 2 personas de prueba y se
  caiga con cualquier variación de luz/ángulo sin que el documento lo reconozca como
  limitación conocida.
- Optimizar por tener "más features" en vez de profundidad defendible en lo que el
  job post pregunta explícitamente.

## Constraints

- Hardware: 1 PC con **RTX 3050 6GB**, webcam integrada/USB + cámara ELP (USB) como
  las 2 fuentes de video. 6GB es limitado: se usan variantes ligeras de YOLOv8
  (n/s, no l/x) y, si el stretch goal de pose estimation se construye, corre en un
  paso separado del pipeline principal en vez de simultáneo, para no competir por
  VRAM con detección + tracking + Re-ID + export de TensorRT.
- Tiempo: 1 semana, trabajo en solitario.
- **Fuente de video:** el pipeline es agnóstico a la fuente (archivo de video o
  cámara en vivo — mismo código, solo cambia de dónde vienen los frames). Desarrollo
  y validación corren sobre **clips grabados** con las 2 cámaras, no en vivo: así
  los tests son deterministas y reproducibles (pytest necesita inputs fijos, no un
  webcam cambiando cada corrida). Las cámaras en vivo quedan soportadas para la demo
  final, no como vía principal de prueba.
- SO: Windows. Librerías pesadas de Linux-only (DeepStream) quedan fuera por esto
  mismo, no solo por tiempo.
- Experiencia previa: sólida en detección/backend, nula en tracking/Re-ID — hay
  curva de aprendizaje real, no solo de implementación.

## Arquitectura

```
┌─────────────┐        ┌─────────────┐
│  Webcam PC  │        │  Cámara ELP │
│  (zona A)   │        │  (zona B)   │
└──────┬──────┘        └──────┬──────┘
       │ frames                │ frames
       ▼                       ▼
┌─────────────────────────────────────┐
│   Worker de cámara (uno por stream)  │
│   YOLOv8 detect → track() ByteTrack  │
│   cada N frames: embedding Re-ID     │
│   emite: (cam_id, local_track_id,    │
│           embedding, bbox, ts)       │
└──────────────────┬────────────────┬─┘
                    │                │
                    ▼                ▼
          ┌───────────────────────────┐
          │  Identity Resolver        │
          │  (galería de embeddings   │
          │   + similitud coseno +    │
          │   ventana temporal)       │
          │  asigna global_id         │
          └──────────────┬────────────┘
                          ▼
                ┌───────────────────┐
                │  SQLite            │
                │  tracks, eventos,  │
                │  dwell_time        │
                └─────────┬──────────┘
                          ▼
                ┌───────────────────┐
                │  FastAPI           │
                │  /visitors/unique  │
                │  /visitors/{id}    │
                │  /events           │
                └───────────────────┘
```

Todo corre en un solo proceso Python con 2 tareas async (una por cámara) para
simplicidad — no se monta mensajería distribuida (Kafka/Redis) porque con 2 fuentes
no aporta nada real; el documento final explica cómo ese mismo patrón de "worker por
cámara + resolver central" es el que se escala horizontalmente en producción.

## Decisiones ya tomadas (y por qué)

- **Tracker:** ByteTrack como default (viene integrado en Ultralytics,
  `model.track(tracker="bytetrack.yaml")`); se deja BoT-SORT como comparación rápida
  si hay tiempo, porque la oferta nombra ambos y vale la pena poder hablar de la
  diferencia (BoT-SORT usa apariencia + cámara motion compensation, ByteTrack es más
  liviano).
- **Re-ID:** embeddings con un backbone preentrenado tipo OSNet (p. ej. vía
  `torchreid`/`deep-person-reid`, pretrained en Market-1501) en vez de entrenar algo
  propio — no hay dataset propio ni tiempo para eso, y en producción tampoco se
  entrena Re-ID desde cero.
- **Asociación entre cámaras:** similitud coseno entre embeddings + umbral +
  ventana temporal (no tiene sentido comparar con alguien visto hace 3 horas) —
  heurística simple, documentada, no un modelo de asociación aprendido.
- **Persistencia:** SQLite — mismo patrón offline-first que ya usó en Capiro, no
  tiene sentido introducir Postgres para 2 cámaras.
- **Backend:** FastAPI (ya lo domina) para exponer métricas, no para servir un
  frontend.
- **Optimización GPU:** exportar a TensorRT vía `model.export(format="engine")` y
  comparar FPS/VRAM contra PyTorch puro, en vez de DeepStream (overkill de setup
  para esta escala y este SO).
- **Interacción producto (stretch):** pose estimation (YOLOv8-pose) + regla
  espacial "muñeca dentro de zona producto + permanece N frames" — heurística
  explícita, no un modelo de HOI entrenado.

## Riesgos técnicos y plan B (mi llamada, documentada para que la objetes si no estás de acuerdo)

- **Export a TensorRT en Windows** puede fallar por incompatibilidad de versiones
  CUDA↔cuDNN↔TensorRT↔driver (la misma clase de fricción que motivó excluir
  DeepStream). Día 1 se verifica rápido con un modelo de prueba; si no cierra en
  ~1 hora de troubleshooting, el plan B es benchmark PyTorch vs ONNX Runtime con
  CUDA Execution Provider — sigue siendo una comparación de optimización de
  inferencia válida para la entrevista, solo que no es TensorRT puro.
- **`torchreid`/`deep-person-reid`** tiene mantenimiento irregular y puede no
  instalar limpio contra el PyTorch/CUDA actuales. Plan B si falla: usar
  `fast-reid` como alternativa, o en última instancia un ResNet50 preentrenado en
  ImageNet (vía torchvision) con global average pooling como embedding de
  apariencia — más débil que un backbone entrenado para Re-ID, pero documentado
  explícitamente como degradación consciente, no como ocultamiento.

## Protocolo de validación del Re-ID (mi llamada, documentada)

Sin fecha de entrevista externa confirmada, se trata la semana como presupuesto
autoimpuesto; si algo se complica, se recorta en este orden: stretch goal de
interacción con producto → comparación BoT-SORT vs ByteTrack → benchmark TensorRT
(cae a ONNX Runtime, ver riesgos arriba).

**Actualización:** en vez de depender solo de grabación propia, se usa un dataset
público con licencia abierta como fuente principal de validación, porque trae
ground truth de identidad por persona entre cámaras (permite medir precision/recall
real del Re-ID, no solo "funcionó una vez"):

- **Primario: EPFL CVLAB multi-camera pedestrian (terrace/passageway)** —
  3-4 cámaras parcialmente superpuestas, liviano, uso de investigación permitido.
  Encaja con el storytelling de "zona A / zona B" del proyecto y permite tener el
  pipeline corriendo el primer día.
- **Extensión opcional si sobra tiempo: WILDTRACK** (EPFL, 7 cámaras muy
  superpuestas, versión reducida ~10GB en Kaggle bajo CC0) — ground truth de
  identidad más riguroso, para un benchmark de Re-ID más fuerte si el tiempo
  alcanza.
- **Explícitamente descartado: DukeMTMC** — fue retirado por Duke por violación de
  privacidad/consentimiento de los sujetos grabados; no se usa pese a ser muy
  citado en la literatura de Re-ID, por riesgo ético y porque el proyecto lleva el
  nombre de Juan Diego en GitHub.

Además, se graba un clip corto propio con webcam + ELP (aunque sea con 1-2
personas) para la demo final en vivo — el dataset público valida el algoritmo con
rigor, la grabación propia demuestra que el pipeline funciona con hardware real,
no solo con benchmarks académicos.

## Qué queda para el builder (fase de implementación)

Estas son decisiones de lógica de negocio/algoritmo con trade-offs reales — se
resuelven escribiendo código, no en este documento:

- Umbral exacto de similitud coseno para considerar "misma persona" y cómo se
  calibra (con qué datos de prueba).
- Tamaño de la ventana temporal de asociación entre cámaras.
- Cada cuántos frames se recalcula el embedding Re-ID de un track (costo vs.
  precisión).
- La heurística exacta de "tomó un producto" si se llega al stretch goal.
- Esquema concreto de las tablas SQLite y de los endpoints FastAPI.

Esto se detalla en el plan de implementación (paso siguiente, no en este doc).
