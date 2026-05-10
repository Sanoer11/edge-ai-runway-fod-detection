# Sentinel — Edge-AI Runway Foreign Object Debris Detection

A real-time computer vision and multi-sensor fusion system that detects **Foreign Object Debris (FOD)** on airport runways using **YOLOv10s** deployed on **Raspberry Pi 5** — with MQTT-based distributed telemetry, motion-gated inference, and a **Streamlit operations dashboard**. Built as a B.Tech capstone with academic guidance from Dr. Nilam Pradhan and Dr. Netra Lokhande at MIT-WPU Pune.

---

## The problem

A 2-cm metal screw on a runway can puncture a tire, get ingested into a jet engine, or be kicked into another aircraft at takeoff speeds. FOD events have caused fatal crashes — most famously Air France 4590 (Concorde, 2000) — and cost the global aviation industry an estimated $13B annually.

Most existing FOD detection systems (Stratech Tarsier, Xsight FODetect, Trex FODFinder) are radar-based, fixed-installation, and cost **$1M–5M per runway**. We asked a different question: **can a low-cost, distributed edge-AI system using consumer hardware achieve useful detection performance at two orders of magnitude lower cost?**

Sentinel is the answer — at a Bill of Materials of **~₹20,000 per node** (~$240 USD).

---

## System architecture

![Sentinel system architecture](architecture.png)

A camera array captures runway imagery, which flows through a preprocessing stage into the FOD detection module (YOLOv10s). Detection outputs branch to two consumers — an alert and notification system, and a data-logging dashboard for telemetry and review.

The detailed implementation extends this with a multi-sensor MQTT contract, threaded capture, motion-gated inference, and a Singleton-pattern dashboard connection (see *Key technical decisions* below).

---

## Key technical decisions

### 1. YOLOv10s + NCNN FP16 — model and runtime trade-off study

I benchmarked four YOLO variants on the FOD detection task before deploying:

| Model | Params | Backend | mAP@50 | Pi 5 latency | FPS |
|---|---|---|---|---|---|
| YOLOv8n | ~3.2M | PyTorch | 0.68 | ~180 ms | ~5.5 |
| YOLOv8s | ~11.2M | PyTorch | 0.75 | ~480 ms | ~2.1 |
| YOLOv10n | ~2.3M | NCNN (FP16) | 0.71 | ~85 ms | ~11.8 |
| **YOLOv10s** | **~7.2M** | **NCNN (FP16)** | **0.78** | **~155 ms** | **~6.5** |

YOLOv10s was selected over the faster YOLOv10n because **detection accuracy is the binding constraint for a safety-critical FOD application** — missing a runway debris event has materially worse consequences than a few extra ms of inference latency. With motion-gated inference (Section 4) the effective end-to-end frame rate is well above 6.5 FPS.

The model was exported to **NCNN with FP16 quantisation** for further latency reduction on the Pi 5's ARM Cortex-A76 cores. NCNN's ARM-optimised kernels gave a ~2.3× inference speedup over PyTorch native on the same hardware, with no measurable mAP loss from the FP16 conversion.

### 2. Multi-sensor fusion over MQTT

Rather than tightly coupling sensors in code, each modality publishes to its own MQTT topic:

```
fod/node_01/camera/detections   # YOLO outputs (bounding boxes + confidence)
fod/node_01/radar/range         # measured object distances
fod/node_01/pir/motion          # binary motion events
fod/node_01/health              # node liveness, FPS, temperature
```

The MQTT message contract is the interface — meaning **real hardware can replace simulated nodes with zero changes to the dashboard or backend.** Currently tested with a mock-node Python publisher that emits the full sensor stream; real radar + PIR integration on physical Pi nodes is in progress.

### 3. Threaded capture architecture (`queue.Queue`)

A naive capture-then-infer loop bottlenecks on the slower step. Sentinel decouples them with two threads communicating via a bounded queue:

```python
# Producer thread: continuously capture frames
while running:
    frame = camera.read()
    if frame_queue.full():
        frame_queue.get_nowait()   # drop oldest, keep freshest
    frame_queue.put(frame)

# Consumer thread: pull most recent frame, run inference
while running:
    frame = frame_queue.get(timeout=1.0)
    detections = yolo_model(frame)
    publish_to_mqtt(detections)
```

This guarantees inference always runs on the **freshest available frame**, and that capture is never blocked by inference taking longer than the camera's frame interval.

### 4. Motion-gated inference (MOG2 background subtraction)

Running YOLO on every frame is wasteful when most frames show static runway. I apply OpenCV's **MOG2 background subtractor** as a cheap motion gate — if no significant pixel changes are detected, the frame is skipped before the expensive inference call. This **reduced average CPU load by ~62%** in bench testing while maintaining detection on every actual motion event.

### 5. The Streamlit + MQTT threading conflict (and the Singleton fix)

This was the hardest bug in the project. Streamlit re-runs the entire script on every UI interaction. A naive Paho-MQTT client gets re-initialised on every rerun, creating zombie connection threads that eventually saturate the broker.

I resolved it with a **Singleton pattern** using `@st.cache_resource`:

```python
@st.cache_resource
def get_mqtt_client():
    """Singleton MQTT client — survives Streamlit reruns."""
    client = mqtt.Client(client_id="sentinel_dashboard")
    client.connect(BROKER_HOST, BROKER_PORT, keepalive=60)
    client.loop_start()
    return client
```

Combined with `@st.fragment(run_every=0.5)` for the telemetry panels, this gives **isolated 500 ms telemetry refreshes** without re-running the full dashboard or recreating the MQTT connection.

---

## Hardware bill of materials

| Component | Part | Cost (INR) |
|---|---|---|
| SBC | Raspberry Pi 5 (8 GB) | ~9,500 |
| Camera | Raspberry Pi Camera Module 3 | ~3,800 |
| Radar | RCWL-0516 doppler module | ~250 |
| PIR | HC-SR501 motion sensor | ~150 |
| Storage | SanDisk 64 GB microSD (A2) | ~800 |
| Power | 27 W USB-C PSU | ~1,500 |
| Enclosure | IP65 weatherproof case + mounts | ~2,200 |
| Misc | jumper wires, breadboard, fan | ~1,800 |
| **Total** | | **~₹20,000** |

(~$240 USD per node, vs. **$1M+ for commercial radar-based FODDS** systems.)

---

## Setup

### 1. Download model weights

The trained YOLOv10s weights (`best.pt`, ~33 MB) are hosted as a GitHub Release, not in the repo.

**Download from:** [Releases → v0.1](https://github.com/Sanoer11/edge-ai-runway-fod-detection/releases/tag/v0.1)

Place `best.pt` in the project root before running.

### 2. Install dependencies

On the Raspberry Pi (edge node):

```bash
pip install ultralytics ncnn opencv-python paho-mqtt numpy
```

On the laptop (dashboard):

```bash
pip install streamlit paho-mqtt opencv-python pandas plotly
```

### 3. Run the MQTT broker

```bash
mosquitto -p 1883
```

### 4. Run the edge logic (on Pi or simulated node)

```bash
python edge_logic.py
```

### 5. Run the dashboard (on laptop)

```bash
streamlit run dashboard.py
```

The dashboard auto-discovers MQTT-publishing nodes and renders their feeds.

---

## Repository layout

```
edge-ai-runway-fod-detection/
├── edge_logic.py          # Camera capture, motion gating, YOLO inference, MQTT publish
├── dashboard.py           # Streamlit operations dashboard (Singleton MQTT client)
├── architecture.png       # System architecture diagram
├── best.pt                # → Download from Releases (33 MB)
└── README.md
```

---

## Status & limitations

This is a **working B.Tech capstone prototype**, currently ~60% complete on the software side. Honest current state:

- ✅ Edge inference pipeline operational on Raspberry Pi 5 (YOLOv10s + NCNN FP16, ~155 ms / frame)
- ✅ Multi-sensor MQTT contract defined and tested with mock nodes
- ✅ Streamlit dashboard renders live video, sensor status, alert queue
- ✅ Singleton MQTT pattern verified stable across long-running sessions
- ⏳ Full multi-node deployment with real radar + PIR hardware — in progress
- ⏳ Persistent alert log + maintenance work order export — planned
- ⏳ Field validation on a real airfield — planned

### Limitations

- **Detection accuracy is dataset-bound.** Real airfield conditions (rain, glare, varied debris types) will degrade benchmark performance.
- **Single camera per node.** Stereo / depth fusion for 3D object localisation is future work.
- **No formal certification path.** A real deployment would require ICAO / DGCA airfield safety review.

---

## Research output

Alongside the implementation, this project produced:

- 📄 **An IEEE-format conference paper** benchmarking Sentinel against commercial FODDS systems (Tarsier, FODetect, Xsight) on a cost-vs-coverage trade-off curve.
- 📚 **A literature review across 25+ peer-reviewed papers** on FOD detection, edge-AI deployment, and aviation safety automation (2018–2025).

---

## Stack

Python · Ultralytics YOLO · NCNN · OpenCV · Paho-MQTT · Streamlit · Raspberry Pi 5

---

## References

Selected key references from the literature review:

1. Wang et al., *YOLOv10: Real-Time End-to-End Object Detection*, NeurIPS 2024.
2. Cao et al., *FOD-A: A Dataset for Foreign Object Debris in Airport Pavements*, IEEE Access, 2022.
3. Patterson, *Foreign Object Debris (FOD) Detection Research*, FAA Technical Note, 2017.
4. Manufacturer technical briefs: Stratech Tarsier, Xsight FODetect, Trex FODFinder.

---

## Team

- **Sanoer Soni** — Lead developer, edge inference pipeline, MQTT architecture, dashboard implementation
- **Tanisha Jaiswal** — Multi-sensor integration, hardware mounting, field testing
- **Shipra Bagade** — Dataset curation, literature review, IEEE paper drafting

**Academic guidance:** Dr. Nilam Pradhan & Dr. Netra Lokhande, Department of Electronics & Communication Engineering, MIT-WPU Pune.

---

*B.Tech capstone project — School of Electronics & Communication Engineering (AI/ML), MIT World Peace University, Pune.*
*January 2026 – present.*
