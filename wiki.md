# PERODUA Robotic Arm FastAPI Backend

> Internal wiki for the PERODUA Robotic Arm FastAPI backend service. This service acts as the central hub between the robotic arm hardware, computer vision models, depth camera input, and the P-Circle cloud platform.

---

## Table of Contents

- [System Overview](#system-overview)
- [Architecture](#architecture)
- [API Contract](#api-contract)
- [Communication Protocols](#communication-protocols)
- [Component Descriptions](#component-descriptions)

---

## System Overview

The FastAPI backend serves as the central orchestrator for the PERODUA robotic arm system. It bridges hardware control (robotic arm via TCP socket), computer vision pipelines (YOLOv2.6 + DCP/GeoTransformer), depth camera input, and the external P-Circle cloud platform via HTTP.

---

## Architecture

![Untitled Diagram.drawio.png](uploads/9208e35131d2b26a3a4907700ebe7b3f/Untitled_Diagram.drawio.png){width="671" height="502"}

---

## API Contract

Base URL: `http://<host>:<port>`

### `GET /robot`

Retrieves the current status of the robotic arm.

**Request**

```
GET /robot
```

No request body required.

**Response**

```json
{
  "status": "idle | running | error",
  "position": {
    "x": 0.0,
    "y": 0.0,
    "z": 0.0
  },
  "timestamp": "2025-01-01T00:00:00Z"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Current operational state of the robot |
| `position` | object | Last known joint/end-effector position |
| `timestamp` | string | ISO 8601 timestamp of last update |

---

### `POST /robot`

Sends a command to the robotic arm. Command parameters are passed via the request headers and/or JSON body.

**Request**

```
POST /robot
Content-Type: application/json
```

**Headers**

| Header | Required | Description |
|--------|----------|-------------|
| `X-Command-Type` | Yes | Type of command (e.g., `move`, `grip`, `reset`) |
| `X-Session-ID` | Optional | Session identifier for tracking |

**Request Body**

```json
{
  "command": "move",
  "target": "charge|stop|pause|home"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `command` | string | Action to perform (`move`, `grip`, `release`, `reset`) |
| `target` | object | Target coordinates for the end-effector |
| `speed` | float | Movement speed factor (0.0 – 1.0) |

**Response**

```json
{
  "accepted": true,
  "message": "Command dispatched to robotic arm",
  "timestamp": "2025-01-01T00:00:00Z"
}
```

---

## Communication Protocols

| Connection | Protocol | Direction | Description |
|------------|----------|-----------|-------------|
| FastAPI :left_right_arrow: Robotic Arm | **Raw TCP Socket** | Bidirectional | FastAPI sends commands/coordinates; arm sends feedback |
| FastAPI :left_right_arrow: P-Circle Cloud | **HTTP** | Bidirectional | FastAPI receives commands from cloud; sends feedback |
| FastAPI :left_right_arrow: YOLOv2.6 | Internal | Bidirectional | FastAPI sends image stream; receives coordinate stream |
| FastAPI :left_right_arrow: DCP/GeoTransformer | Internal | Bidirectional | FastAPI sends point cloud snapshot; receives transformation data |
| FastAPI :left_right_arrow: Depth Camera | Internal | Bidirectional | FastAPI triggers snapshot/video stream; receives RGB-D point clouds |

---

## Component Descriptions

### FastAPI (Central Hub)

The core backend service. Handles all routing between hardware and software components. Exposes the HTTP REST API for the P-Circle cloud and internally manages TCP socket connections to the robotic arm.

### Robotic Arm

Physical arm hardware. Communicates over a **raw TCP socket**. Receives movement commands and coordinates from FastAPI and returns operational feedback.

### YOLOv26 Model

Object detection model. Receives a continuous image stream from the depth camera (via FastAPI) and outputs a coordinate stream back to FastAPI for downstream command generation.

### DCP/GeoTransformer

Point cloud registration model. Receives point cloud snapshots from the depth camera and outputs 6DoF transformation data to FastAPI for pose estimation.

### Depth Camera Input (RGB-D)

Intel RealSense or equivalent RGB-D sensor. Produces RGB images and depth point clouds. FastAPI controls snapshot and video stream triggers.

### P-Circle Cloud

External cloud platform. Communicates with FastAPI over **HTTP**. Sends high-level commands and receives system feedback.

---

_Last updated: May 2025 — Robopreneur Sdn. Bhd._