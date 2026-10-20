# Screenshot folder

Screenshots captured from the running Docker container:

- `docker-health.png` — `/health`
- `docker-features.png` — `/features`
- `docker-predict-response.png` — sample `/predict` request and response
- `docker-metrics.png` — Prometheus `/metrics`

Screenshots captured from the local Kubernetes deployment:
- `kubernetes-health.png` — `/health` served through the Kubernetes Service
- `kubernetes-predict-response.png` — `/predict` served through the Kubernetes Service
- `kubernetes-prometheus-targets.png` — API scrape target shown as `UP`
- `kubernetes-prometheus-graph.png` — Prometheus query for API target health

The container is published locally on port 8001:
- http://localhost:8001/docs
- http://localhost:8001/health
- http://localhost:8001/features
- http://localhost:8001/metrics

The local Kubernetes API and Prometheus UIs are reachable via port forwarding:
- http://localhost:8080/docs
- http://localhost:9090/targets

The short silent demo video assembled from the Docker and Kubernetes captures is at `../docs/assignment-demo.mp4`. Re-create it after refreshing screenshots with `swift scripts/create_demo_video.swift screenshots docs/assignment-demo.mp4`.
