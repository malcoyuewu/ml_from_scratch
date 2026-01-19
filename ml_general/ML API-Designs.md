API Calls

1. Recommedation endpoint
  ### API Design for Snapchat Lens Recommendation ML System

Building on the end-to-end ML system design for Snapchat Lens recommendations, the API layer serves as the interface between the mobile app (client) and the backend services. It needs to be RESTful (or gRPC for internal efficiency), secure (OAuth/JWT auth), rate-limited, and optimized for mobile (low payload sizes, async where possible). The design focuses on real-time recommendations, feedback ingestion for model improvement, and admin endpoints for monitoring/scaling.

We'll use HTTPS, JSON payloads, and standard HTTP status codes (e.g., 200 OK, 429 Too Many Requests). Assume deployment on Kubernetes (as per the system design), with API Gateway (e.g., Kong or AWS API Gateway) for routing, auth, and rate-limiting.

#### Key Endpoints
Here's a breakdown of the core APIs, grouped by functionality. I've prioritized low-latency paths (e.g., recs endpoint) and included error handling (e.g., retries for transient failures).

1. **Recommendation Endpoint** (Core: Fetch personalized Lenses)
   - **Method**: GET
   - **Path**: `/v1/lens_recs`
   - **Query Params**:
     - `user_id` (string, required): Unique user identifier.
     - `context` (JSON string, optional): Real-time signals like {"camera_features": {"faces_detected": 2, "mood": "happy"}, "location": "urban", "time_of_day": "evening"}. Base64-encoded if complex.
     - `num` (integer, optional, default: 10): Number of recommendations (max 20 to avoid overload).
     - `session_id` (string, optional): For A/B testing variants.
   - **Headers**: Authorization (Bearer token), Content-Type: application/json.
   - **Response** (200 OK):
     ```json
     {
       "recommendations": [
         {
           "lens_id": "abc123",
           "name": "Funny Face Filter",
           "preview_url": "https://cdn.snapchat.com/previews/abc123.jpg",
           "category": "beauty",
           "score": 0.95,
           "reason": "Matches detected faces and happy mood"
         },
         // ... more items
       ],
       "request_id": "xyz-456"  // For tracing
     }
     ```
   - **Errors**: 400 (invalid params), 429 (rate limit exceeded), 503 (service unavailable during spikes).
   - **Rate Limit**: 50 requests/min per user to prevent abuse.

2. **Feedback Endpoint** (For model retraining loops)
   - **Method**: POST
   - **Path**: `/v1/feedback`
   - **Body** (JSON, required):
     ```json
     {
       "user_id": "user123",
       "lens_id": "abc123",
       "action": "apply",  // Options: "apply", "skip", "save", "share"
       "duration_ms": 5000,  // Optional: Engagement time
       "context": {"reason": "too glitchy"}  // Optional: User notes
     }
     ```
   - **Headers**: Authorization, Content-Type: application/json.
   - **Response** (202 Accepted): Empty body (async processing via Kafka).
   - **Errors**: 400 (invalid action), 429.
   - **Rate Limit**: 100/min per user (high-volume for feedback).

3. **Search Endpoint** (Fallback for manual Lens discovery)
   - **Method**: GET
   - **Path**: `/v1/lens_search`
   - **Query Params**:
     - `query` (string, required): Keywords like "dog filter".
     - `filters` (JSON string, optional): {"category": "animals", "popularity": "trending"}.
     - `num` (integer, optional, default: 10).
   - **Response**: Similar to `/v1/lens_recs`, but without personalization scores.
   - **Purpose**: Hybrid with ML—boost results using embeddings from Pinecone.

4. **Admin/Monitoring Endpoints** (Internal, restricted)
   - **GET /v1/health**: Returns {"status": "healthy", "uptime": "24h"} for liveness probes.
   - **POST /v1/models/deploy**: Body: {"model_version": "v2.1"} – Triggers canary releases.
   - **GET /v1/metrics**: Integrates with Prometheus (e.g., query latency, QPS).

#### Security and Best Practices
- **Auth**: OAuth 2.0 for users; API keys for internal services.
- **Validation**: Use schemas (e.g., OpenAPI) to enforce inputs.
- **Logging**: Request IDs for tracing (e.g., via Jaeger).
- **Versioning**: /v1/ for backward compatibility.
- **Edge Integration**: Responses include CDN URLs for previews to offload delivery.

#### Sample API Calls
Here are curl examples for testing/integration. Assume a dev environment at `api.snap-ml.example.com`.

1. **Fetching Recommendations**:
   ```bash
   curl -X GET "https://api.snap-ml.example.com/v1/lens_recs?user_id=user123&context=%7B%22camera_features%22%3A%7B%22faces_detected%22%3A1%7D%7D&num=5" \
   -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
   -H "Content-Type: application/json"
   ```
   Expected Response: JSON array of 5 Lenses, served from edge/K8s inference.

2. **Submitting Feedback**:
   ```bash
   curl -X POST "https://api.snap-ml.example.com/v1/feedback" \
   -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
   -H "Content-Type: application/json" \
   -d '{
     "user_id": "user123",
     "lens_id": "abc123",
     "action": "apply",
     "duration_ms": 3000
   }'
   ```
   This feeds into Kafka for monitoring and retraining.

3. **Searching Lenses**:
   ```bash
   curl -X GET "https://api.snap-ml.example.com/v1/lens_search?query=party%20filters&num=10" \
   -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
   ```
   Response: Non-personalized results, cached in Redis for speed.

These calls integrate with the system's components: Params trigger feature lookups (from DynamoDB/Redis), inference (TensorFlow Serving on GPUs), and CDN delivery.

### Follow-Up Discussion: Scaling the API for a 10x Traffic Spike (e.g., Viral Trend)

In a Snap interview or design review, scaling discussions often arise from scenarios like a viral Lens (e.g., a celebrity-endorsed AR effect causing a surge during events like Coachella or holidays). A 10x spike could mean jumping from ~10K QPS to 100K QPS, stressing APIs, databases, and inference layers. The core response strategy—auto-scaling groups in Kubernetes (K8s), circuit breakers, caching (Redis for ~70% hit rate), with trade-offs on over/under-provisioning and predictive scaling—is spot on. Let's expand with rationale, implementation details, and alternatives.

#### Core Scaling Mechanisms
- **Auto-Scaling Groups in K8s**: Use Horizontal Pod Autoscaler (HPA) to dynamically add/remove pods based on metrics like CPU (target 60% utilization) or custom QPS from Prometheus. For AWS-based K8s (EKS), integrate Cluster Autoscaler for node-level scaling.
  - **Implementation**: Define HPA in YAML: `minReplicas: 10, maxReplicas: 200`. During a spike, it scales up in ~1-2 minutes.
  - **Why effective?**: Handles bursty traffic without manual intervention; pods run TensorFlow Serving for inference.

- **Circuit Breakers**: Implement via libraries like Resilience4j or Istio (in K8s service mesh). If backend latency >200ms or errors >5%, "break" the circuit to fail-fast and return cached/fallback results (e.g., popular Lenses).
  - **Implementation**: In API code, wrap calls to DynamoDB/Pinecone with breakers; retry after 30s cooldown.

- **Caching (Redis for 70% Hit Rate)**: Cache hot responses (e.g., trending Lenses) in Redis (clustered for high availability). Use TTLs (e.g., 5-10min) to balance freshness/ephemerality.
  - **Implementation**: On GET /v1/lens_recs, check Redis first (key: "recs:user123:context_hash"). Aim for 70% hits by caching non-personalized parts; invalidate on feedback.

- **Predictive Scaling Based on Trends**: Use ML tools like AWS Forecast or Prometheus alerts on historical patterns (e.g., spikes during weekends/events). Pre-scale clusters 15-30min before predicted surges.
  - **Implementation**: Monitor via Prometheus queries (e.g., rate(api_requests[5m]) > threshold); trigger via webhooks to K8s.

#### Trade-Off Analysis
Use this table for clarity in discussions:

| Mechanism | Pros | Cons | Mitigation |
|-----------|------|------|------------|
| **Auto-Scaling (HPA)** | Elastic, cost-efficient (scale down post-spike) | Lag time (1-5min) risks initial overload | Combine with over-provisioning baseline (e.g., 20% headroom). |
| **Circuit Breakers** | Prevents cascading failures; graceful degradation | May drop valid requests during breaks | Tune thresholds; provide fallback recs (e.g., static top-10). |
| **Caching (Redis)** | Reduces DB/inference load (70% hits → 3x effective capacity) | Staleness for viral trends; memory cost | Short TTLs; event-driven invalidation (e.g., via Kafka on new virals). |
| **Predictive Scaling** | Proactive (avoids downtime) | Inaccurate predictions waste resources (cost up 20% if false positive) | Refine with ML on logs; hybrid with reactive auto-scaling. |
| **Overall (Over- vs. Under-Provisioning)** | Over: High availability during spikes | Wasteful (idle resources = +20% cost) | Under: Cheaper but risks downtime/SLO breaches (e.g., >100ms latency). Balance via monitoring. |

#### Additional Considerations
- **Database Scaling**: Shard DynamoDB by user_id; use read replicas for PostgreSQL.
- **CDN Offload**: Push static assets (Lens previews) to CloudFront—handles 10x easily.
- **Testing**: Simulate spikes with tools like Locust; monitor SLOs (e.g., 99% <150ms).
- **Alternatives**: Serverless (e.g., Lambda for low-traffic endpoints) vs. K8s (better for ML state). If spikes are global, use multi-region sharding.

This ensures the API remains resilient, aligning with Snap's real-time, user-centric ethos. If you'd like to mock a spike scenario or refine endpoints, let me know!
