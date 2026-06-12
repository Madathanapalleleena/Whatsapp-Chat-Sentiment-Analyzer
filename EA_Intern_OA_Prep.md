# EA Software Engineer Intern (Systest Engineer) — OA Prep Guide
*Role: EADP | Focus: Load/Stress Testing, C++/Scala/Python, Cloud, Distributed Systems*

---

## 1. Assessment Format (What to Expect)

| Section | Type | Details |
|---------|------|---------|
| Coding | 2–3 problems | Easy–Medium DSA, Python/C++ preferred |
| MCQs | 15–30 questions | Testing concepts, cloud, databases, networking |
| Scenario/Work Sim | Optional | Situational testing judgment questions |
| Behavioral | Optional | Short video/written responses |

**Tips:** Tackle easiest problem first. For MCQs, eliminate wrong answers. No negative marking usually.

---

## 2. Coding — DSA Patterns (Python Focus)

### Arrays & Strings
```python
# Two-pointer
def two_sum_sorted(arr, target):
    l, r = 0, len(arr) - 1
    while l < r:
        s = arr[l] + arr[r]
        if s == target: return [l, r]
        elif s < target: l += 1
        else: r -= 1

# Sliding window — max sum subarray of size k
def max_subarray_sum(arr, k):
    window = sum(arr[:k])
    max_sum = window
    for i in range(k, len(arr)):
        window += arr[i] - arr[i - k]
        max_sum = max(max_sum, window)
    return max_sum
```

### HashMap / Frequency
```python
from collections import Counter, defaultdict

# Frequency count
def top_k_frequent(nums, k):
    count = Counter(nums)
    return sorted(count, key=count.get, reverse=True)[:k]
```

### Binary Search
```python
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target: return mid
        elif arr[mid] < target: lo = mid + 1
        else: hi = mid - 1
    return -1
```

### BFS / DFS (Graph/Tree)
```python
from collections import deque

def bfs(graph, start):
    visited = set([start])
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited

def dfs(graph, node, visited=None):
    if visited is None: visited = set()
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited
```

### Complexity Reference

| Structure | Access | Search | Insert | Delete |
|-----------|--------|--------|--------|--------|
| Array     | O(1)   | O(n)   | O(n)   | O(n)   |
| Hash Map  | O(1)   | O(1)   | O(1)   | O(1)   |
| BST       | O(log n) | O(log n) | O(log n) | O(log n) |
| Heap      | O(n)   | O(n)   | O(log n) | O(log n) |

### Practice Problems (LeetCode)
- Two Sum (#1) — Easy
- Valid Parentheses (#20) — Easy
- Merge Intervals (#56) — Medium ← very relevant (simulating time ranges)
- LRU Cache (#146) — Medium ← Redis relevance
- Number of Islands (#200) — Medium ← distributed nodes
- Top K Frequent Elements (#347) — Medium

---

## 3. Software Testing Concepts (HIGH PRIORITY — MCQ)

### Types of Testing
| Type | Description |
|------|-------------|
| Unit Testing | Test individual functions/modules |
| Integration Testing | Test interactions between components |
| System Testing | Test the entire system end-to-end |
| **Load Testing** | Test behavior under expected load |
| **Stress Testing** | Push system beyond limits to find breaking point |
| **Performance Testing** | Measure latency, throughput, resource usage |
| Regression Testing | Ensure new changes don't break existing functionality |
| Smoke Testing | Quick sanity check before deeper testing |
| Functional Testing | Does the feature do what it should? |

### Load vs Stress Testing
```
Load Testing:
  → Simulate expected number of concurrent users
  → Measure: response time, throughput, error rate
  → Goal: verify system handles normal + peak load

Stress Testing:
  → Push beyond normal capacity (2x, 5x, 10x users)
  → Find: breaking point, failure modes, recovery behavior
  → Goal: identify bottlenecks before production
```

### Key Metrics to Know
```
Throughput     → requests per second (RPS) the system handles
Latency        → time from request to response (P50, P95, P99)
Error Rate     → % of failed requests under load
Concurrency    → number of simultaneous users/threads
CPU/Memory     → resource utilization under load
Connection pool → max DB/network connections available
```

### P50/P95/P99 Percentiles
```
If P99 latency = 2s:
  → 99% of requests complete in under 2 seconds
  → 1% take longer (outliers)

P50 = median, P95 = 95th percentile, P99 = 99th percentile
Always look at P99 for production SLAs — worst-case user experience.
```

### Test Design Process (EA Specific)
1. **Analyze game logs** → understand real traffic patterns
2. **Define scenarios** → login surge, matchmaking burst, in-game events
3. **Build simulation** → C++/Scala code mimicking game call patterns
4. **Set up distributed environment** → multiple load generators
5. **Run test** → ramp up, sustain, ramp down
6. **Analyze results** → latency graphs, error spikes, resource saturation
7. **Report** → root cause any failures

---

## 4. C++ Basics (MCQ / Scenario)

```cpp
// Key concepts likely tested
#include <iostream>
#include <vector>
#include <map>
#include <thread>
#include <mutex>

// Pointers vs References
int x = 10;
int* ptr = &x;    // pointer — can be null, reassigned
int& ref = x;     // reference — cannot be null, cannot reassign

// Smart pointers (modern C++)
#include <memory>
auto sp = std::make_shared<int>(42);  // shared ownership
auto up = std::make_unique<int>(42);  // unique ownership

// Multithreading basics
std::mutex mtx;
void thread_safe_fn() {
    std::lock_guard<std::mutex> lock(mtx);  // auto-release on scope exit
    // critical section
}

// Templates (generic code)
template<typename T>
T max_val(T a, T b) { return (a > b) ? a : b; }
```

### C++ MCQ Traps
- `new` allocates on heap — must `delete` or use smart pointer
- Virtual functions → runtime polymorphism (vtable)
- `const` correctness: `const int*` vs `int* const` vs `const int* const`
- Stack vs heap: stack = local vars (fast, auto-cleanup); heap = dynamic alloc
- Copy vs move semantics: `std::move` avoids expensive copies

---

## 5. Scala Basics (MCQ)

```scala
// Immutable by default
val x = 42       // immutable (use this)
var y = 42       // mutable

// Collections
val list = List(1, 2, 3)
val mapped = list.map(_ * 2)          // List(2, 4, 6)
val filtered = list.filter(_ > 1)     // List(2, 3)
val reduced = list.reduce(_ + _)      // 6

// Case classes (common in Scala simulation code)
case class GameEvent(playerId: String, eventType: String, timestamp: Long)
val e = GameEvent("p1", "LOGIN", 1234567890L)
// Automatically gets: equals, hashCode, copy, toString

// Pattern matching
e.eventType match {
  case "LOGIN"    => println("Player logged in")
  case "MATCH"    => println("Match started")
  case _          => println("Unknown event")
}

// Futures (async/concurrency — critical for load simulation)
import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global

val result: Future[Int] = Future {
  // simulate API call
  Thread.sleep(100)
  42
}
```

### Scala Key Concepts for Systest
- **Akka** — actor model for concurrent load simulation
- **Gatling** — Scala-based load testing framework (very relevant to this role)
- **Futures/Promises** — async game call simulation
- **Collections API** — map/filter/reduce for processing game logs

---

## 6. Cloud — AWS (HIGH PRIORITY)

### Core Services for Systest

| Service | Systest Use Case |
|---------|-----------------|
| EC2 | Run load generators / test agents |
| S3 | Store test logs, results, game log archives |
| CloudWatch | Monitor metrics during tests (CPU, latency, errors) |
| ECS/EKS | Deploy distributed test infrastructure in containers |
| SQS | Queue-based load simulation (game events) |
| RDS (MySQL) | Game data backend under test |
| ElastiCache (Redis) | Session/cache layer being stress tested |
| VPC | Isolated test network environment |
| Auto Scaling | Observe how system scales under load |

### Cloud Concepts (MCQ)
```
Horizontal scaling   → add more instances (scale out)
Vertical scaling     → bigger instance (scale up)
Load balancer        → distribute traffic (ALB, NLB)
Auto Scaling Group   → automatically add/remove EC2 instances
Region / AZ          → geographic area / data center within region
SLA                  → Service Level Agreement (e.g., 99.9% uptime)
SLO                  → Service Level Objective (e.g., P99 < 200ms)
```

---

## 7. Databases — MySQL & Redis

### MySQL (Relational)
```sql
-- Likely tested: reads under concurrent load
SELECT player_id, COUNT(*) as sessions, AVG(duration) as avg_duration
FROM game_sessions
WHERE start_time > '2024-01-01'
GROUP BY player_id
HAVING COUNT(*) > 10
ORDER BY avg_duration DESC
LIMIT 100;

-- JOIN
SELECT p.name, s.score
FROM players p
INNER JOIN scores s ON p.id = s.player_id
WHERE s.game_id = 'FIFA24';

-- Index — critical for performance testing
CREATE INDEX idx_session_player ON game_sessions(player_id, start_time);
-- Without index → full table scan (O(n)) = slow under load
-- With index → O(log n) lookup

-- EXPLAIN — analyze query plan
EXPLAIN SELECT * FROM players WHERE player_id = 'abc123';
```

### Redis (In-Memory Cache/Session Store)
```
Data structures:
  String    → SET key value / GET key          (session tokens)
  Hash      → HSET user:1 name "Alice"         (player profile)
  List      → LPUSH queue event1               (event queue)
  Set       → SADD online:players "p1" "p2"   (online player set)
  Sorted Set→ ZADD leaderboard 1500 "p1"      (ranked leaderboard)

Key commands:
  SET key value EX 3600    → set with TTL (expiry)
  TTL key                  → check time-to-live
  INCR counter             → atomic increment (request counter)
  EXPIRE key 60            → set expiry
  DEL key                  → delete

Why Redis for gaming:
  → Sub-millisecond reads (vs MySQL's ms-range)
  → Leaderboards via sorted sets
  → Session management (TTL auto-expiry)
  → Rate limiting with INCR + EXPIRE
```

---

## 8. Grafana & Monitoring (HIGH PRIORITY for this role)

### What Grafana Does
- Visualizes time-series metrics during/after load tests
- Connects to: Prometheus, CloudWatch, InfluxDB, MySQL, Elasticsearch

### Key Metrics to Visualize in Systest
```
During load test, watch for:
  1. Response time (P50/P95/P99 over time)
  2. Throughput (RPS — requests per second)
  3. Error rate (% 5xx responses)
  4. CPU utilization (server/container)
  5. Memory usage (look for memory leaks under sustained load)
  6. DB connection pool saturation
  7. Cache hit rate (Redis hits vs misses)
  8. Thread pool queue depth (indicates backpressure)
```

### Reading a Grafana Dashboard
- **Flat line that suddenly spikes** → saturation point hit
- **Latency increases while throughput stays flat** → bottleneck in DB or thread pool
- **Error rate rises as RPS increases** → capacity limit found
- **Memory slowly grows** → memory leak under load

---

## 9. Networking & Distributed Systems (MCQ)

### Networking Basics
```
TCP vs UDP:
  TCP → reliable, ordered, connection-based (HTTP, game state)
  UDP → unreliable, fast, connectionless (game position updates)

HTTP status codes:
  200 OK, 201 Created, 400 Bad Request
  401 Unauthorized, 403 Forbidden, 404 Not Found
  429 Too Many Requests (rate limit) ← important in load tests
  500 Internal Server Error, 503 Service Unavailable

Latency contributors:
  Network RTT + Serialization + Queue time + Processing + DB query

Common ports:
  80 HTTP, 443 HTTPS, 3306 MySQL, 6379 Redis, 22 SSH
```

### Distributed Systems Concepts
```
CAP Theorem:
  Consistency   → every read gets latest write
  Availability  → every request gets a response
  Partition tolerance → system works despite network splits
  → Can only guarantee 2 of 3

Common patterns:
  Circuit Breaker  → stop calling failing service (prevent cascade)
  Retry + Backoff  → retry failed calls with exponential backoff
  Rate Limiting    → cap requests per second per client
  Bulkhead         → isolate failures (thread pools per service)
  Timeout          → don't wait forever for slow dependencies

Bottleneck types in load tests:
  CPU bound    → computation too heavy
  I/O bound    → waiting on DB/disk/network
  Memory bound → GC pressure, heap exhaustion
  Thread bound → thread pool exhausted, requests queuing
```

---

## 10. Test Automation (MCQ + Scenario)

### Load Testing Tools (Know These)
| Tool | Language | Best For |
|------|----------|---------|
| **Gatling** | Scala | High-performance, code-based scenarios — likely used at EA |
| **JMeter** | Java/GUI | Enterprise testing, easy to configure |
| **Locust** | Python | Easy to write, good for distributed testing |
| **k6** | JavaScript | Modern, CI/CD friendly |
| **wrk / hey** | CLI | Quick HTTP benchmarks |

### Gatling (Scala) — Know the Pattern
```scala
import io.gatling.core.Predef._
import io.gatling.http.Predef._

class GameLoginSimulation extends Simulation {

  val httpProtocol = http
    .baseUrl("https://api.ea.com")
    .acceptHeader("application/json")

  val loginScenario = scenario("Player Login")
    .exec(
      http("Login Request")
        .post("/auth/login")
        .body(StringBody("""{"username":"player1","password":"pass"}"""))
        .check(status.is(200))
    )
    .pause(1)  // think time between requests

  setUp(
    loginScenario.inject(
      rampUsers(1000).during(60)  // ramp to 1000 users over 60s
    )
  ).protocols(httpProtocol)
   .assertions(
     global.responseTime.percentile3.lt(500),   // P99 < 500ms
     global.successfulRequests.percent.gt(99)   // >99% success
   )
}
```

### Python Load Testing (Locust)
```python
from locust import HttpUser, task, between

class GamePlayer(HttpUser):
    wait_time = between(1, 3)  # think time

    @task(3)                   # weight 3 = 3x more frequent
    def get_player_stats(self):
        self.client.get("/api/player/stats")

    @task(1)
    def update_score(self):
        self.client.post("/api/score", json={"score": 1500})

# Run: locust -f locustfile.py --host=https://api.ea.com
```

---

## 11. Troubleshooting Approach (Scenario Questions)

When asked "how would you investigate X issue during a load test":

```
Step 1: Identify WHAT is failing
  → Check error rate, error codes (500? 503? timeout?)
  → Check which endpoint/service is affected

Step 2: Isolate WHERE the bottleneck is
  → CPU/Memory graphs (server saturation?)
  → DB slow query logs (query taking too long?)
  → Thread pool metrics (requests queuing?)
  → Network latency (external dependency slow?)

Step 3: Check WHEN it started
  → Correlate timeline: when did RPS cross X? When did latency spike?
  → Was there a deployment? Config change? Dependent service issue?

Step 4: Test vs System issue
  → Is the test itself misconfigured? (Too many connections from 1 IP?)
  → Is the system genuinely at capacity?
  → Reproduce with controlled load to confirm

Step 5: Report
  → Document: load level, observed behavior, root cause hypothesis
  → Recommend: fix, infrastructure change, or capacity limit documentation
```

---

## 12. Behavioral / EA-Specific

### EA Values to Align With
- **Player-first**: Every service supports the player experience
- **Reliability**: Games need 99.99% uptime during launches
- **Innovation**: Proactively improve test coverage and automation
- **Collaboration**: Work with dev teams to resolve perf issues
- **Ownership**: You own the test result and its accuracy

### Common Questions + Answers

**"Why EA / Why Systest?"**
> "EA serves 300M+ players globally — the scale of testing needed is unlike anything else. I'm excited about the mix of performance engineering, distributed systems, and real impact: if a test I design catches a bottleneck before an EA Sports FC launch, millions of players have a better experience."

**"Tell me about a time you debugged a hard problem"**
> STAR: Situation → what was broken. Task → your responsibility. Action → how you isolated the root cause. Result → what you fixed and how you prevented recurrence.

**"How do you handle working independently on an unclear task?"**
> Break it into sub-problems, research existing patterns, identify the smallest experiment to validate your hypothesis, then communicate progress early.

**"What do you know about stress testing?"**
> "It's about finding the system's breaking point by generating load beyond normal capacity. Key things to measure: where latency degrades, what the error rate becomes, and whether the system recovers gracefully after the load drops."

**Questions to Ask EA:**
- "What does a realistic game-day scenario look like — for example, how do you model an EA Sports FC launch weekend?"
- "Which load testing frameworks does the team use — Gatling, JMeter, something custom?"
- "How does the team collaborate with game studio engineering when a test surfaces a bottleneck?"

---

## 13. 1-Day Study Plan

| Time | Focus | Action |
|------|-------|--------|
| 9–10:30 AM | Testing Concepts | Sections 3, 8, 11 — load/stress testing, Grafana, troubleshooting |
| 10:30–12 PM | Databases | Section 7 — MySQL queries + Redis commands |
| 12–1 PM | Break | — |
| 1–2:30 PM | Cloud + Networking | Sections 6, 9 — AWS services, distributed systems |
| 2:30–4 PM | Coding | Section 2 — Solve 3 LeetCode problems (Two Sum, Merge Intervals, LRU Cache) |
| 4–5 PM | C++ / Scala | Sections 4, 5 — MCQ prep, key concepts |
| 5–6 PM | Behavioral | Section 12 — write STAR answers out loud |
| Evening | Light review | Quick reference below. Sleep well. |

---

## 14. Quick Reference Cheat Sheet

```
LOAD TESTING
  Throughput   → RPS the system handles
  P99 latency  → worst-case 1% of requests
  Stress test  → find breaking point (not just peak normal)
  Bottleneck   → CPU / I/O / Memory / Thread pool

REDIS
  String       → session tokens, counters (INCR)
  Sorted Set   → leaderboards (ZADD, ZRANK)
  TTL          → auto-expiry (SET key val EX 3600)

MYSQL
  INDEX        → O(log n) lookup vs O(n) scan
  EXPLAIN      → show query execution plan
  GROUP BY + HAVING → aggregate + filter aggregates

AWS (Systest)
  EC2          → run load generators
  CloudWatch   → monitor CPU, latency, errors
  SQS          → queue-based event simulation
  ElastiCache  → Redis layer

NETWORKING
  TCP          → reliable, ordered (game state)
  UDP          → fast, lossy (position updates)
  Circuit Breaker → stop calling failing services
  429          → rate limit hit

C++ TRAPS
  stack = local/auto, heap = new/delete
  shared_ptr → shared ownership, auto-delete
  mutex + lock_guard → thread-safe critical section

SCALA (Gatling)
  rampUsers(N).during(T)  → gradual load ramp
  assertions: P99 < X ms, success% > 99
  Future → async game call simulation

TROUBLESHOOT ORDER
  1. What failed (error code, endpoint)
  2. Where (CPU/DB/network/thread pool)
  3. When (correlate timeline with load level)
  4. Test issue vs system issue
  5. Document and report
```

---

**Good luck on your EA Online Assessment!**

*Prepared for EA Software Engineer Intern (Systest Engineer) — EADP | 2026*
