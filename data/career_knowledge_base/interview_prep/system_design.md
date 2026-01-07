# System Design Interview Preparation

## Core Concepts

### Fundamental Principles

**Scalability:**
- **Vertical vs horizontal scaling**
- **Load balancing strategies**
- Database scaling approaches

**Reliability:**

- Redundancy
- Failover mechanisms
- Data replication

**Availability:**
- SLA definitions
- Redundancy strategies
- Disaster recovery

**Maintainability:**
- Modular design
- Documentation
- Monitoring and logging

## Design Patterns
### Client-Server Architecture
- REST APIs vs GraphQL
- API gateway pattern
- Service discovery

### Microservices vs Monolith
- Pros and cons
- When to choose each
- Migration strategies

### Databases
**SQL Databases:**
- ACID properties
- Indexing strategies
- Normalization
- Sharding techniques

**NoSQL Databases:**
- Document stores (MongoDB)
- Key-value stores (Redis)
- Column stores (Cassandra)
- Graph databases (Neo4j)

**Caching Strategies:**
- Cache aside
- Write through/behind
- Cache invalidation
- CDN usage

## Common System Components
### Load Balancers
- Round robin
- Least connections
- IP hash
- Health checks

### Message Queues
- Pub-sub pattern
- Message brokers (Kafka, RabbitMQ)
- Ordering guarantees
- Dead letter queues

### Storage Systems
- Object storage (S3)
- Block storage
- File systems
- Content delivery networks

## Design Process
### Step 1: Requirements Gathering
- Clarify functional requirements
- Identify non-functional requirements
- Define constraints
- Estimate scale

### Step 2: High-Level Design
- Sketch architecture diagram
- Identify components
- Define APIs
- Choose technologies

### Step 3: Detailed Design
- Database schema
- Data flow diagrams
- API specifications
- Security considerations

### Step 4: Scaling & Optimization
- Identify bottlenecks
- Caching strategies
- Database optimization
- CDN implementation

## Common Interview Questions
### Beginner Level:
1. Design a URL shortener (like TinyURL)
2. Design a key-value store
3. Design a web crawler

### Intermediate Level:
1. Design Twitter/Twitter feed
2. Design Uber/Lyft
3. Design Netflix/YouTube

### Advanced Level:
1. Design Google Search
2. Design Amazon/AWS
3. Design Facebook social graph

## Estimation Techniques
### Back-of-the-Envelope Calculations
- Storage requirements
- Bandwidth needs
- QPS (Queries per second)
- Latency calculations

### Sample Calculation:
**Designing a photo-sharing app:**
- Users: 1 million DAU
- Photos per user: 10 per day
- Average photo size: 2MB
- Daily storage: 1M × 10 × 2MB = 20TB/day
- Monthly storage: 20TB × 30 = 600TB

## Resources
- **Books:** Designing Data-Intensive Applications
- **Websites:** System Design Primer, High Scalability
- **Practice:** Grokking System Design, Pramp
- **Tools:** Draw.io, Excalidraw for diagrams