# Preprocess Pipeline

This preprocessing pipeline is designed with a clear separation between:

- **runner / orchestration**
- **stage transforms**
- **I/O and checkpoint policy**
- **metrics / reporting**

The goal is to keep stage logic modular while avoiding unnecessary reads, writes, and checkpoints between every stage.

## Design principles

- Each **stage transform** should do only one thing:  
  **dataset in → dataset out**
- The **runner** is responsible for:
  - reading input data
  - calling stages in order
  - deciding whether to checkpoint
  - writing final outputs
  - logging metrics and reports
- **Logical stages** are not the same as **physical checkpoints**
- Checkpoints should only be used when needed for:
  - resumability
  - expensive recomputation boundaries
  - debugging / inspection

---

## Pipeline overview

```mermaid
flowchart TD

    R[Preprocess Runner] --> A[Read Input Dataset]
    A --> B[Optional Debug or Sampling Limit]

    B --> C[Canonicalize]
    C --> C1[Optional Parseable View filter parse_ok]

    C --> D[Minhash]
    C1 --> D

    D --> E[Pairs]
    E --> F[Cluster Map]
    F --> G[Snapshot]
    G --> H[Split]
    H --> I[Final Outputs]

    I --> J[Quality Reports]
    I --> K[Manifest and Done Files]

    F -. optional checkpoint .-> FC[Checkpoint Cluster Map]
    G -. optional checkpoint .-> GC[Checkpoint Snapshot]
    H -. final write .-> HC[Write Train Val Test]