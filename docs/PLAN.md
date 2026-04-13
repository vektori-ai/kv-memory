KV Memory System
Tier 2 — End-to-End Build Plan
Optimized for Claude Code · Self-hosted LLMs · HuggingFace / llama.cpp / vLLM

0. What you are building
A compute-saving, attention-native retrieval system. Not “memory for LLMs” in the fuzzy sense — a system that stores KV tensors from past interactions, retrieves the relevant ones via two-stage ranking, and injects them directly into attention layers. The model skips prefill on retrieved context. Cost is paid once at write time, never again.

Without this libraryEvery turn: full prefill over all retrieved context. Cost scales with history length.With this libraryRetrieve top-K KV blocks, inject directly. Only current tokens hit prefill. History is free.vs RAGRAG re-prefills retrieved text every turn. You inject KV once-stored, zero re-prefill.Model constraintKV is model-locked. Same model across runs = reusable store. Different model = new store.Framework constraintFramework-agnostic via adapters. HF / llama.cpp / vLLM all share same KV store for same model.
1. Repository structure
Give Claude Code this exact structure as the target. Every module maps to a design decision we have locked in.

kvmemory/
  core/
    chunker.py          # semantic chunking of conversation turns
    importance.py       # perplexity + signal scoring, importance filter
    retrieval.py        # two-stage retrieval: coarse + rerank
    injector.py         # KV injection + doc-wise RoPE + token budget
    queue.py            # async write queue (Redis or in-process)
  adapters/
    base.py             # BaseAdapter interface
    hf_adapter.py       # HuggingFace transformers
    llamacpp_adapter.py # llama.cpp C API
    vllm_adapter.py     # vLLM (degraded: no inject, hidden states only)
  storage/
    vector_db.py        # Qdrant wrapper (retrieval vectors)
    kv_store.py         # local blob store (KV tensors, layerwise)
    schema.py           # KVBlock dataclass, VectorEntry dataclass
  memory.py             # KVMemory: public API surface
  config.py             # Config dataclass
tests/
  test_chunker.py
  test_retrieval.py
  test_injection.py
  test_hf_adapter.py
  beam_eval.py          # BEAM benchmark runner
requirements.txt
README.md

2. Data schemas
Define these first. Every other module depends on them. Claude Code should generate schema.py before touching anything else.
2.1 KVBlock
@dataclass
class KVBlock:
    block_id:          str              # uuid4
    model_id:          str              # e.g. 'llama-3-8b'
    session_id:        str
    agent_id:          str | None
    shared:            bool             # cross-agent retrieval allowed

    # Retrieval keys (live in vector DB)
    hidden_vecs:       dict[int, np.ndarray]  # layer -> [d_model] float32
                                              # weighted mean pool, sqrt(n) norm, L2 norm
    token_count:       int
    chunk_text:        str              # original text, for debug + recompute

    # KV tensors (live in blob store)
    kv_by_layer:       dict[int, tuple[np.ndarray, np.ndarray]]
                       # layer -> (K, V) each shape [seq, heads, dim] int8
    quant_scales:      dict[int, tuple[float, float]]
                       # layer -> (K_scale, V_scale) for dequant
    original_positions: list[int]       # for doc-wise RoPE

    # Lifecycle
    created_at:        float            # unix timestamp
    last_accessed:     float
    access_count:      int
    importance_score:  float            # 0.0 to 1.0
2.2 Config
@dataclass
class KVMemoryConfig:
    model_id:          str
    retrieval_layers:  list[int]        # e.g. [8, 16, 24] for 32-layer model
                                        # target 25%, 50%, 75% of depth
    store_layers:      list[int]        # same as retrieval_layers by default
    token_budget:      int = 2000       # HARD CAP on injected tokens
    coarse_top_k:      int = 200        # stage 1 candidates
    final_top_k:       int = 10         # stage 2 winners
    importance_threshold: float = 0.3  # drop below this
    dedup_threshold:   float = 0.95     # skip if too similar to existing
    async_write:       bool = True
    qdrant_url:        str = 'localhost:6333'
    blob_store_path:   str = './kv_store'

3. Adapter interface (base.py)
This is the contract every backend must implement. The rest of the library only calls these methods. Zero framework-specific code anywhere outside adapters.
class BaseAdapter(ABC):

    @abstractmethod
    def capture(
        self,
        tokens: list[int],
        text: str,
        layers: list[int]
    ) -> tuple[dict[int, tuple[Tensor, Tensor]], dict[int, Tensor]]:
        '''
        Run tokens through model up to max(layers).
        Returns:
          kv_by_layer: layer -> (K, V) tensors [seq, heads, dim] float16
          hidden_by_layer: layer -> hidden states [seq, d_model] float32
        '''

    @abstractmethod
    def inject_and_generate(
        self,
        blocks: list[KVBlock],
        current_tokens: list[int],
        generation_kwargs: dict
    ) -> GenerationOutput:
        '''
        Prepend retrieved KV blocks to context using doc-wise RoPE.
        Only current_tokens hit prefill.
        blocks must be ordered: block_0 gets pos [0..n0],
                                block_1 gets pos [0..n1], ...
                                query gets pos [num_blocks..]
        '''

    @abstractmethod
    def supports_kv_inject(self) -> bool:
        '''HF + llamacpp: True. vLLM: False.'''

    @property
    @abstractmethod
    def d_model(self) -> int: ...

    @property
    @abstractmethod
    def num_layers(self) -> int: ...

    @property
    @abstractmethod
    def num_heads(self) -> int: ...

Claude Code instruction: implement HFAdapter first. It is the only adapter needed to prove the system works. llamacpp and vLLM adapters are Phase 2.

4. Write path (async)
The write path runs after every response, asynchronously. The user sees zero added latency. If the process crashes before completion, that turn's memory is lost — acceptable tradeoff.
4.1 Chunker (chunker.py)
Split conversation text into semantic chunks at sentence boundaries. Target 80–150 tokens per chunk.
def chunk_turn(text: str, tokenizer, target_tokens=100) -> list[str]:
    sentences = split_sentences(text)          # use spacy or simple regex
    chunks = []
    current, count = [], 0
    for sent in sentences:
        n = len(tokenizer.encode(sent))
        if count + n > target_tokens and current:
            chunks.append(' '.join(current))
            current, count = [sent], n
        else:
            current.append(sent)
            count += n
    if current:
        chunks.append(' '.join(current))
    return chunks
4.2 Importance scorer (importance.py)
Score each chunk before storing. Drop below threshold. This is Gate 1 — reduces write volume by ~60-70%.
def score_importance(
    chunk_text: str,
    model_loss: float,          # cross-entropy on chunk tokens
    baseline_loss: float,       # model's average loss
    explicit_signal: float = 0.0  # 1.0 if user flagged 'remember this'
) -> float:
    perplexity_score = max(0, model_loss - baseline_loss) / baseline_loss
    return 0.6 * perplexity_score + 0.4 * explicit_signal
Perplexity proxy: high loss = model found this surprising = worth remembering. Low loss = model already knew it = low value to store.
4.3 Dedup check (storage/vector_db.py)
Before storing, check if a near-identical block already exists. If cosine similarity > 0.95, skip write and increment the existing block's access_count instead.
def is_duplicate(hidden_vec: np.ndarray, model_id: str, threshold=0.95) -> str | None:
    results = qdrant.search(
        collection_name=model_id,
        query_vector=hidden_vec.tolist(),
        limit=1,
        score_threshold=threshold
    )
    return results[0].id if results else None
4.4 Adapter capture
Run tokens through model up to max(retrieval_layers). Extract KV tensors and hidden states at each target layer in one forward pass.
kv_by_layer, hidden_by_layer = adapter.capture(
    tokens=tokenizer.encode(chunk_text),
    text=chunk_text,
    layers=config.retrieval_layers
)
4.5 Hidden vector computation
This is your Stage 2 retrieval key. Three steps must happen in this exact order.
def compute_retrieval_vec(hidden: Tensor, token_count: int) -> np.ndarray:
    # Step 1: weighted mean pool (salience-weighted, not flat average)
    # use attention entropy as token salience weight
    weights = F.softmax(-entropy(hidden), dim=0)   # low entropy = salient
    pooled = (weights.unsqueeze(-1) * hidden).sum(0)  # [d_model]

    # Step 2: sqrt(n) length normalization (removes chunk length bias)
    pooled = pooled / math.sqrt(token_count)

    # Step 3: L2 normalize (stable cosine comparison)
    pooled = F.normalize(pooled, dim=0)
    return pooled.cpu().numpy().astype(np.float32)
One retrieval vec per layer. So retrieval_layers=[8,16,24] produces three vecs per block stored in vector DB.
4.6 Quantization (INT8)
Quantize K and V tensors before writing to blob store. 2x size reduction vs float16, minimal quality loss.
def quantize_int8(tensor: Tensor) -> tuple[np.ndarray, float]:
    scale = tensor.abs().max().item() / 127.0
    quantized = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized.cpu().numpy(), scale
4.7 Write to stores
# Vector DB entry (one per block, holds all layer retrieval vecs)
qdrant.upsert(collection=model_id, points=[PointStruct(
    id=block_id,
    vector={f'layer_{l}': vec for l, vec in hidden_vecs.items()},
    payload={
        'session_id': session_id,
        'agent_id': agent_id,
        'shared': shared,
        'token_count': token_count,
        'chunk_text': chunk_text,
        'created_at': time.time(),
        'importance_score': importance_score,
        'access_count': 0,
    }
)])

# Blob store (pickled KVBlock to disk, keyed by block_id)
blob_path = f'{config.blob_store_path}/{model_id}/{block_id}.pkl'
with open(blob_path, 'wb') as f:
    pickle.dump(kv_block, f)
4.8 Async queue (queue.py)
The write path is enqueued immediately after response, processed in background. Never blocks the user.
class WriteQueue:
    def __init__(self):
        self._queue = asyncio.Queue()
        self._worker = None

    async def enqueue(self, session_id, tokens, text, adapter, config):
        await self._queue.put((session_id, tokens, text, adapter, config))

    async def _process(self):
        while True:
            item = await self._queue.get()
            try:
                await _run_write_pipeline(*item)   # full write path above
            except Exception as e:
                logger.error(f'Write failed: {e}')  # never crash user session
            finally:
                self._queue.task_done()

5. Read path
The read path runs at the start of every new turn, synchronously, before generation. Target total latency < 50ms.
5.1 Stage 1: Coarse filter (retrieval.py)
Embed the query using hidden states at retrieval layers. Search vector DB for top-200 candidates by cosine similarity. This is fast — ANN over pre-computed normalized vectors.
async def stage1_coarse(
    query_tokens: list[int],
    adapter: BaseAdapter,
    config: KVMemoryConfig,
    session_filter: dict | None = None
) -> list[str]:                        # returns block_ids

    # partial forward pass to deepest retrieval layer
    _, hidden_by_layer = adapter.capture(
        tokens=query_tokens,
        text='',
        layers=config.retrieval_layers
    )

    # compute normalized query vec per layer (same 3 steps as write)
    query_vecs = {
        l: compute_retrieval_vec(h, len(query_tokens))
        for l, h in hidden_by_layer.items()
    }

    # named vector search in Qdrant (one query per layer, merge results)
    candidate_scores = {}
    weights = {l: w for l, w in zip(config.retrieval_layers, [0.25, 0.50, 0.25])}
    for layer, q_vec in query_vecs.items():
        results = qdrant.search(
            collection_name=config.model_id,
            query_vector=(f'layer_{layer}', q_vec.tolist()),
            limit=config.coarse_top_k,
            query_filter=session_filter    # filter by model_id, shared, etc
        )
        for r in results:
            candidate_scores[r.id] = (
                candidate_scores.get(r.id, 0) + weights[layer] * r.score
            )

    # sort by combined score, return top coarse_top_k
    sorted_ids = sorted(candidate_scores, key=candidate_scores.get, reverse=True)
    return sorted_ids[:config.coarse_top_k]
5.2 Stage 2: Rerank with MMR (retrieval.py)
Rerank the 200 candidates using weighted cosine similarity. Apply Maximal Marginal Relevance (MMR) to prevent picking near-duplicate blocks.
def stage2_rerank_mmr(
    candidate_ids: list[str],
    query_vecs: dict[int, np.ndarray],
    config: KVMemoryConfig,
    token_budget: int,
    mmr_lambda: float = 0.7    # 1.0 = pure relevance, 0.0 = pure diversity
) -> list[str]:

    # fetch lightweight payloads (no KV tensors yet)
    payloads = qdrant.retrieve(ids=candidate_ids, with_vectors=True)

    # compute relevance scores (weighted cosine across layers)
    def relevance(payload) -> float:
        score = 0
        layer_weights = {l: w for l, w in
                         zip(config.retrieval_layers, [0.25, 0.50, 0.25])}
        for l, w in layer_weights.items():
            q = query_vecs[l]
            k = np.array(payload.vector[f'layer_{l}'])
            score += w * float(np.dot(q, k))  # both L2-normalized -> cosine
        return score

    # MMR selection loop
    selected = []
    selected_vecs = []
    tokens_used = 0
    remaining = list(payloads)

    while remaining and tokens_used < token_budget:
        best, best_score = None, -999
        for cand in remaining:
            rel = relevance(cand)
            red = max((np.dot(np.array(cand.vector[f'layer_{config.retrieval_layers[1]}']),
                              sv) for sv in selected_vecs), default=0)
            score = mmr_lambda * rel - (1 - mmr_lambda) * red
            if score > best_score:
                best, best_score = cand, score
        if best is None:
            break
        n_tokens = best.payload['token_count']
        if tokens_used + n_tokens > token_budget:
            break
        selected.append(best.id)
        selected_vecs.append(np.array(best.vector[f'layer_{config.retrieval_layers[1]}']))
        tokens_used += n_tokens
        remaining.remove(best)

    return selected
5.3 Fetch full KV tensors (storage/kv_store.py)
Only now, after MMR selection, do you load the actual tensor blobs. Update access metadata asynchronously.
def fetch_blocks(block_ids: list[str], model_id: str, blob_path: str) -> list[KVBlock]:
    blocks = []
    for bid in block_ids:
        path = f'{blob_path}/{model_id}/{bid}.pkl'
        with open(path, 'rb') as f:
            block = pickle.load(f)
        blocks.append(block)
    # update access metadata async (fire and forget)
    asyncio.create_task(_update_access(block_ids))
    return blocks
5.4 Injection with doc-wise RoPE (injector.py)
This is the core mechanism. Each retrieved block is treated as an independent document with positions starting from 0. The current query gets positions offset by the number of retrieved blocks.
def inject_and_generate(
    adapter: BaseAdapter,
    blocks: list[KVBlock],
    current_tokens: list[int],
    generation_kwargs: dict
) -> GenerationOutput:

    if not adapter.supports_kv_inject():
        # degraded mode: inject as text prefix instead
        prefix = ' '.join(b.chunk_text for b in blocks)
        return adapter.generate(prefix + current_tokens, generation_kwargs)

    # doc-wise RoPE: each block gets independent positions [0..n]
    # query gets positions [num_blocks..num_blocks+len(current)]
    # this means NO re-rotation needed - positions are always valid
    return adapter.inject_and_generate(blocks, current_tokens, generation_kwargs)

The HFAdapter implements inject_and_generate by prepending past_key_values. Each block's KV is dequantized on the fly (int8 -> float16) and concatenated at each layer before the forward pass begins. The adapter must enforce consistent block ordering: same blocks in same order = same position encoding = reproducible attention.

6. HuggingFace adapter (adapters/hf_adapter.py)
Build this first. It is the only adapter needed for Phase 1. All other adapters follow the same interface.
6.1 Capture
class HFAdapter(BaseAdapter):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._num_layers = model.config.num_hidden_layers
        self._d_model = model.config.hidden_size
        self._num_heads = model.config.num_attention_heads

    def capture(self, tokens, text, layers):
        input_ids = torch.tensor([tokens]).to(self.model.device)
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
                output_attentions=False,
                use_cache=True
            )
        kv_by_layer = {
            l: (out.past_key_values[l][0].squeeze(0),  # K [heads, seq, dim]
                out.past_key_values[l][1].squeeze(0))  # V [heads, seq, dim]
            for l in layers
        }
        hidden_by_layer = {
            l: out.hidden_states[l].squeeze(0)         # [seq, d_model]
            for l in layers
        }
        return kv_by_layer, hidden_by_layer
6.2 Inject and generate
    def inject_and_generate(self, blocks, current_tokens, gen_kwargs):
        # dequantize and build past_key_values from blocks
        # shape expected by HF: list of (K, V) per layer
        # K, V: [batch, heads, seq, head_dim]
        n_layers = self.model.config.num_hidden_layers
        combined_kv = [None] * n_layers

        for layer_idx in range(n_layers):
            K_parts, V_parts = [], []
            for block in blocks:
                if layer_idx in block.kv_by_layer:
                    K_q, V_q = block.kv_by_layer[layer_idx]
                    k_scale, v_scale = block.quant_scales[layer_idx]
                    K = torch.from_numpy(K_q.astype(np.float32)) * k_scale
                    V = torch.from_numpy(V_q.astype(np.float32)) * v_scale
                    K_parts.append(K.to(self.model.device))
                    V_parts.append(V.to(self.model.device))
            if K_parts:
                K_cat = torch.cat(K_parts, dim=-2).unsqueeze(0)  # [1,h,seq,d]
                V_cat = torch.cat(V_parts, dim=-2).unsqueeze(0)
                combined_kv[layer_idx] = (K_cat, V_cat)

        # run generation with prepended KV
        input_ids = torch.tensor([current_tokens]).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                past_key_values=combined_kv if any(x for x in combined_kv) else None,
                **gen_kwargs
            )
        return out

7. Public API surface (memory.py)
This is what developers integrate. Two meaningful lines of change from their existing code.
class KVMemory:
    def __init__(self, adapter: BaseAdapter, config: KVMemoryConfig):
        self.adapter = adapter
        self.config = config
        self.vector_db = VectorDB(config)
        self.kv_store = KVStore(config)
        self.write_queue = WriteQueue()

    async def generate(
        self,
        prompt: str,
        session_id: str,
        agent_id: str | None = None,
        retrieve_shared: bool = False,
        generation_kwargs: dict = {}
    ) -> GenerationOutput:

        tokens = self.adapter.tokenizer.encode(prompt)

        # --- READ PATH ---
        session_filter = build_filter(
            model_id=self.config.model_id,
            session_id=session_id,
            agent_id=agent_id,
            retrieve_shared=retrieve_shared
        )
        candidate_ids = await stage1_coarse(tokens, self.adapter,
                                            self.config, session_filter)
        query_vecs = compute_query_vecs(tokens, self.adapter, self.config)
        final_ids = stage2_rerank_mmr(candidate_ids, query_vecs, self.config,
                                      self.config.token_budget)
        blocks = self.kv_store.fetch(final_ids)

        # --- GENERATE ---
        output = inject_and_generate(self.adapter, blocks, tokens,
                                     generation_kwargs)

        # --- WRITE (async, non-blocking) ---
        response_text = self.adapter.tokenizer.decode(output.sequences[0])
        full_turn = prompt + ' ' + response_text
        await self.write_queue.enqueue(
            session_id, tokens, full_turn, self.adapter, self.config
        )

        return output

    def close(self):
        self.write_queue.shutdown()

Developer integration (2 lines changed):
# Before
output = model.generate(input_ids)

# After
memory = KVMemory(adapter=HFAdapter(model, tokenizer), config=cfg)
output = await memory.generate(prompt, session_id='user_123')

8. Non-negotiable constraints
Claude Code must enforce all of these. They are not optional optimizations.

ConstraintRuleWhyToken budgetHard cap at config.token_budget (default 2000). MMR stops when budget hit. Never exceeded.Attention cost is O((retrieved + current)^2). Injecting too much recreates long-context cost.Async writeWrite pipeline never runs in request path. Always enqueued. User sees zero added latency.Forward pass + quantize + embed + storage = ~400-500ms. Unacceptable synchronous cost.model_id enforcementEvery vector DB query filters on model_id first. KV from different model is never returned.KV tensors are weight-coupled. Wrong model KV injected = garbage attention, corrupted output.Doc-wise RoPE orderingBlocks must be injected in consistent order. Block positions always [0..n]. Query always offset by num_blocks.Inconsistent ordering changes position encoding per request. Non-reproducible behavior.No pooling across layersEach layer's K and V stored and retrieved independently. Never average across layers.Layers operate in different geometric spaces. Cross-layer averaging is mathematically meaningless.INT8 quantizationQuantize on write, dequantize on inject. Store scale factors alongside tensors.Float16 KV at 3 layers = ~50MB/chunk. INT8 = ~25MB. Required for storage viability.Importance filterDrop chunks below config.importance_threshold before any storage. Gate 1.Without filtering, every turn is stored. 10k users x 100 convos x 40 chunks = TBs quickly.Dedup checkBefore writing, cosine check against existing blocks. Skip if similarity > 0.95.Repeated context (agent loops, system prompts) creates near-duplicate blocks. Waste storage.
9. Build order for Claude Code
Execute strictly in this order. Each phase is independently testable before moving to the next.
Phase 1 — Prove KV reuse works (3–5 days)
•	schema.py — KVBlock and Config dataclasses
•	adapters/base.py — abstract interface
•	adapters/hf_adapter.py — capture() and inject_and_generate() only
•	storage/kv_store.py — local disk blob store (pickle, no optimization)
•	storage/vector_db.py — Qdrant wrapper, single named vector per layer
•	core/retrieval.py — stage1 only (no MMR yet), single layer
•	memory.py — synchronous generate() (no async yet)
•	tests/test_hf_adapter.py — capture and inject a known block, verify output changes
•	Goal: inject a manually-created KV block and verify model output differs from baseline
Phase 2 — Make it real (1–2 weeks)
•	core/chunker.py — sentence-boundary chunker
•	core/importance.py — perplexity-based importance scorer
•	Upgrade retrieval.py — multi-layer (3 layers), weighted cosine combination
•	Upgrade retrieval.py — stage2 MMR reranking
•	core/queue.py — async write queue
•	Add INT8 quantization to write path
•	Add dedup check to write path
•	Add token budget enforcement to read path
•	Upgrade memory.py — async generate()
•	tests/test_retrieval.py — verify stage1 + stage2 pipeline returns correct blocks
•	Goal: full end-to-end: conversation turn gets stored, next turn retrieves and injects it
Phase 3 — Benchmark (3–5 days)
•	beam_eval.py — run BEAM benchmark against RAG baseline and sliding window baseline
•	Measure: answer accuracy per question type (IE, multi-hop, knowledge update, temporal)
•	Measure: prefill token count per turn (cost reduction metric)
•	Measure: latency breakdown (stage1, stage2, fetch, inject, generate)
•	Goal: beat RAG baseline on information extraction and multi-hop reasoning categories
Phase 4 — Adapters (1 week each)
•	adapters/llamacpp_adapter.py — C API binding for KV capture and inject
•	adapters/vllm_adapter.py — degraded mode (hidden states only, inject as text)

10. What NOT to build yet
Claude Code should refuse to implement these until Phase 3 is complete and benchmarks confirm value.

Text embeddings (Tier 1)We skip this entirely. Tier 2 only. No BGE, no E5, no text ANN search.Trained router projectorModel-specific, requires training data. Phase 4+ if benchmarks justify.KV mergingDedup skip is sufficient for Phase 1-2. Merging adds complexity for marginal gain.Hot/cold tieringLocal disk blob store is fine for Phase 1-2. S3/Redis tiering is Phase 4.Eviction scorerImportance filter + dedup handles growth in Phase 1-2. Full LRU eviction later.Feedback loop on attention massExpensive, framework-specific. Not worth it until Phase 3 benchmarks reveal gaps.vLLM full KV injectvLLM's block manager doesn't expose injection cleanly. Degraded mode is fine for now.Multi-model supportOne model per store. Model switching = new config. No migration needed yet.
11. Testing strategy
These are the three tests that prove the system actually works, not just that the code runs.
Test 1: KV reuse correctness
Store a KV block manually. Run generation with and without injection. Assert outputs differ. Assert injected output is more contextually relevant.
def test_kv_injection_changes_output():
    block = create_kv_block('The capital of France is Paris', adapter)
    store.write(block)
    q = 'What is the capital of France?'
    out_without = adapter.generate(q)
    out_with = memory.generate(q, session_id='test')
    assert 'Paris' in decode(out_with)    # model uses the injected memory
    assert out_without != out_with          # injection changed output
Test 2: Prefill token count reduction
Verify that injected tokens do NOT appear as prefill tokens in the generation call.
def test_prefill_reduction():
    blocks = fetch_top_k_blocks(query)
    injected_token_count = sum(b.token_count for b in blocks)
    prefill_count = count_prefill_tokens(adapter, blocks, query)
    assert prefill_count == len(query_tokens)  # only query hits prefill
    assert prefill_count < len(query_tokens) + injected_token_count  # blocks skipped
Test 3: Retrieval recall
Store 100 known blocks. Query for each one. Assert it appears in top-10 results.
def test_retrieval_recall_at_10():
    blocks = [store_known_block(i) for i in range(100)]
    hits = 0
    for block in blocks:
        results = retrieve(block.chunk_text, top_k=10)
        if block.block_id in [r.id for r in results]:
            hits += 1
    recall = hits / 100
    assert recall > 0.80    # target: 80%+ recall@10

12. Dependencies

torch >= 2.0Core tensor operations, model forward passtransformers >= 4.40HuggingFace model interface (HFAdapter)qdrant-client >= 1.8Vector DB for retrieval keys (named vectors support)numpy >= 1.24KV tensor storage formatasyncioAsync write queue (stdlib)spacy >= 3.0 (optional)Sentence boundary detection for chunkerpytest + pytest-asyncioTest suitegguf (optional)llama.cpp adapter (Phase 2)vllm (optional)vLLM adapter (Phase 2)
Qdrant must be running locally before any test. Start with: docker run -p 6333:6333 qdrant/qdrant. Create one collection per model_id with named vectors: one per retrieval layer.

13. Qdrant collection setup
Create this collection before any writes. Named vectors allow per-layer similarity search in one collection.
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient('localhost', port=6333)

# create collection with one named vector per retrieval layer
# retrieval_layers = [8, 16, 24], d_model = 4096 (llama-3-8b example)
client.create_collection(
    collection_name='llama-3-8b',
    vectors_config={
        'layer_8':  VectorParams(size=4096, distance=Distance.COSINE),
        'layer_16': VectorParams(size=4096, distance=Distance.COSINE),
        'layer_24': VectorParams(size=4096, distance=Distance.COSINE),
    }
)
Use Distance.COSINE because vectors are L2-normalized at write time. Qdrant will use dot product internally (equivalent for unit vectors) which is fast.

Summary
This plan specifies Tier 2 of the KV Memory System end-to-end. Every design decision has a reason. Every constraint is non-negotiable. The build order is optimized to prove correctness before adding complexity.

Core insightPay prefill cost once at write. Zero at inject. Token budget caps attention overhead.RetrievalTwo-stage: hidden-state coarse ANN + weighted cosine MMR rerank across 3 layers.Retrieval keysWeighted mean pool -> sqrt(n) normalize -> L2 normalize. Stable, no training needed.InjectionDoc-wise RoPE: each block pos [0..n], query offset by num_blocks. No re-rotation.StorageINT8 KV tensors layerwise in blob store. Normalized hidden vecs in Qdrant.Write pathAsync queue. Importance filter. Dedup check. Never blocks user response.First adapterHuggingFace only. past_key_values for inject. output_hidden_states for capture.First benchmarkBEAM dataset. Beat RAG on information extraction and multi-hop reasoning.
