# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

- User: orencollaco
- GitHub CLI location: `$HOME/bin/gh` (use full path: `/home/orencollaco/bin/gh`)

## Build Commands

llama.cpp uses CMake as its build system. The Makefile is deprecated and should not be used.

**Note**: This fork focuses exclusively on CPU and CUDA acceleration. Other acceleration methods (Metal, Vulkan, SYCL, HIP, etc.) are not being developed or maintained.

**Primary Focus**: This repository is implementing the **ATLAS architecture** from the paper "ATLAS: Learning to Optimally Memorize the Context at Test Time" (arXiv:2505.23735). ATLAS enables linear-time complexity for long-context inference through test-time memory optimization, achieving unprecedented context lengths with fixed memory requirements.

### Standard builds
```bash
# CPU-only Release build (recommended for CPU)
cmake -B build
cmake --build build --config Release

# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Build with tests
cmake -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build

# Build with server
cmake -B build -DLLAMA_BUILD_SERVER=ON
cmake --build build --config Release
```

### CUDA-accelerated build
```bash
# CUDA build (for NVIDIA GPUs)
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

## Testing

Tests use CMake's CTest framework. Test files are located in `/tests/`.

```bash
# Run all tests
ctest --test-dir build

# Run specific test category
ctest --test-dir build -L main
ctest --test-dir build -L model

# Run a single test by name
ctest --test-dir build -R test-tokenizer-0

# Run with verbose output
ctest --test-dir build -V
```

## Code Quality

### C++ formatting
The project uses clang-format. Configuration is in `.clang-format`.
```bash
# Format specific files
clang-format -i src/llama.cpp

# Format all C++ files
find src common ggml -name "*.cpp" -o -name "*.h" | xargs clang-format -i
```

### Python type checking
```bash
# Type check Python scripts
mypy convert_hf_to_gguf.py
```

## Architecture Overview

### Core Components

- **`/src/`**: Core llama library - model loading, inference engine, sampling, KV cache management
- **`/ggml/`**: GGML tensor library - low-level mathematical operations, backend abstraction, GPU kernels
- **`/common/`**: Shared utilities - argument parsing, logging, sampling helpers, JSON schema handling
- **`/tools/`**: Standalone executables built from the library

### Key Abstractions

1. **Model Loading**: Models are loaded in GGUF format. The library handles memory mapping, quantization, and tensor allocation.

2. **Backend System**: GGML provides a backend abstraction allowing the same code to run on CPU, CUDA, Metal, Vulkan, etc. Backend selection happens at runtime.

3. **Inference Pipeline**: 
   - Context creation (`llama_context`)
   - Batch processing (`llama_batch`)
   - Token generation with sampling
   - KV cache management for efficient inference

4. **Server Architecture**: The HTTP server (`examples/server/`) provides an OpenAI-compatible API, supporting both completion and chat endpoints with streaming.

### Important Files

- `src/llama.cpp`: Core inference implementation
- `ggml/src/ggml.c`: Tensor operations
- `ggml/src/ggml-backend.cpp`: Backend abstraction
- `common/common.cpp`: Shared utilities
- `examples/server/server.cpp`: HTTP server implementation

## Common Development Tasks

### Converting models to GGUF
```bash
# From Hugging Face format
python convert_hf_to_gguf.py <model_path> --outfile model.gguf

# Quantize model
./build/bin/llama-quantize model.gguf model-q4_0.gguf q4_0
```

### Running inference
```bash
# CLI inference
./build/bin/llama-cli -m model.gguf -p "Hello, world"

# Start server
./build/bin/llama-server -m model.gguf --port 8080
```

### Local CI testing
```bash
# Run CI locally before pushing
mkdir -p tmp
bash ./ci/run.sh ./tmp/results ./tmp/mnt
```

## Development Notes

- Always use CMake, never the deprecated Makefile
- The project uses C11 for C code and C++17 for C++ code
- **This fork focuses only on CPU and CUDA backends** - other acceleration methods are not maintained
- **ATLAS implementation is the primary objective** - all changes should consider ATLAS integration
- When modifying GGML operations, test changes on both CPU and CUDA (if available)
- For ATLAS work, follow the 5-phase implementation plan in `atlas_feature.md`
- ATLAS components should be modular and backward-compatible
- Use `ggml-ci` in commit messages to trigger CI runs on draft PRs

## ATLAS Implementation

This fork's primary objective is implementing the ATLAS architecture for long-context inference:

### Key ATLAS Components
- **Deep Memory Modules**: 2-layer MLPs with residual connections instead of simple KV caches
- **Omega Rule**: Context-aware memory updates over sliding windows
- **Muon Optimizer**: Second-order optimization with Newton-Schulz iterations  
- **Feature Mapping**: Polynomial/exponential kernels for enhanced memory capacity

### ATLAS Benefits
- **Linear complexity**: O(n²) → O(w·n) where w << n for long contexts
- **Fixed memory**: ~268MB regardless of context length (vs. gigabytes for standard attention)
- **Long contexts**: Up to 10M tokens theoretical capability
- **Break-even point**: ~8K tokens where ATLAS becomes advantageous

### Implementation Details
- See `atlas_feature.md` for comprehensive architecture guide
- Focus on CPU and CUDA backends only
- Modular design for backward compatibility
- 5-phase implementation plan (10 weeks total)

### Key Files for ATLAS Work
- `atlas_feature.md` - Complete architecture specification
- `ggml/src/` - Core tensor operations (ATLAS extensions)
- `src/llama.cpp` - Inference engine integration
- `ggml/src/ggml-cuda/` - CUDA kernels for ATLAS components

### ATLAS Development Considerations
- **Research Implementation**: This is cutting-edge research - expect experimentation and iteration
- **Performance Testing**: Must validate linear scaling and memory efficiency claims
- **Long-Context Testing**: Test with contexts up to 32K+ tokens to verify ATLAS advantages
- **Numerical Stability**: Pay special attention to Newton-Schulz iterations and gradient computations
- **Backward Compatibility**: All existing llama.cpp functionality must remain intact
- **Modular Design**: ATLAS components should be compile-time optional

## GitHub Issue Resolution Workflow

When working on issues in this repository, follow this structured workflow:

### Communication Standards
- **No emojis** in any commits, issues, pull requests, or code comments
- **No subjective or hyperbolic language** - avoid words like "amazing", "perfect", "professional", "excellent", "interesting", "incredible", "fantastic", "brilliant", etc.
- **Use direct, functional language** that describes what was done, not how good/bad it is
- **Focus on technical accuracy** and clear communication of facts
- **Example transformations:**
  - Instead of: "Fixed an amazing bug" → Use: "Fixed memory leak in tensor allocation"
  - Instead of: "Excellent performance improvement" → Use: "Reduced latency by 15ms"
  - Instead of: "Perfect solution for X" → Use: "Implemented solution for X using approach Y"

### 1. Take Issue
- Pick an existing issue to work on using `$HOME/bin/gh issue list` to view available issues
- Assign yourself to the issue: `$HOME/bin/gh issue edit <number> --add-assignee @me`
- Review issue requirements and acceptance criteria thoroughly

### 2. Implement
- Create a feature branch: `git checkout -b feature/issue-<number>-<brief-description>`
- Develop the solution with focused, well-messaged commits
- Follow atomic commit principles - each commit should represent one logical change
- Use conventional commit messages: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, etc.
- **No emojis or subjective language** in commits, issues, or PRs - use direct, functional language only

### 3. Test
- Write comprehensive tests for all new code
- Ensure all existing tests pass: `ctest --test-dir build`
- Run ATLAS-specific tests if applicable: `ctest --test-dir build -L atlas`
- Validate both CPU and CUDA paths when relevant

### 4. Document
- Update relevant documentation (README, API docs, inline code comments)
- Update CLAUDE.md if adding new development patterns or workflows
- Ensure ATLAS documentation is updated for ATLAS-related changes
- Add clear docstrings for new functions and classes

### 5. Pull Request
- Open PR referencing the issue: `Fixes #<issue-number>` in PR description
- Use `$HOME/bin/gh pr create` with comprehensive description
- Include:
  - Summary of changes made
  - Testing steps and results
  - Performance impact (if applicable)
  - Screenshots/logs for UI or output changes
- Request reviews from appropriate team members

### 6. Review
- Address reviewer feedback promptly and professionally
- Push fixes as new commits (don't force-push during review)
- Re-request review after addressing all comments
- Ensure CI/CD passes before merge

### 7. Complete Issue
- **Only after PR is merged and all tests pass**
- Verify implementation meets all acceptance criteria
- Confirm all tests pass in main branch after merge
- Add completion comment with:
  - Summary of what was implemented
  - Link to merged PR
  - Test results confirmation
  - Any relevant metrics or performance data
- Close issue with: `$HOME/bin/gh issue close <number> --comment "Implementation completed. See PR #<pr-number>"`

### Example Workflow Commands
```bash
# 1. Find and take an issue
$HOME/bin/gh issue list --label "good first issue"
$HOME/bin/gh issue view 42
$HOME/bin/gh issue edit 42 --add-assignee @me

# 2. Create branch and implement
git checkout -b feature/issue-42-atlas-memory-optimization
# ... make changes with focused commits ...
git commit -m "feat: implement ATLAS memory pool optimization"
git commit -m "test: add unit tests for memory pool"

# 3. Run tests
cmake -B build -DLLAMA_BUILD_TESTS=ON -DGGML_ATLAS=ON
cmake --build build
ctest --test-dir build

# 4. Update documentation
# ... edit relevant .md files and code comments ...

# 5. Create pull request
git push -u origin feature/issue-42-atlas-memory-optimization
$HOME/bin/gh pr create \
  --title "feat: ATLAS memory pool optimization" \
  --body "Fixes #42

## Summary
Implemented memory pooling for ATLAS deep memory modules. Reduces allocation overhead by 40%.

## Changes
- Added thread-safe memory pool with configurable strategies
- Implemented memory coalescing for fragmentation reduction
- Added performance metrics collection

## Testing
- Added 15 unit tests covering all pool operations
- All existing tests pass
- Measured 40% reduction in allocation time

## Performance Impact
- Memory allocation: 40% faster
- No impact on inference speed
- Memory usage: 5% reduction due to improved packing"

# 6. Address review feedback
$HOME/bin/gh pr view --comments
# ... make requested changes ...
git commit -m "fix: address review feedback on thread safety"
git push
```

## Environment Notes

- GitHub CLI (`gh`) is located at `$HOME/bin/gh` (user: orencollaco)
- Use `$HOME/bin/gh` for GitHub operations (PRs, issues, etc.)