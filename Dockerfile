# syntax=docker/dockerfile:1.7

ARG UBUNTU_VERSION=24.04
ARG ZIG_VERSION=0.15.2
ARG BUILD_JOBS=1
ARG LLVM_REPO=https://github.com/llvm/llvm-project.git
ARG LLVM_REF=ee8c14be14deabace692ab51f5d5d432b0a83d58
ARG Z3_REPO=https://github.com/Z3Prover/z3.git
ARG Z3_REF=master
ARG SENSEI_REPO=https://github.com/oralang/sensei-monorepo.git
ARG SENSEI_REF=main

FROM ubuntu:${UBUNTU_VERSION} AS builder
ARG DEBIAN_FRONTEND=noninteractive
ARG TARGETARCH
ARG ZIG_VERSION
ARG BUILD_JOBS
ARG LLVM_REPO
ARG LLVM_REF
ARG Z3_REPO
ARG Z3_REF
ARG SENSEI_REPO
ARG SENSEI_REF

ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    CARGO_BUILD_JOBS=${BUILD_JOBS} \
    NINJAFLAGS=-j${BUILD_JOBS}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    clang \
    cmake \
    curl \
    git \
    libc++-dev \
    libc++abi-dev \
    ninja-build \
    pkg-config \
    python3 \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"
RUN curl -fsSL https://sh.rustup.rs -o /tmp/rustup-init.sh \
    && sh /tmp/rustup-init.sh -y --profile minimal \
    && rm -f /tmp/rustup-init.sh \
    && cargo --version \
    && rustc --version

RUN case "${TARGETARCH}" in \
      "amd64") zig_arch="x86_64" ;; \
      "arm64") zig_arch="aarch64" ;; \
      *) echo "unsupported TARGETARCH: ${TARGETARCH}"; exit 1 ;; \
    esac \
    && curl -fsSL "https://ziglang.org/download/${ZIG_VERSION}/zig-${zig_arch}-linux-${ZIG_VERSION}.tar.xz" -o /tmp/zig.tar.xz \
    && tar -xJf /tmp/zig.tar.xz -C /opt \
    && ln -s "/opt/zig-${zig_arch}-linux-${ZIG_VERSION}/zig" /usr/local/bin/zig \
    && zig version

WORKDIR /src
COPY . .

# Recreate vendor toolchains explicitly inside the image.
RUN set -eux; \
    rm -rf vendor/llvm-project vendor/z3 vendor/sensei; \
    mkdir -p vendor; \
    git init vendor/llvm-project; \
    git -C vendor/llvm-project remote add origin "${LLVM_REPO}"; \
    git -C vendor/llvm-project fetch --depth=1 origin "${LLVM_REF}"; \
    git -C vendor/llvm-project checkout --detach FETCH_HEAD; \
    git clone --depth=1 --branch "${Z3_REF}" "${Z3_REPO}" vendor/z3; \
    git clone --depth=1 --branch "${SENSEI_REF}" "${SENSEI_REPO}" /tmp/sensei-monorepo; \
    mkdir -p vendor/sensei; \
    cp -a /tmp/sensei-monorepo/senseic vendor/sensei/senseic; \
    rm -rf /tmp/sensei-monorepo

# Require the SIR-only workspace manifest from the selected Sensei ref.
RUN test -f vendor/sensei/senseic/sir/Cargo.toml \
    || (echo "error: missing vendor/sensei/senseic/sir/Cargo.toml in selected Sensei ref" && exit 1)

# Build Ora (includes MLIR + vendored Z3 compilation as needed).
RUN zig build -j${BUILD_JOBS}

# Build Sensei SIR CLI used by Ora bytecode emission path.
RUN cargo build --jobs ${BUILD_JOBS} --manifest-path vendor/sensei/senseic/sir/Cargo.toml -p sir-cli --release

RUN test -x zig-out/bin/ora \
    && test -x zig-out/bin/ora-lsp \
    && test -x vendor/sensei/senseic/sir/target/release/sir \
    && /src/zig-out/bin/ora --help >/dev/null \
    && /src/vendor/sensei/senseic/sir/target/release/sir --help >/dev/null

# Collect runtime shared libraries used by executables.
RUN mkdir -p /tmp/ora-runtime/lib \
    && for bin in /src/zig-out/bin/ora /src/zig-out/bin/ora-lsp /src/vendor/sensei/senseic/sir/target/release/sir; do \
        ldd "$bin" 2>/dev/null | awk '/=> \/.* \(/ {print $3} /^\// {print $1}' | sort -u | while read -r lib; do \
          [ -f "$lib" ] || continue; \
          cp -L "$lib" /tmp/ora-runtime/lib/; \
        done; \
      done

FROM ubuntu:${UBUNTU_VERSION} AS runtime
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY --from=builder /src/zig-out/bin/ora /usr/local/bin/ora
COPY --from=builder /src/zig-out/bin/ora-lsp /usr/local/bin/ora-lsp
COPY --from=builder /src/vendor/sensei/senseic/sir/target/release/sir /usr/local/bin/sir
COPY --from=builder /tmp/ora-runtime/lib/ /usr/local/lib/ora/

ENV LD_LIBRARY_PATH=/usr/local/lib/ora
ENV ORA_SENSEI_SIR=/usr/local/bin/sir

ENTRYPOINT ["/usr/local/bin/ora"]
CMD ["--help"]
