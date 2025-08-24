FROM python:3.11-slim

# Create a writable directory for Rust/Cargo
RUN mkdir -p /tmp/cargo-home /tmp/rustup-home
ENV CARGO_HOME=/tmp/cargo-home
ENV RUSTUP_HOME=/tmp/rustup-home
ENV PATH="/tmp/cargo-home/bin:${PATH}"

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    pkg-config \
    libssl-dev \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Rust in writable temp directory
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path

WORKDIR /app

# Copy and install requirements
COPY requirements.txt .

# Install with aggressive caching disabled and prefer wheels
RUN CARGO_NET_GIT_FETCH_WITH_CLI=true \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY . .

EXPOSE $PORT
CMD uvicorn api:app --host 0.0.0.0 --port $PORT