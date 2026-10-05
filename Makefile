.PHONY: run build dev
#
# build:
# 	cd ui && npm ci && npm run build
# 	cargo build --release -p server
#
# run: build
# 	./target/release/server --open

# run:
# 	npm run build
# 	cargo run --release --manifest-path src-tauri/Cargo.toml
#
# dev:  # hot reload: Vite on :5173 proxying /api to the Rust server
# 	cargo run -p server & cd ui && npm run dev

build:
	npm ci
	npm run build
	cargo build --release -p server

run: build
	./target/release/server --open

dev:
	cargo run -p server & npm run dev
