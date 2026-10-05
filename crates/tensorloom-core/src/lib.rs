use serde::{Deserialize, Serialize};

pub mod config;
pub mod model;
pub mod events;
pub mod train;
pub mod hardware;

#[derive(Serialize)]
pub struct SystemInfo {
    pub cpu_count: usize,
    pub total_ram: u64,
}

#[derive(Deserialize)]
pub struct RunArgs {
    pub input: String,
}

pub fn greet(name: &str) -> String {
    format!("Hello, {name}! Greeted from Rust core.")
}

pub fn system_info() -> SystemInfo {
    let mut sys = sysinfo::System::new_all();
    sys.refresh_all();
    SystemInfo {
        cpu_count: sys.cpus().len(),
        total_ram: sys.total_memory(),
    }
}

pub fn run_job(args: RunArgs) -> Result<String, String> {
    Ok(format!("ran {}", args.input))
}