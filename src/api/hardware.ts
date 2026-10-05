import { invoke } from "@tauri-apps/api/core";
import type { HardwareInventory } from "../types/type";

/**
 * Requires a `detect_hardware` command on the Rust side that returns
 * tensorloom_core::hardware::detect_available_hardware().
 */
export function detectHardware(): Promise<HardwareInventory> {
  return invoke<HardwareInventory>("detect_hardware");
}