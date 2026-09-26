//! Host GPU memory for `GET /v1/aivyx/residency`. No LLM backend reports
//! free or total VRAM (Ollama's `/api/ps` lists only what its own models
//! hold; `llama-server` reports nothing), so the broker reads the GPU
//! itself: `nvidia-smi` (summed across GPUs), else the AMD card with the
//! most VRAM via sysfs. Best-effort: `None` when neither answers.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

pub(crate) const MIB: u64 = 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct Vram {
    pub total_bytes: u64,
    pub used_bytes: u64,
}

/// Where to read VRAM from (`--vram-source`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum VramSource {
    /// `nvidia-smi`, else AMD sysfs.
    Auto,
    Nvidia,
    Amd,
    /// Report no VRAM.
    None,
}

/// Reads the host's VRAM. A trait so tests can fake it.
pub trait GpuProbe: Send + Sync {
    fn vram(&self) -> Option<Vram>;
}

pub struct SystemGpuProbe {
    source: VramSource,
    drm_root: PathBuf,
}

impl SystemGpuProbe {
    pub fn new(source: VramSource) -> Self {
        SystemGpuProbe {
            source,
            drm_root: PathBuf::from("/sys/class/drm"),
        }
    }
}

impl GpuProbe for SystemGpuProbe {
    fn vram(&self) -> Option<Vram> {
        match self.source {
            VramSource::None => None,
            VramSource::Nvidia => nvidia(),
            VramSource::Amd => amd(&self.drm_root),
            VramSource::Auto => nvidia().or_else(|| amd(&self.drm_root)),
        }
    }
}

/// How long `nvidia-smi` may run. It hangs when the driver is wedged or a
/// GPU has fallen off the bus; past this it is killed and VRAM is unknown.
pub const NVIDIA_SMI_TIMEOUT: Duration = Duration::from_secs(2);

fn nvidia() -> Option<Vram> {
    let mut cmd = Command::new("nvidia-smi");
    cmd.args([
        "--query-gpu=memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]);
    let out = run_with_deadline(cmd, NVIDIA_SMI_TIMEOUT)?;
    parse_nvidia_smi(&String::from_utf8_lossy(&out))
}

/// Runs `cmd` and returns its stdout if it exits successfully within
/// `timeout`. Past the deadline the child is killed and reaped, and the
/// result is `None`. Polls rather than blocking on `wait`, so a hung child
/// never pins the calling thread. Stdout is read after exit, so this suits
/// commands with small output (a full pipe would stall the child until the
/// deadline).
pub(crate) fn run_with_deadline(mut cmd: Command, timeout: Duration) -> Option<Vec<u8>> {
    use std::io::Read;
    use std::process::Stdio;
    use std::time::Instant;

    const POLL: Duration = Duration::from_millis(20);
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let deadline = Instant::now() + timeout;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(POLL),
            // Timed out, or can't tell: kill and reap.
            Ok(None) | Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    };
    if !status.success() {
        return None;
    }
    let mut out = Vec::new();
    child.stdout.take()?.read_to_end(&mut out).ok()?;
    Some(out)
}

/// One `"<total MiB>, <used MiB>"` line per GPU, summed.
pub(crate) fn parse_nvidia_smi(out: &str) -> Option<Vram> {
    let mut vram: Option<Vram> = None;
    for line in out.lines().filter(|l| !l.trim().is_empty()) {
        let (total, used) = line.split_once(',')?;
        let total = total.trim().parse::<u64>().ok()? * MIB;
        let used = used.trim().parse::<u64>().ok()? * MIB;
        let sum = vram.get_or_insert(Vram {
            total_bytes: 0,
            used_bytes: 0,
        });
        sum.total_bytes += total;
        sum.used_bytes += used;
    }
    vram
}

/// The `cardN` under `drm_root` with the largest `mem_info_vram_total` (a
/// discrete GPU beside an integrated one). Connector entries (`cardN-DP-1`)
/// and cards without the files are skipped.
pub(crate) fn amd(drm_root: &Path) -> Option<Vram> {
    std::fs::read_dir(drm_root)
        .ok()?
        .filter_map(Result::ok)
        .filter(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.starts_with("card") && !name.contains('-')
        })
        .filter_map(|e| {
            let device = e.path().join("device");
            Some(Vram {
                total_bytes: read_u64(&device.join("mem_info_vram_total"))?,
                used_bytes: read_u64(&device.join("mem_info_vram_used"))?,
            })
        })
        .max_by_key(|v| v.total_bytes)
}

fn read_u64(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvidia_smi_output_sums_every_gpu() {
        assert_eq!(
            parse_nvidia_smi("23028, 689\n"),
            Some(Vram {
                total_bytes: 23028 * MIB,
                used_bytes: 689 * MIB
            })
        );
        assert_eq!(
            parse_nvidia_smi("24564, 1000\n24564, 2000\n\n"),
            Some(Vram {
                total_bytes: 2 * 24564 * MIB,
                used_bytes: 3000 * MIB
            })
        );
        assert_eq!(parse_nvidia_smi(""), None);
        assert_eq!(parse_nvidia_smi("[N/A], [N/A]\n"), None);
    }

    fn card(root: &std::path::Path, name: &str, total: u64, used: u64) {
        let dev = root.join(name).join("device");
        std::fs::create_dir_all(&dev).unwrap();
        std::fs::write(dev.join("mem_info_vram_total"), format!("{total}\n")).unwrap();
        std::fs::write(dev.join("mem_info_vram_used"), format!("{used}\n")).unwrap();
    }

    #[test]
    fn amd_sysfs_picks_the_card_with_the_most_vram() {
        let root = tempfile::tempdir().unwrap();
        // An iGPU beside a discrete card, plus a connector entry to ignore.
        card(root.path(), "card0", 2 << 30, 1 << 30);
        card(root.path(), "card1", 16 << 30, 4 << 30);
        std::fs::create_dir_all(root.path().join("card1-DP-1")).unwrap();
        assert_eq!(
            amd(root.path()),
            Some(Vram {
                total_bytes: 16 << 30,
                used_bytes: 4 << 30
            })
        );
    }

    #[test]
    fn amd_sysfs_without_vram_files_is_none() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(root.path().join("card0").join("device")).unwrap();
        assert_eq!(amd(root.path()), None);
        assert_eq!(amd(&root.path().join("missing")), None);
    }

    #[test]
    fn a_mixed_line_with_na_makes_the_whole_probe_none() {
        assert_eq!(parse_nvidia_smi("24564, 1000\n[N/A], [N/A]\n"), None);
    }

    #[test]
    fn run_with_deadline_returns_a_quick_commands_stdout() {
        let mut cmd = Command::new("echo");
        cmd.arg("23028, 689");
        assert_eq!(
            run_with_deadline(cmd, Duration::from_secs(2)),
            Some(b"23028, 689\n".to_vec())
        );
        assert_eq!(
            run_with_deadline(Command::new("false"), Duration::from_secs(2)),
            None
        );
        assert_eq!(
            run_with_deadline(
                Command::new("/nonexistent/nvidia-smi"),
                Duration::from_secs(2)
            ),
            None
        );
    }

    #[test]
    fn run_with_deadline_kills_a_hung_command() {
        let mut cmd = Command::new("sleep");
        cmd.arg("5");
        let start = std::time::Instant::now();
        assert_eq!(run_with_deadline(cmd, Duration::from_millis(200)), None);
        let elapsed = start.elapsed();
        assert!(elapsed < Duration::from_secs(2), "took {elapsed:?}");
    }

    #[test]
    fn the_none_source_never_probes() {
        assert_eq!(SystemGpuProbe::new(VramSource::None).vram(), None);
    }
}
